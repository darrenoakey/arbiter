package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"sync"
	"syscall"
	"time"
)

// Backend is the execution seam behind an Instance. Today every instance runs
// as a local Python/llm subprocess (LocalProcBackend); Phase 2 will let a
// model run on a remote HTTP endpoint (RemoteHTTPBackend) without the scheduler
// caring which. The method set is exactly the lifecycle the scheduler/api call
// on an Instance today: Spawn/Load/InferRaw/GetPort/Cancel/Unload/Kill.
//
// Phase 1 contract: LocalProcBackend MOVES no behavior — it delegates byte-for-
// byte to the existing Instance methods, so models behave identically. The
// interface exists so Phase 2 can add a remote implementation; nothing routes
// through it yet (Instance still exposes its own methods, which the local
// backend wraps).
type Backend interface {
	// Spawn starts the worker if not already running (local: subprocess).
	Spawn() error
	// Load brings the model into memory and blocks until ready.
	Load(device string) error
	// InferRaw runs one inference; the caller owns slot accounting.
	InferRaw(jobID, jobType string, params json.RawMessage, outputDir string) (*WorkerResponse, error)
	// GetPort returns the HTTP port of a chat-proxy worker (llm/vllm).
	GetPort() (int, error)
	// Cancel signals an in-flight job to abort.
	Cancel() error
	// Unload releases the model (local: process-tree obliteration).
	Unload() error
	// Kill force-terminates the worker and verifies it is gone.
	Kill()
	// IsRemote reports whether this backend runs off-host. Remote backends hold
	// ZERO audited VRAM on spark — they must never enter usedGB /
	// AuditVRAMConsistency.
	IsRemote() bool
}

// NewRemoteInstance builds an Instance whose backend is a RemoteHTTPBackend on
// the given host. The instance carries the model's MaxConcurrent for slot
// accounting but holds ZERO audited VRAM (isRemote()==true via host != spark).
// memoryGB is advisory only (tracked on the host's remoteHostBudget, never in
// usedGB).
func NewRemoteInstance(modelID, instanceID, host, addr, modelTag string, maxConcurrent int, memoryGB float64) *Instance {
	inst := &Instance{
		ModelID:       modelID,
		InstanceID:    instanceID,
		MaxConcurrent: maxConcurrent,
		host:          host,
		state:         "stopped",
		memoryGB:      memoryGB,
		pending:       make(map[string]chan json.RawMessage),
	}
	inst.backend = &RemoteHTTPBackend{
		inst:     inst,
		host:     host,
		addr:     addr,
		modelTag: modelTag,
	}
	return inst
}

// LocalProcBackend is the today-path backend: a local worker subprocess managed
// by the embedded *Instance. Every method delegates to the existing Instance
// implementation so behavior is byte-identical to before the seam existed.
type LocalProcBackend struct {
	inst *Instance
}

func (b *LocalProcBackend) Spawn() error { return b.inst.Spawn() }
func (b *LocalProcBackend) Load(device string) error {
	return b.inst.Load(device)
}
func (b *LocalProcBackend) InferRaw(jobID, jobType string, params json.RawMessage, outputDir string) (*WorkerResponse, error) {
	return b.inst.InferRaw(jobID, jobType, params, outputDir)
}
func (b *LocalProcBackend) GetPort() (int, error) { return b.inst.GetPort() }
func (b *LocalProcBackend) Cancel() error         { return b.inst.Cancel() }
func (b *LocalProcBackend) Unload() error         { return b.inst.Unload() }
func (b *LocalProcBackend) Kill()                 { b.inst.Kill() }
func (b *LocalProcBackend) IsRemote() bool        { return false }

// RemoteHTTPBackend executes a model on a remote OpenAI-compatible HTTP endpoint
// (ollama/MLX on another box). It satisfies the same Backend lifecycle the
// scheduler drives, but every method talks HTTP instead of stdin/stdout to a
// local subprocess. A remote backend reports IsRemote()==true so its instance
// never touches spark's audited VRAM ledger.
//
// Two invariants are baked into this type and MUST NOT regress:
//
//  1. NEVER cancel a slow-but-alive ollama request — aborting mid-generation
//     wedges that model's runner (every later /api/chat hangs). So Infer runs
//     the upstream HTTP call on a context DETACHED from the caller's request
//     context: abandoning the client (or a failover) never cancels the upstream;
//     an abandoned call simply drains in the background (it warms the model).
//     The ONLY thing that cancels the upstream is the per-backend cancel signal
//     (Cancel()), which Phase 3's liveness poll fires on CONFIRMED host absence.
//
//  2. Per-request http.Client with an explicit DialContext + timeouts — NEVER a
//     shared keep-alive pool to a remote host. A flapped LAN route bricks a
//     pooled connection forever with EHOSTUNREACH (the documented Go net/http
//     wedge); a fresh client+conn per call dodges it.
type RemoteHTTPBackend struct {
	inst     *Instance // owning instance; used to flip its state on Load/Unload
	host     string    // host id, e.g. "boringstack"
	addr     string    // base URL of the remote backend, e.g. http://10.0.0.42:11434
	modelTag string    // remote model tag (e.g. ollama "gemma4-26b-32k")

	// loadTimeout / inferTimeout bound the upstream calls. They are generous on
	// purpose: a slow-but-alive call must DRAIN, not be cancelled (see invariant
	// 1). Failover is driven by CONFIRMED absence (dial/conn errors or the
	// liveness poll firing Cancel), never by these timeouts firing on a healthy
	// host. Zero means use the package defaults.
	loadTimeout  time.Duration
	inferTimeout time.Duration

	// absent is a per-backend cancel hook. Cancel() closes the current epoch's
	// channel; an in-flight Infer selects on it and abandons (returning an INFRA
	// error) WITHOUT cancelling the detached upstream call. Phase 3's liveness
	// poll calls Cancel when a host flips absent so failover fires in seconds
	// rather than waiting for inferTimeout (a slept laptop's socket hangs
	// silently with no RST).
	mu     sync.Mutex
	cancel chan struct{}
}

const (
	defaultRemoteLoadTimeout  = 5 * time.Minute
	defaultRemoteInferTimeout = 30 * time.Minute
	// remoteMaxTokensDefault is the generous completion budget injected when a
	// caller didn't set one. Reasoning models (gemma4) split output into hidden
	// reasoning + visible content; a small budget lets reasoning eat it all,
	// leaving content empty with finish_reason:length. 4096 keeps content
	// populated for normal chat/summary/planning replies.
	remoteMaxTokensDefault = 4096
)

// errRemoteAbsent classifies a confirmed host-absence failure: dial refused,
// EHOSTUNREACH/ENETUNREACH/connection reset, or the liveness poll firing
// Cancel. The scheduler treats this as INFRA → transparent failover (requeue
// + excluded_hosts), never a terminal job failure.
type errRemoteAbsent struct{ err error }

func (e errRemoteAbsent) Error() string { return "remote host absent: " + e.err.Error() }
func (e errRemoteAbsent) Unwrap() error { return e.err }

// isRemoteAbsence reports whether an error is a CONFIRMED host-absence (INFRA)
// failure that should trigger transparent failover, as opposed to a genuine
// job/model error (4xx/5xx body) that should fail terminal. Confirmed absence =
// the box is gone / unreachable, so there is no live queue to wedge.
func isRemoteAbsence(err error) bool {
	if err == nil {
		return false
	}
	var absent errRemoteAbsent
	if errors.As(err, &absent) {
		return true
	}
	if errors.Is(err, syscall.EHOSTUNREACH) || errors.Is(err, syscall.ENETUNREACH) ||
		errors.Is(err, syscall.ECONNREFUSED) || errors.Is(err, syscall.ECONNRESET) {
		return true
	}
	// Detached-call deadline / cancellation also means "couldn't reach a live
	// host in time" — treat as absence so the job fails over rather than dying.
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		return true
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true // dial/read network error (timeout, refused, no route)
	}
	return false
}

// newRemoteHTTPClient returns a single-use http.Client with an explicit dialer
// and DisableKeepAlives — see invariant 2. The dial timeout is short so a dead
// addr/port fails fast (conn-refused / no-route), while the overall request
// timeout is the caller-supplied generous bound.
func newRemoteHTTPClient(reqTimeout time.Duration) *http.Client {
	dialer := &net.Dialer{Timeout: 5 * time.Second, KeepAlive: -1}
	return &http.Client{
		Timeout: reqTimeout,
		Transport: &http.Transport{
			DialContext:         dialer.DialContext,
			DisableKeepAlives:   true,
			MaxIdleConns:        1,
			IdleConnTimeout:     1 * time.Second,
			TLSHandshakeTimeout: 5 * time.Second,
		},
	}
}

// currentCancelChan returns (creating if needed) the channel that the next
// Cancel() will close. Resetting it per epoch means a Cancel during one Infer
// doesn't poison the next.
func (b *RemoteHTTPBackend) currentCancelChan() chan struct{} {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.cancel == nil {
		b.cancel = make(chan struct{})
	}
	return b.cancel
}

func (b *RemoteHTTPBackend) Spawn() error { return nil } // no local process to spawn

// Load warms the remote model so the first real request isn't paying the cold
// load. It is best-effort: a tiny chat with keep_alive primes the runner. On a
// CONFIRMED absence it returns an INFRA error so the scheduler fails the load
// over to the next placement; on success it flips the instance state to loaded.
func (b *RemoteHTTPBackend) Load(device string) error {
	if b.inst != nil {
		b.inst.setState("loading")
	}
	warm := map[string]any{
		"model":      b.modelTag,
		"messages":   []map[string]string{{"role": "user", "content": "ok"}},
		"max_tokens": 1,
		"keep_alive": "10m",
	}
	body, _ := json.Marshal(warm)
	to := b.loadTimeout
	if to == 0 {
		to = defaultRemoteLoadTimeout
	}
	// Detached context: a slow warm must drain, not be cancelled.
	ctx, cancel := context.WithTimeout(context.Background(), to)
	defer cancel()
	_, err := b.doChat(ctx, body)
	if err != nil {
		if isRemoteAbsence(err) {
			if b.inst != nil {
				b.inst.setState("error")
			}
			return errRemoteAbsent{err: err}
		}
		// A non-absence error during warm (e.g. model tag typo, 4xx) is a real
		// problem but shouldn't wedge — surface it so the load fails loudly.
		if b.inst != nil {
			b.inst.setState("error")
		}
		return fmt.Errorf("remote warm-load failed: %w", err)
	}
	if b.inst != nil {
		b.inst.setState("loaded")
	}
	slog.Info("remote model warmed", "host", b.host, "model", b.modelTag, "addr", b.addr)
	return nil
}

// InferRaw runs one chat completion against the remote endpoint and returns a
// WorkerResponse whose Result matches the local llm-worker's shape exactly
// ({"format":"json","response":<full OpenAI JSON>,"text":...,"finish_reason":...,
// "usage":...}), so the rest of the pipeline (api.go non-stream, stream replay)
// is byte-for-byte indifferent to where it ran.
//
// The upstream HTTP call runs on a DETACHED context (see invariant 1). The
// caller's failover signal is the per-backend Cancel() channel, which makes
// Infer RETURN an INFRA error but leaves the upstream call draining.
func (b *RemoteHTTPBackend) InferRaw(jobID, jobType string, params json.RawMessage, outputDir string) (*WorkerResponse, error) {
	reqBody := b.buildChatRequest(params)

	to := b.inferTimeout
	if to == 0 {
		to = defaultRemoteInferTimeout
	}
	// DETACHED context — deliberately NOT derived from any client request, so
	// abandoning the client never cancels the upstream ollama call.
	ctx, cancel := context.WithTimeout(context.Background(), to)
	defer cancel()

	cancelCh := b.currentCancelChan()
	type chatResult struct {
		body []byte
		err  error
	}
	resCh := make(chan chatResult, 1)
	go func() {
		body, err := b.doChat(ctx, reqBody)
		resCh <- chatResult{body, err}
	}()

	select {
	case <-cancelCh:
		// Confirmed absence signalled externally (liveness poll). DO NOT cancel
		// the upstream — let the goroutine drain on the detached context. Return
		// INFRA so the scheduler fails the job over to the next host.
		slog.Warn("remote infer abandoned on absence signal — upstream left to drain",
			"host", b.host, "job", jobID, "model", b.modelTag)
		return nil, errRemoteAbsent{err: fmt.Errorf("host %s flagged absent mid-request", b.host)}
	case res := <-resCh:
		if res.err != nil {
			if isRemoteAbsence(res.err) {
				return nil, errRemoteAbsent{err: res.err}
			}
			// Non-absence transport error — still treat as a worker error so the
			// scheduler records it; classification happens in the scheduler.
			return nil, res.err
		}
		result := mapChatBodyToResult(res.body)
		return &WorkerResponse{Status: "ok", ReqID: jobID, Result: result}, nil
	}
}

// buildChatRequest maps arbiter chat params → an ollama/OpenAI chat request:
// forces the remote model tag, strips streaming flags (buffer-replay handles
// streaming locally), and injects a generous max_tokens when the caller omitted
// it (reasoning-model gotcha — see remoteMaxTokensDefault).
func (b *RemoteHTTPBackend) buildChatRequest(params json.RawMessage) []byte {
	var m map[string]any
	if err := json.Unmarshal(params, &m); err != nil || m == nil {
		m = map[string]any{}
	}
	m["model"] = b.modelTag
	m["stream"] = false
	delete(m, "stream_options")
	if _, ok := m["max_tokens"]; !ok {
		if _, ok := m["n_predict"]; !ok {
			m["max_tokens"] = remoteMaxTokensDefault
		}
	}
	out, _ := json.Marshal(m)
	return out
}

// doChat POSTs to {addr}/v1/chat/completions with a fresh per-request client and
// returns the raw response body, classifying transport vs HTTP-status errors.
func (b *RemoteHTTPBackend) doChat(ctx context.Context, body []byte) ([]byte, error) {
	to := b.inferTimeout
	if to == 0 {
		to = defaultRemoteInferTimeout
	}
	if dl, ok := ctx.Deadline(); ok {
		if remaining := time.Until(dl); remaining > 0 {
			to = remaining
		}
	}
	client := newRemoteHTTPClient(to)
	url := b.addr + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err // transport error (dial refused / no route / timeout)
	}
	defer resp.Body.Close()
	respBody, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return nil, readErr
	}
	if resp.StatusCode != http.StatusOK {
		// A 4xx/5xx with a body is a JOB error (bad model tag, bad params) — NOT
		// host absence. Return a plain error so the scheduler fails it terminal.
		return nil, fmt.Errorf("remote %s returned %d: %s", b.host, resp.StatusCode, string(respBody))
	}
	return respBody, nil
}

// mapChatBodyToResult converts a raw OpenAI chat-completion response body into
// the arbiter chat result shape. Critically it maps message.content (NOT the
// reasoning) into the "text" field; gemma4 splits output into message.content +
// message.reasoning, and clients want the answer, not the chain of thought. If
// content is empty but reasoning is present (small-budget edge case) it falls
// back to reasoning so the client gets *something* and a reasoning flag.
func mapChatBodyToResult(body []byte) json.RawMessage {
	var chatResp struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				Reasoning        string `json:"reasoning"`
				ReasoningContent string `json:"reasoning_content"`
				Role             string `json:"role"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	json.Unmarshal(body, &chatResp)

	result := map[string]any{
		"format":   "json",
		"response": json.RawMessage(body),
	}
	if len(chatResp.Choices) > 0 {
		msg := chatResp.Choices[0].Message
		text := msg.Content
		if text == "" {
			// Reasoning models surface the answer split out into a reasoning
			// field; if content is empty fall back to it so the client isn't
			// handed a blank reply.
			if msg.Reasoning != "" {
				text = msg.Reasoning
				result["reasoning"] = true
			} else if msg.ReasoningContent != "" {
				text = msg.ReasoningContent
				result["reasoning"] = true
			}
		}
		result["text"] = text
		result["finish_reason"] = chatResp.Choices[0].FinishReason
	}
	result["usage"] = chatResp.Usage
	out, _ := json.Marshal(result)
	return out
}

func (b *RemoteHTTPBackend) GetPort() (int, error) {
	// Remote chat is buffered+replayed locally, never proxied to a worker port.
	return 0, fmt.Errorf("remote backend has no local proxy port (buffer-replay path)")
}

// Cancel signals an in-flight Infer to ABANDON (return INFRA) WITHOUT cancelling
// the detached upstream call. Phase 3's liveness poll calls this on confirmed
// absence to make failover fire in seconds. Idempotent and epoch-based: it
// closes the current cancel channel and installs a fresh one for the next call.
func (b *RemoteHTTPBackend) Cancel() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.cancel != nil {
		select {
		case <-b.cancel:
			// already closed
		default:
			close(b.cancel)
		}
	}
	b.cancel = make(chan struct{})
	return nil
}

// Unload best-effort tells ollama to drop the model (keep_alive:0). It never
// blocks the scheduler — a remote unload failing just leaves the model warm.
func (b *RemoteHTTPBackend) Unload() error {
	if b.inst != nil {
		b.inst.setState("stopped")
	}
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		payload, _ := json.Marshal(map[string]any{
			"model":      b.modelTag,
			"messages":   []map[string]string{{"role": "user", "content": "ok"}},
			"max_tokens": 1,
			"keep_alive": 0,
		})
		b.doChat(ctx, payload) // best-effort; ignore result
	}()
	return nil
}

func (b *RemoteHTTPBackend) Kill()          { b.Unload() }
func (b *RemoteHTTPBackend) IsRemote() bool { return true }

// remoteHostBudget is the per-host advisory memory accounting for a remote
// executor. It is deliberately SEPARATE from InstanceManager.usedGB: spark's
// audited VRAM ledger stays local-CUDA-only, while remote capacity is tracked
// here as an advisory number (Phase 2 polls the host's /api/ps to keep it
// truthful). Nothing in AuditVRAMConsistency ever consults this.
type remoteHostBudget struct {
	mu       sync.Mutex
	hostID   string
	addr     string
	budgetGB float64
	usedGB   float64 // advisory; populated by Phase 2 remote dispatch/polling
}

func newRemoteHostBudget(hostID, addr string, budgetGB float64) *remoteHostBudget {
	return &remoteHostBudget{hostID: hostID, addr: addr, budgetGB: budgetGB}
}

// FreeGB returns advisory free capacity on the remote host.
func (r *remoteHostBudget) FreeGB() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.budgetGB - r.usedGB
}

// SetUsedGB records observed/advisory usage on the remote host.
func (r *remoteHostBudget) SetUsedGB(gb float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if gb < 0 {
		gb = 0
	}
	r.usedGB = gb
}
