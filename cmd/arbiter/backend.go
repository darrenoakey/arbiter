package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
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
//
// kind selects the remote protocol dialect:
//   - "nativ": chat against Nativ mlx-vlm-server; unload via POST /unload;
//     health/ps/embed use ollamaAddr when set (Ollama still owns embeddings).
//   - "mlx"/"" (default): legacy ollama/MLX on a single base URL.
func NewRemoteInstance(modelID, instanceID, host, addr, modelTag string, maxConcurrent int, memoryGB float64) *Instance {
	return NewRemoteInstanceWithKind(modelID, instanceID, host, addr, "", modelTag, "", maxConcurrent, memoryGB)
}

// NewRemoteInstanceWithKind is NewRemoteInstance plus explicit kind/ollamaAddr.
func NewRemoteInstanceWithKind(modelID, instanceID, host, addr, ollamaAddr, modelTag, kind string, maxConcurrent int, memoryGB float64) *Instance {
	inst := &Instance{
		ModelID:       modelID,
		InstanceID:    instanceID,
		MaxConcurrent: maxConcurrent,
		host:          host,
		state:         "stopped",
		memoryGB:      memoryGB,
		pending:       make(map[string]chan json.RawMessage),
	}
	if kind == "" {
		kind = "mlx"
	}
	inst.backend = &RemoteHTTPBackend{
		inst:       inst,
		host:       host,
		addr:       addr,
		ollamaAddr: ollamaAddr,
		modelTag:   modelTag,
		kind:       kind,
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
	inst       *Instance // owning instance; used to flip its state on Load/Unload
	host       string    // host id, e.g. "boringstack"
	addr       string    // chat base URL, e.g. http://10.0.0.42:8080 (nativ) or :11434 (ollama)
	ollamaAddr string    // optional Ollama base for embed/health/ps when chat is nativ
	modelTag   string    // remote model tag (ollama name or HF id for nativ)
	kind       string    // "nativ" | "mlx" (default)

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
	remoteEmbedDimension   = 768
	remoteEmbedMaxContext  = 8192
	remoteEmbedModelTag    = "nomic-embed-text:latest"
	remoteEmbedRepository  = "nomic-ai/nomic-embed-text-v1.5"
	remoteEmbedVersion     = "nomic-embed-text-v1.5-F16"
	remoteEmbedDType       = "float16"
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
	to := b.loadTimeout
	if to == 0 {
		to = defaultRemoteLoadTimeout
	}
	// Detached context: a slow warm must drain, not be cancelled.
	ctx, cancel := context.WithTimeout(context.Background(), to)
	defer cancel()
	var err error
	if b.inst != nil && b.inst.ModelID == "embed-text" {
		body, _, buildErr := b.buildEmbedRequest(json.RawMessage(`{"text":"warm","task":"search_document"}`))
		if buildErr != nil {
			err = buildErr
		} else if response, requestErr := b.doEmbed(ctx, body); requestErr != nil {
			err = requestErr
		} else {
			_, err = mapEmbedBodyToResult(response, 1, "search_document")
		}
	} else {
		warm := map[string]any{
			"model":      b.modelTag,
			"messages":   []map[string]string{{"role": "user", "content": "ok"}},
			"max_tokens": 1,
			"keep_alive": "10m",
		}
		body, _ := json.Marshal(warm)
		_, err = b.doChat(ctx, body)
	}
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
	if jobType == "embed-text" {
		return b.inferEmbedText(jobID, params)
	}
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
		if err != nil && shouldRetryWithoutResponseFormat(err, reqBody) {
			if stripped, changed := stripResponseFormat(reqBody); changed {
				slog.Warn("remote chat rejected with response_format present — retrying once without it",
					"host", b.host, "job", jobID, "model", b.modelTag, "error", err)
				body, err = b.doChat(ctx, stripped)
			}
		}
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

func (b *RemoteHTTPBackend) inferEmbedText(jobID string, params json.RawMessage) (*WorkerResponse, error) {
	requestBody, inputCount, err := b.buildEmbedRequest(params)
	if err != nil {
		return nil, err
	}
	task, err := embedTaskFromParams(params)
	if err != nil {
		return nil, err
	}
	timeout := b.inferTimeout
	if timeout == 0 {
		timeout = defaultRemoteInferTimeout
	}
	requestContext, cancel := context.WithTimeout(context.Background(), timeout)
	type embedResult struct {
		body []byte
		err  error
	}
	resultChannel := make(chan embedResult, 1)
	go func() {
		body, requestErr := b.doEmbed(requestContext, requestBody)
		cancel()
		resultChannel <- embedResult{body: body, err: requestErr}
	}()
	select {
	case <-b.currentCancelChan():
		return nil, errRemoteAbsent{err: fmt.Errorf("host %s flagged absent mid-request", b.host)}
	case response := <-resultChannel:
		return b.embedWorkerResponse(jobID, response.body, inputCount, task, response.err)
	}
}

func (b *RemoteHTTPBackend) embedWorkerResponse(jobID string, body []byte, inputCount int, task string, requestErr error) (*WorkerResponse, error) {
	if requestErr != nil {
		if isRemoteAbsence(requestErr) {
			return nil, errRemoteAbsent{err: requestErr}
		}
		return nil, requestErr
	}
	result, err := mapEmbedBodyToResult(body, inputCount, task)
	if err != nil {
		return nil, err
	}
	return &WorkerResponse{Status: "ok", ReqID: jobID, Result: result}, nil
}

func (b *RemoteHTTPBackend) buildEmbedRequest(params json.RawMessage) ([]byte, int, error) {
	if b.modelTag != remoteEmbedModelTag {
		return nil, 0, fmt.Errorf("embed-text remote_model_tag must be %q, got %q", remoteEmbedModelTag, b.modelTag)
	}
	texts, err := embedTextsFromParams(params)
	if err != nil {
		return nil, 0, err
	}
	task, err := embedTaskFromParams(params)
	if err != nil {
		return nil, 0, err
	}
	if err := validateEmbedModelVersion(params); err != nil {
		return nil, 0, err
	}
	for index := range texts {
		texts[index] = task + ": " + texts[index]
	}
	body, err := json.Marshal(map[string]any{
		"model": b.modelTag, "input": texts, "truncate": true, "keep_alive": "10m",
		"options": map[string]int{"num_ctx": remoteEmbedMaxContext},
	})
	if err != nil {
		return nil, 0, fmt.Errorf("encoding remote embed request: %w", err)
	}
	return body, len(texts), nil
}

func validateEmbedModelVersion(params json.RawMessage) error {
	var values map[string]any
	if err := json.Unmarshal(params, &values); err != nil || values == nil {
		return fmt.Errorf("embed-text params must be a JSON object")
	}
	value, exists := values["model_version"]
	if !exists {
		return nil
	}
	version, valid := value.(string)
	if !valid {
		return fmt.Errorf("'model_version' must be a string")
	}
	if version != remoteEmbedVersion {
		return fmt.Errorf("embed-text model_version must be %q, got %q", remoteEmbedVersion, version)
	}
	return nil
}

func embedTaskFromParams(params json.RawMessage) (string, error) {
	var values map[string]any
	if err := json.Unmarshal(params, &values); err != nil || values == nil {
		return "", fmt.Errorf("embed-text params must be a JSON object")
	}
	task := "search_document"
	if rawTask, exists := values["task"]; exists {
		var valid bool
		task, valid = rawTask.(string)
		if !valid {
			return "", fmt.Errorf("'task' must be a string")
		}
	}
	switch task {
	case "search_document", "search_query", "classification", "clustering":
		return task, nil
	default:
		return "", fmt.Errorf("invalid task %q; valid: classification, clustering, search_document, search_query", task)
	}
}

func embedTextsFromParams(params json.RawMessage) ([]string, error) {
	var values map[string]any
	if err := json.Unmarshal(params, &values); err != nil || values == nil {
		return nil, fmt.Errorf("embed-text params must be a JSON object")
	}
	if rawTexts, exists := values["texts"]; exists && rawTexts != nil {
		return validateEmbedTexts(rawTexts)
	}
	rawText, exists := values["text"]
	if !exists || rawText == nil {
		return nil, fmt.Errorf("embed-text requires 'texts' (list[string]) or 'text' (string)")
	}
	text, valid := rawText.(string)
	if !valid {
		return nil, fmt.Errorf("'text' must be a string")
	}
	return []string{text}, nil
}

func validateEmbedTexts(value any) ([]string, error) {
	items, valid := value.([]any)
	if !valid || len(items) == 0 {
		return nil, fmt.Errorf("'texts' must be a non-empty list of strings")
	}
	texts := make([]string, len(items))
	for index, item := range items {
		text, isString := item.(string)
		if !isString {
			return nil, fmt.Errorf("texts[%d] is not a string", index)
		}
		texts[index] = text
	}
	return texts, nil
}

// buildChatRequest maps arbiter chat params → an ollama/OpenAI chat request:
// forces the remote model tag, strips streaming flags (buffer-replay handles
// streaming locally), and injects a generous max_tokens when the caller omitted
// it (reasoning-model gotcha — see remoteMaxTokensDefault).
//
// response_format passes through to EVERY backend kind, nativ included, so a
// caller-supplied json_object/json_schema constraint is enforced at token
// generation by the serving engine. History: the pre-8480 Nativ generation
// (mlx-vlm-server on :8080) hard-500'd mid-generation on response_format
// ("packed token mask must be int32 with one complete row per token") and the
// field was stripped for nativ here; on 2026-08-21 that old server crashed
// outright on an adversarial json_schema request, while the current
// NativServerKit generation (nativ_server on :8480) was verified to both
// ENFORCE json_schema (schema-conformant output against a contrary prompt)
// and survive it. If a nativ host ever regresses to the old server, restore
// the strip for that host rather than losing schema enforcement everywhere.
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
	defer func() {
		if err := resp.Body.Close(); err != nil {
			slog.Debug("close remote chat response", "host", b.host, "error", err)
		}
	}()
	respBody, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return nil, readErr
	}
	if resp.StatusCode != http.StatusOK {
		// A 4xx/5xx with a body is a JOB error (bad model tag, bad params) — NOT
		// host absence. Return a typed error so the scheduler fails it terminal
		// and InferRaw can recognize a status rejection for the
		// response_format degrade retry.
		return nil, errRemoteHTTPStatus{host: b.host, code: resp.StatusCode, body: string(respBody)}
	}
	return respBody, nil
}

// errRemoteHTTPStatus is a non-2xx reply from a live remote host. It renders
// exactly like the previous plain error so log grep and scheduler
// classification are unchanged.
type errRemoteHTTPStatus struct {
	host string
	code int
	body string
}

func (e errRemoteHTTPStatus) Error() string {
	return fmt.Sprintf("remote %s returned %d: %s", e.host, e.code, e.body)
}

// stripResponseFormat returns body without its response_format key, and
// whether the key was present at all.
func stripResponseFormat(body []byte) ([]byte, bool) {
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil || m == nil {
		return body, false
	}
	if _, ok := m["response_format"]; !ok {
		return body, false
	}
	delete(m, "response_format")
	out, err := json.Marshal(m)
	if err != nil {
		return body, false
	}
	return out, true
}

// shouldRetryWithoutResponseFormat reports whether a failed chat call should
// be retried once with response_format removed: the host answered (an HTTP
// status rejection, not a transport failure) and the request actually carried
// the field. Some Nativ builds 500 mid-generation on grammar-constrained
// output ("packed token mask must be int32...") — state-dependently, under
// batch load — while the same payload without the field succeeds. The degrade
// keeps the job alive; callers that demanded JSON still get the system-prompt
// contract plus their own robust parsing.
func shouldRetryWithoutResponseFormat(err error, requestBody []byte) bool {
	var status errRemoteHTTPStatus
	if !errors.As(err, &status) {
		return false
	}
	return bytes.Contains(requestBody, []byte(`"response_format"`))
}

func (b *RemoteHTTPBackend) doEmbed(ctx context.Context, body []byte) ([]byte, error) {
	timeout := b.inferTimeout
	if timeout == 0 {
		timeout = defaultRemoteInferTimeout
	}
	if deadline, exists := ctx.Deadline(); exists {
		if remaining := time.Until(deadline); remaining > 0 {
			timeout = remaining
		}
	}
	client := newRemoteHTTPClient(timeout)
	// Embeddings stay on Ollama even when chat has moved to Nativ.
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, b.ollamaBase()+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer func() {
		if closeErr := response.Body.Close(); closeErr != nil {
			slog.Debug("close remote embed response", "host", b.host, "error", closeErr)
		}
	}()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("remote %s returned %d: %s", b.host, response.StatusCode, string(responseBody))
	}
	return responseBody, nil
}

func mapEmbedBodyToResult(body []byte, inputCount int, task string) (json.RawMessage, error) {
	var upstream struct {
		Embeddings    [][]float64 `json:"embeddings"`
		TotalDuration int64       `json:"total_duration"`
	}
	if err := json.Unmarshal(body, &upstream); err != nil {
		return nil, fmt.Errorf("decoding remote embed response: %w", err)
	}
	if len(upstream.Embeddings) == 0 {
		return nil, fmt.Errorf("remote embed response has no embeddings")
	}
	if len(upstream.Embeddings) != inputCount {
		return nil, fmt.Errorf("remote embed response count %d does not match input count %d", len(upstream.Embeddings), inputCount)
	}
	if err := validateRemoteEmbeddings(upstream.Embeddings); err != nil {
		return nil, err
	}
	result := struct {
		Embeddings      [][]float64 `json:"embeddings"`
		Dimension       int         `json:"dimension"`
		Count           int         `json:"count"`
		Task            string      `json:"task"`
		ModelRepository string      `json:"model_repository"`
		ModelVersion    string      `json:"model_version"`
		DType           string      `json:"dtype"`
		ElapsedMS       float64     `json:"elapsed_ms"`
	}{
		Embeddings: upstream.Embeddings, Dimension: remoteEmbedDimension,
		Count: len(upstream.Embeddings), Task: task,
		ModelRepository: remoteEmbedRepository, ModelVersion: remoteEmbedVersion,
		DType:     remoteEmbedDType,
		ElapsedMS: float64(upstream.TotalDuration) / float64(time.Millisecond),
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return nil, fmt.Errorf("encoding remote embed result: %w", err)
	}
	return encoded, nil
}

func validateRemoteEmbeddings(embeddings [][]float64) error {
	for index, embedding := range embeddings {
		if len(embedding) != remoteEmbedDimension {
			return fmt.Errorf("remote embed response embedding %d has dimension %d, want %d", index, len(embedding), remoteEmbedDimension)
		}
		for valueIndex, value := range embedding {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return fmt.Errorf("remote embed response embedding %d value %d is not finite", index, valueIndex)
			}
		}
	}
	return nil
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
	if err := json.Unmarshal(body, &chatResp); err != nil {
		slog.Warn("decode remote chat response", "error", err)
	}

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

// Unload best-effort drops the remote model. Ollama gets keep_alive:0 on a
// tiny chat call; Nativ gets POST /unload. Never blocks the scheduler — a
// remote unload failing just leaves the model warm.
func (b *RemoteHTTPBackend) Unload() error {
	if b.inst != nil {
		b.inst.setState("stopped")
	}
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if b.kind == "nativ" {
			if err := b.doNativUnload(ctx); err != nil {
				slog.Debug("remote nativ unload failed", "model", b.modelTag, "error", err)
			}
			return
		}
		payload, _ := json.Marshal(map[string]any{
			"model":      b.modelTag,
			"messages":   []map[string]string{{"role": "user", "content": "ok"}},
			"max_tokens": 1,
			"keep_alive": 0,
		})
		if _, err := b.doChat(ctx, payload); err != nil {
			slog.Debug("remote model unload request failed", "model", b.modelTag, "error", err)
		}
	}()
	return nil
}

// doNativUnload POSTs /unload on the Nativ server (best-effort).
func (b *RemoteHTTPBackend) doNativUnload(ctx context.Context) error {
	client := newRemoteHTTPClient(10 * time.Second)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.addr+"/unload", nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			slog.Debug("close nativ unload response", "host", b.host, "error", err)
		}
	}()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("nativ unload returned %d", resp.StatusCode)
	}
	return nil
}

// ollamaBase returns the base URL for Ollama-native routes on this backend.
func (b *RemoteHTTPBackend) ollamaBase() string {
	if b.ollamaAddr != "" {
		return b.ollamaAddr
	}
	return b.addr
}

func (b *RemoteHTTPBackend) Kill() {
	if err := b.Unload(); err != nil {
		slog.Debug("remote model kill failed", "model", b.modelTag, "error", err)
	}
}
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
