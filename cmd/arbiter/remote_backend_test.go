package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// localOllama is the owned SSH forward to the real Mnemos ollama endpoint that
// the Phase-2 remote tests dispatch against. Mnemos serves llama3.2:3b; it is the
// small, fast, reliable model the spec mandates (NOT gemma — that would contend
// with other work and is slow). Host-absence is simulated with an unreachable
// addr (deadOllamaAddr), so one real endpoint + one dead addr covers routing +
// failover without depending on the flaky this-MBP backup daemon.
const (
	localOllamaAddr   = "http://127.0.0.1:11434"
	localOllamaTag    = "llama3.2:3b"
	mnemosOllamaAddr  = localOllamaAddr
	macminiOllamaAddr = "http://127.0.0.1:11435"
	mnemosEmbedTag    = "nomic-embed-text:latest"
)

func TestEmbedTextsFromParamsMatchesAdapterInputs(t *testing.T) {
	tests := []struct {
		name      string
		params    string
		want      []string
		wantErr   bool
		errorPart string
	}{
		{name: "Texts", params: `{"texts":["first","second"]}`, want: []string{"first", "second"}},
		{name: "SingleText", params: `{"text":"single"}`, want: []string{"single"}},
		{name: "TextsTakePrecedence", params: `{"texts":["list"],"text":"single"}`, want: []string{"list"}},
		{name: "Missing", params: `{}`, wantErr: true, errorPart: "requires 'texts'"},
		{name: "EmptyTexts", params: `{"texts":[]}`, wantErr: true, errorPart: "non-empty list"},
		{name: "TextsWrongType", params: `{"texts":"bad"}`, wantErr: true, errorPart: "non-empty list"},
		{name: "ItemWrongType", params: `{"texts":["ok",3]}`, wantErr: true, errorPart: "texts[1]"},
		{name: "TextWrongType", params: `{"text":3}`, wantErr: true, errorPart: "must be a string"},
		{name: "InvalidJson", params: `{`, wantErr: true, errorPart: "JSON object"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := embedTextsFromParams(json.RawMessage(test.params))
			if test.wantErr {
				if err == nil || !strings.Contains(err.Error(), test.errorPart) {
					t.Fatalf("embedTextsFromParams error=%v, want containing %q", err, test.errorPart)
				}
				return
			}
			if err != nil {
				t.Fatalf("embedTextsFromParams: %v", err)
			}
			if !slices.Equal(got, test.want) {
				t.Fatalf("embedTextsFromParams=%v, want %v", got, test.want)
			}
		})
	}
}

func TestBuildEmbedRequestUsesOllamaContract(t *testing.T) {
	backend := &RemoteHTTPBackend{modelTag: mnemosEmbedTag}
	body, count, err := backend.buildEmbedRequest(json.RawMessage(`{"texts":["first","second"],"task":"search_query"}`))
	if err != nil {
		t.Fatalf("buildEmbedRequest: %v", err)
	}
	if count != 2 {
		t.Fatalf("input count=%d, want 2", count)
	}
	var request struct {
		Model     string   `json:"model"`
		Input     []string `json:"input"`
		Truncate  bool     `json:"truncate"`
		KeepAlive string   `json:"keep_alive"`
		Options   struct {
			NumContext int `json:"num_ctx"`
		} `json:"options"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if request.Model != mnemosEmbedTag || !request.Truncate || request.KeepAlive != "10m" || request.Options.NumContext != remoteEmbedMaxContext {
		t.Fatalf("request metadata=%+v", request)
	}
	if !slices.Equal(request.Input, []string{"search_query: first", "search_query: second"}) {
		t.Fatalf("request input=%v, want task-prefixed preserved order", request.Input)
	}
}

func TestBuildEmbedRequestMatchesLocalTaskContract(t *testing.T) {
	backend := &RemoteHTTPBackend{modelTag: mnemosEmbedTag}
	body, _, err := backend.buildEmbedRequest(json.RawMessage(`{"text":"fact"}`))
	if err != nil {
		t.Fatalf("default task: %v", err)
	}
	var request struct {
		Input []string `json:"input"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatalf("decode default task request: %v", err)
	}
	if !slices.Equal(request.Input, []string{"search_document: fact"}) {
		t.Fatalf("default task input=%v", request.Input)
	}
	for _, params := range []string{`{"text":"fact","task":"invalid"}`, `{"text":"fact","task":3}`} {
		if _, _, err := backend.buildEmbedRequest(json.RawMessage(params)); err == nil {
			t.Fatalf("expected invalid task error for %s", params)
		}
	}
	wrongTag := &RemoteHTTPBackend{modelTag: "nomic-embed-text:q4"}
	if _, _, err := wrongTag.buildEmbedRequest(json.RawMessage(`{"text":"fact"}`)); err == nil {
		t.Fatal("expected quantization/model-tag mismatch to fail")
	}
	if _, _, err := backend.buildEmbedRequest(json.RawMessage(`{"text":"fact","model_version":"nomic-embed-text-v1.5-Q4"}`)); err == nil {
		t.Fatal("expected model-version mismatch to fail")
	}
}

func TestBuildChatRequestKeepsResponseFormatForNativ(t *testing.T) {
	// The current NativServerKit generation (nativ_server on :8480) enforces
	// response_format natively — verified 2026-08-21 with a json_schema that
	// forced schema-conformant output against a contrary prompt, server
	// healthy afterwards. The old strip protected the pre-8480 server, which
	// hard-500'd on the field; see buildChatRequest doc comment.
	nativ := &RemoteHTTPBackend{modelTag: "mlx-community/Qwen3.6-35B-A3B-4bit", kind: "nativ"}
	body := nativ.buildChatRequest(json.RawMessage(`{
		"model":"qwen3.6-35b",
		"messages":[{"role":"user","content":"hi"}],
		"max_tokens":32,
		"temperature":0,
		"reasoning_effort":"none",
		"response_format":{"type":"json_schema","json_schema":{"name":"x","strict":true,"schema":{"type":"object"}}},
		"stream":true,
		"stream_options":{"include_usage":true}
	}`))
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got["model"] != nativ.modelTag {
		t.Fatalf("model=%v, want remote tag", got["model"])
	}
	if got["stream"] != false {
		t.Fatalf("stream=%v, want false", got["stream"])
	}
	if _, ok := got["stream_options"]; ok {
		t.Fatalf("stream_options should be stripped: %v", got["stream_options"])
	}
	rf, ok := got["response_format"].(map[string]any)
	if !ok || rf["type"] != "json_schema" {
		t.Fatalf("nativ response_format should be preserved: %v", got["response_format"])
	}
	schema, ok := rf["json_schema"].(map[string]any)
	if !ok || schema["name"] != "x" {
		t.Fatalf("json_schema payload should ride through untouched: %v", rf["json_schema"])
	}
	if got["reasoning_effort"] != "none" {
		t.Fatalf("reasoning_effort should pass through: %v", got["reasoning_effort"])
	}
	if got["max_tokens"] != float64(32) {
		t.Fatalf("max_tokens=%v, want 32", got["max_tokens"])
	}
}

func TestBuildChatRequestKeepsJSONResponseFormatForMLX(t *testing.T) {
	// Legacy ollama/MLX path is not known to 500 on response_format; leave it
	// alone so a host that understands JSON mode still gets the hint.
	mlx := &RemoteHTTPBackend{modelTag: "qwen3.6:35b-a3b", kind: "mlx"}
	body := mlx.buildChatRequest(json.RawMessage(`{
		"model":"qwen3.6-35b",
		"messages":[{"role":"user","content":"hi"}],
		"response_format":{"type":"json_object"}
	}`))
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	rf, ok := got["response_format"].(map[string]any)
	if !ok || rf["type"] != "json_object" {
		t.Fatalf("mlx response_format should be preserved: %v", got["response_format"])
	}
}

func TestMapEmbedBodyToResultValidatesAndMatchesLocalShape(t *testing.T) {
	first := testEmbedding(0.25)
	second := testEmbedding(-0.5)
	body, err := json.Marshal(map[string]any{"embeddings": [][]float64{first, second}, "model": "ignored"})
	if err != nil {
		t.Fatalf("encode upstream response: %v", err)
	}
	result, err := mapEmbedBodyToResult(body, 2, "search_query")
	if err != nil {
		t.Fatalf("mapEmbedBodyToResult: %v", err)
	}
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(result, &decoded); err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if len(decoded) != 8 || decoded["embeddings"] == nil || decoded["dimension"] == nil || decoded["count"] == nil {
		t.Fatalf("result keys=%v, want local adapter-compatible shape", decoded)
	}
	if string(decoded["dimension"]) != "768" || string(decoded["count"]) != "2" {
		t.Fatalf("result dimension=%s count=%s", decoded["dimension"], decoded["count"])
	}
	if string(decoded["task"]) != `"search_query"` || string(decoded["model_repository"]) != `"nomic-ai/nomic-embed-text-v1.5"` || string(decoded["model_version"]) != `"nomic-embed-text-v1.5-F16"` || string(decoded["dtype"]) != `"float16"` {
		t.Fatalf("result identity metadata task=%s repo=%s version=%s dtype=%s", decoded["task"], decoded["model_repository"], decoded["model_version"], decoded["dtype"])
	}
}

func TestMapEmbedBodyToResultRejectsMalformedSuccess(t *testing.T) {
	valid := testEmbedding(0.1)
	short := valid[:remoteEmbedDimension-1]
	tests := []struct {
		name      string
		body      []byte
		count     int
		errorPart string
	}{
		{name: "InvalidJson", body: []byte(`{`), count: 1, errorPart: "decoding"},
		{name: "Empty", body: []byte(`{"embeddings":[]}`), count: 1, errorPart: "no embeddings"},
		{name: "CountMismatch", body: mustEmbedBody(t, [][]float64{valid}), count: 2, errorPart: "does not match"},
		{name: "WrongDimension", body: mustEmbedBody(t, [][]float64{short}), count: 1, errorPart: "dimension 767"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := mapEmbedBodyToResult(test.body, test.count, "search_document")
			if err == nil || !strings.Contains(err.Error(), test.errorPart) {
				t.Fatalf("mapEmbedBodyToResult error=%v, want containing %q", err, test.errorPart)
			}
			if isRemoteAbsence(err) {
				t.Fatalf("malformed successful response classified as absence: %v", err)
			}
		})
	}
	valid[7] = math.NaN()
	if err := validateRemoteEmbeddings([][]float64{valid}); err == nil || !strings.Contains(err.Error(), "not finite") {
		t.Fatalf("non-finite validation error=%v", err)
	}
}

func TestRemoteEmbedTextRejectsMalformedParamsAsJobError(t *testing.T) {
	backend := &RemoteHTTPBackend{host: "mnemos", addr: mnemosOllamaAddr, modelTag: mnemosEmbedTag}
	_, err := backend.InferRaw("invalid", "embed-text", json.RawMessage(`{"texts":[]}`), "")
	if err == nil {
		t.Fatal("expected malformed params error")
	}
	if isRemoteAbsence(err) {
		t.Fatalf("malformed params classified as absence: %v", err)
	}
}

func TestRemoteEmbedTextAgainstMnemos(t *testing.T) {
	instance := NewRemoteInstance("embed-text", "mnemos-embed", "mnemos", mnemosOllamaAddr, mnemosEmbedTag, 1, 1)
	backend := instance.backend.(*RemoteHTTPBackend)
	backend.loadTimeout = 2 * time.Minute
	backend.inferTimeout = 2 * time.Minute
	if err := backend.Load(""); err != nil {
		t.Fatalf("Mnemos embed endpoint %s model %s unavailable: %v", mnemosOllamaAddr, mnemosEmbedTag, err)
	}
	inputs := []string{"arbiter routes GPU jobs", "how are memories embedded"}
	batch := inferRemoteEmbeddings(t, backend, inputs)
	if len(batch) != 2 || len(batch[0]) != remoteEmbedDimension || len(batch[1]) != remoteEmbedDimension {
		t.Fatalf("batch shape=%d x [%d,%d], want 2 x 768", len(batch), len(batch[0]), len(batch[1]))
	}
	if slices.Equal(batch[0], batch[1]) {
		t.Fatal("two distinct inputs returned identical vectors")
	}
	for index, input := range inputs {
		single := inferRemoteEmbeddings(t, backend, []string{input})
		if !slices.Equal(batch[index], single[0]) {
			t.Fatalf("batch vector %d does not match its individual input; response order changed", index)
		}
	}
}

func TestRemoteEmbedTextParityAcrossBoringstackAndMacmini(t *testing.T) {
	input := []string{"Mnemos stores durable memories with exact embedding identity."}
	boringstack := NewRemoteInstance("embed-text", "embed-boringstack", "boringstack", mnemosOllamaAddr, mnemosEmbedTag, 1, 1)
	macmini := NewRemoteInstance("embed-text", "embed-macmini", "macmini", macminiOllamaAddr, mnemosEmbedTag, 1, 1)
	left := inferRemoteEmbeddings(t, boringstack.backend.(*RemoteHTTPBackend), input)[0]
	right := inferRemoteEmbeddings(t, macmini.backend.(*RemoteHTTPBackend), input)[0]
	if similarity := cosineSimilarity(left, right); similarity < 0.99999 {
		t.Fatalf("same model/input cross-placement cosine=%0.9f, want >=0.99999", similarity)
	}
}

func cosineSimilarity(left, right []float64) float64 {
	var dot, leftNorm, rightNorm float64
	for index := range left {
		dot += left[index] * right[index]
		leftNorm += left[index] * left[index]
		rightNorm += right[index] * right[index]
	}
	return dot / (math.Sqrt(leftNorm) * math.Sqrt(rightNorm))
}

func testEmbedding(value float64) []float64 {
	embedding := make([]float64, remoteEmbedDimension)
	for index := range embedding {
		embedding[index] = value + float64(index)/10000
	}
	return embedding
}

func mustEmbedBody(t *testing.T, embeddings [][]float64) []byte {
	t.Helper()
	body, err := json.Marshal(map[string]any{"embeddings": embeddings})
	if err != nil {
		t.Fatalf("encode embed body: %v", err)
	}
	return body
}

func inferRemoteEmbeddings(t *testing.T, backend *RemoteHTTPBackend, inputs []string) [][]float64 {
	t.Helper()
	params, err := json.Marshal(map[string]any{"texts": inputs})
	if err != nil {
		t.Fatalf("encode embed params: %v", err)
	}
	response, err := backend.InferRaw("mnemos-real", "embed-text", params, "")
	if err != nil {
		t.Fatalf("Mnemos embed request to %s model %s failed: %v", mnemosOllamaAddr, mnemosEmbedTag, err)
	}
	var result struct {
		Embeddings [][]float64 `json:"embeddings"`
		Dimension  int         `json:"dimension"`
		Count      int         `json:"count"`
	}
	if err := json.Unmarshal(response.Result, &result); err != nil {
		t.Fatalf("decode remote embed result: %v", err)
	}
	if result.Count != len(inputs) || result.Dimension != remoteEmbedDimension {
		t.Fatalf("result count=%d dimension=%d, want count=%d dimension=768", result.Count, result.Dimension, len(inputs))
	}
	return result.Embeddings
}

// requireReachableOllama requires the owned Mnemos forward and the live model.
func requireReachableOllama(t *testing.T) {
	t.Helper()
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(localOllamaAddr + "/api/tags")
	if err != nil {
		t.Fatalf("owned Mnemos forward is not reachable: %v", err)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			t.Logf("close local ollama response: %v", err)
		}
	}()
	if resp.StatusCode != 200 {
		t.Fatalf("owned Mnemos forward /api/tags returned %d", resp.StatusCode)
	}
	var tags struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		t.Fatalf("decode Mnemos tags: %v", err)
	}
	for _, m := range tags.Models {
		if m.Name == localOllamaTag {
			return
		}
	}
	t.Fatalf("Mnemos is reachable but required model %s is absent", localOllamaTag)
}

// deadOllamaAddr returns an addr that will refuse/never-route — a TCP listener
// bound and immediately closed leaves a port nothing listens on, so a dial gets
// connection-refused (a CONFIRMED-absence INFRA error). OS-assigned port avoids
// collisions.
func deadOllamaAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve dead port: %v", err)
	}
	addr := ln.Addr().String()
	if err := ln.Close(); err != nil { // nothing listens now → dial refused
		t.Fatalf("close dead-port listener: %v", err)
	}
	return "http://" + addr
}

func newRemoteTestScheduler(t *testing.T, cfg *Config) (*Scheduler, *Store, *InstanceManager, func()) {
	t.Helper()
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	store.InitDedup()
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	mgr := NewInstanceManager(cfg, "python3", projectRoot)
	setupInstances(cfg, mgr, "python3", projectRoot)
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)
	cleanup := func() {
		store.Close()
		logger.Close()
	}
	return sched, store, mgr, cleanup
}

func chatPayload(prompt string) json.RawMessage {
	body, _ := json.Marshal(map[string]any{
		"messages":   []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens": 32,
	})
	return body
}

// waitForState polls the store until the job reaches one of the wanted states or
// the deadline elapses.
func waitForState(t *testing.T, store *Store, jobID string, deadline time.Duration, want ...string) *Job {
	t.Helper()
	end := time.Now().Add(deadline)
	for time.Now().Before(end) {
		j, _ := store.GetJob(jobID)
		if j != nil {
			for _, w := range want {
				if j.State == w {
					return j
				}
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	j, _ := store.GetJob(jobID)
	if j != nil {
		t.Fatalf("job %s did not reach %v within %s (state=%s error=%q)", jobID, want, deadline, j.State, j.Error)
	}
	t.Fatalf("job %s vanished waiting for %v", jobID, want)
	return nil
}

func pi() *float64 { p := 1.0; return &p }

// (a) A chat job routed to a REMOTE instance returns a real completion.
func TestRemoteDispatchReturnsRealCompletion(t *testing.T) {
	requireReachableOllama(t)
	cfg := &Config{
		VRAMBudgetGB: 100,
		Hosts: map[string]HostConfig{
			"mac": {Addr: localOllamaAddr, Kind: "mlx", BudgetGB: 64},
		},
		Models: map[string]ModelConfig{
			"chat": {
				MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
				Placements:    []string{"mac"},
				AdapterParams: map[string]string{"remote_model_tag": localOllamaTag},
			},
		},
	}
	sched, store, mgr, cleanup := newRemoteTestScheduler(t, cfg)
	defer cleanup()

	// The single instance must be remote.
	inst := mgr.GetModelInstances("chat")[0]
	if !inst.isRemote() {
		t.Fatalf("expected remote instance, host=%s", inst.host)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go sched.Run(ctx)

	job, err := store.CreateJob("chat", "chat-completion", chatPayload("Reply with exactly the word PONG."), 1)
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	sched.Wake()

	done := waitForState(t, store, job.ID, 90*time.Second, "completed", "failed")
	if done.State != "completed" {
		t.Fatalf("job state=%s error=%q, want completed", done.State, done.Error)
	}
	var result map[string]any
	if err := json.Unmarshal(*done.Result, &result); err != nil {
		t.Fatalf("decode completed result: %v", err)
	}
	text, _ := result["text"].(string)
	if text == "" {
		t.Fatalf("completed job had empty text; result=%s", string(*done.Result))
	}
	if _, ok := result["response"]; !ok {
		t.Fatalf("result missing full OpenAI response field; result=%s", string(*done.Result))
	}
	t.Logf("remote completion text=%q", text)
}

// (b) CONFIRMED host-absence on the preferred host → the job transparently fails
// over to a reachable endpoint, NEVER fails, exactly one result.
func TestRemoteFailoverOnAbsenceCompletesElsewhere(t *testing.T) {
	requireReachableOllama(t)
	dead := deadOllamaAddr(t)
	cfg := &Config{
		VRAMBudgetGB: 100,
		Hosts: map[string]HostConfig{
			"dead": {Addr: dead, Kind: "mlx", BudgetGB: 64},
			"live": {Addr: localOllamaAddr, Kind: "mlx", BudgetGB: 64},
		},
		Models: map[string]ModelConfig{
			"chat": {
				MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
				// Preferred host is unreachable; failover must spill to "live".
				Placements:    []string{"dead", "live"},
				AdapterParams: map[string]string{"remote_model_tag": localOllamaTag},
			},
		},
	}
	sched, store, _, cleanup := newRemoteTestScheduler(t, cfg)
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go sched.Run(ctx)

	job, err := store.CreateJob("chat", "chat-completion", chatPayload("Reply with the word OK."), 1)
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	sched.Wake()

	done := waitForState(t, store, job.ID, 90*time.Second, "completed", "failed")
	if done.State != "completed" {
		t.Fatalf("job state=%s error=%q, want completed (failover should never fail it)", done.State, done.Error)
	}
	// The dead host must be in the durable excluded set.
	if !done.HostExcluded("dead") {
		t.Fatalf("expected 'dead' in excluded_hosts, got %v", done.ExcludedHosts)
	}
	// Exactly one result: it's completed with a result and a single finished_at.
	if done.Result == nil {
		t.Fatalf("completed job has no result")
	}
	if done.FinishedAt == nil {
		t.Fatalf("completed job has no finished_at")
	}
	var result map[string]any
	if err := json.Unmarshal(*done.Result, &result); err != nil {
		t.Fatalf("decode completed result: %v", err)
	}
	if text, _ := result["text"].(string); text == "" {
		t.Fatalf("failover completion had empty text; result=%s", string(*done.Result))
	}
	t.Logf("failover: excluded=%v final_text=%v", done.ExcludedHosts, result["text"])
}

// (c) PickInstanceForJob honors excluded_hosts and the placement order.
func TestPickInstanceForJobHonorsExcludedHosts(t *testing.T) {
	cfg := &Config{
		VRAMBudgetGB: 100,
		Hosts: map[string]HostConfig{
			"h1": {Addr: "http://10.255.255.1:11434", Kind: "mlx", BudgetGB: 64},
			"h2": {Addr: "http://10.255.255.2:11434", Kind: "mlx", BudgetGB: 64},
		},
		Models: map[string]ModelConfig{
			"chat": {
				MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
				Placements:    []string{"h1", "h2", "spark"},
			},
		},
	}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())

	// No exclusions: must pick the most-preferred host h1.
	job := &Job{ID: "j1", ModelID: "chat", State: "queued"}
	got := mgr.PickInstanceForJob(job, true)
	if got == nil || got.host != "h1" {
		t.Fatalf("no-exclusion pick host=%v, want h1", hostOf(got))
	}

	// Exclude h1: must spill to h2.
	job.ExcludedHosts = []string{"h1"}
	got = mgr.PickInstanceForJob(job, true)
	if got == nil || got.host != "h2" {
		t.Fatalf("exclude-h1 pick host=%v, want h2", hostOf(got))
	}

	// Exclude h1+h2: must fall back to spark (local, always reachable).
	job.ExcludedHosts = []string{"h1", "h2"}
	got = mgr.PickInstanceForJob(job, true)
	if got == nil || got.host != "spark" {
		t.Fatalf("exclude-both pick host=%v, want spark", hostOf(got))
	}

	// Kill-switch (remoteEnabled=false): remotes skipped, pin to spark even with
	// no exclusions.
	job.ExcludedHosts = nil
	got = mgr.PickInstanceForJob(job, false)
	if got == nil || got.host != "spark" {
		t.Fatalf("kill-switch pick host=%v, want spark", hostOf(got))
	}
}

func TestPickInstanceForJobSpillsFromBusyLocalToRemote(t *testing.T) {
	cfg := &Config{
		VRAMBudgetGB: 100,
		Hosts: map[string]HostConfig{
			"boringstack": {Addr: "http://10.255.255.1:11434", Kind: "mlx", BudgetGB: 96},
			"darrens-mbp": {Addr: "http://10.255.255.2:11434", Kind: "mlx", BudgetGB: 40},
		},
		Models: map[string]ModelConfig{
			"ltx2-dev-denoise1": {
				MemoryGB: 80, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
			},
			"llm:chat-spill-test": {
				MemoryGB: 10, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
				Placements:    []string{"spark", "boringstack", "darrens-mbp"},
			},
		},
	}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())
	mgr.SetReachabilityFunc(func(string) bool { return true })

	markLoaded(t, mgr, "ltx2-dev-denoise1")
	blocker := mgr.GetModelInstances("ltx2-dev-denoise1")[0]
	atomic.AddInt32(&blocker.activeJobs, 1)

	job := &Job{ID: "j1", ModelID: "llm:chat-spill-test", State: "queued"}
	got, reason := mgr.PickInstanceForJobWithReason(job, true)
	if got == nil || got.host != "boringstack" {
		t.Fatalf("spark-busy pick host=%v, want boringstack", hostOf(got))
	}
	if reason != reasonSpill {
		t.Fatalf("spark-busy placement reason=%q, want %q", reason, reasonSpill)
	}

	atomic.AddInt32(&got.activeJobs, 1)
	got.setState("loaded")
	got2, reason2 := mgr.PickInstanceForJobWithReason(&Job{ID: "j2", ModelID: "llm:chat-spill-test", State: "queued"}, true)
	if got2 == nil || got2.host != "darrens-mbp" {
		t.Fatalf("boringstack-full pick host=%v, want darrens-mbp", hostOf(got2))
	}
	if reason2 != reasonSpill {
		t.Fatalf("boringstack-full placement reason=%q, want %q", reason2, reasonSpill)
	}

	atomic.AddInt32(&got2.activeJobs, 1)
	got2.setState("loaded")
	got3, reason3 := mgr.PickInstanceForJobWithReason(&Job{ID: "j3", ModelID: "llm:chat-spill-test", State: "queued"}, true)
	if got3 == nil || got3.host != "spark" {
		t.Fatalf("all-remotes-full pick host=%v, want spark fallback", hostOf(got3))
	}
	if reason3 != reasonPreferred {
		t.Fatalf("all-remotes-full placement reason=%q, want %q", reason3, reasonPreferred)
	}
}

func hostOf(inst *Instance) string {
	if inst == nil {
		return "<nil>"
	}
	return inst.host
}

// (d) Idempotency: a late/duplicate response from a dead host must NOT double-
// write a result or resurrect a job that already reached a terminal state. The
// guard is in store.UpdateState (terminal-stays-terminal) — a late writer can't
// move completed→running, and a late result write to an already-completed job
// is rejected so the first result wins.
func TestLateResponseDoesNotResurrectTerminalJob(t *testing.T) {
	projectRoot := t.TempDir()
	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()
	store.InitDedup()

	job, err := store.CreateJob("chat", "chat-completion", chatPayload("hi"), 1)
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	// Simulate the job completing on the failover target (the real, first result).
	winner := json.RawMessage(`{"format":"json","text":"WINNER"}`)
	if err := store.UpdateState(job.ID, "completed", WithResult(winner), WithFinishedAt(nowTS())); err != nil {
		t.Fatalf("complete: %v", err)
	}

	// A LATE response from the originally-dispatched (dead) host tries to write a
	// second result and/or move the job back to running. Both must be rejected.
	loser := json.RawMessage(`{"format":"json","text":"LATE_LOSER"}`)
	if err := store.UpdateState(job.ID, "running"); err != nil { // terminal→active: rejected
		t.Fatalf("attempt terminal-to-running update: %v", err)
	}
	if err := store.UpdateState(job.ID, "completed", WithResult(loser)); err != nil { // overwrite attempt
		t.Fatalf("attempt terminal overwrite: %v", err)
	}
	if err := store.UpdateState(job.ID, "queued", WithClearStartedAt()); err != nil { // failover-requeue race
		t.Fatalf("attempt terminal requeue: %v", err)
	}

	final, _ := store.GetJob(job.ID)
	if final.State != "completed" {
		t.Fatalf("terminal job moved to %s — terminal-stays-terminal violated", final.State)
	}
	var result map[string]any
	if err := json.Unmarshal(*final.Result, &result); err != nil {
		t.Fatalf("decode final result: %v", err)
	}
	if result["text"] != "WINNER" {
		t.Fatalf("result overwritten by late response: got %v, want WINNER", result["text"])
	}
}

// (d-extension) AddExcludedHost is durable and idempotent — the persistence half
// of failover. A re-add is a no-op; the set survives a re-read from disk.
func TestAddExcludedHostDurableAndIdempotent(t *testing.T) {
	projectRoot := t.TempDir()
	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()
	store.InitDedup()

	job, _ := store.CreateJob("chat", "chat-completion", chatPayload("hi"), 1)

	hosts, err := store.AddExcludedHost(job.ID, "boringstack")
	if err != nil {
		t.Fatalf("add: %v", err)
	}
	if len(hosts) != 1 || hosts[0] != "boringstack" {
		t.Fatalf("add returned %v, want [boringstack]", hosts)
	}
	// Idempotent re-add.
	hosts, _ = store.AddExcludedHost(job.ID, "boringstack")
	if len(hosts) != 1 {
		t.Fatalf("re-add grew set to %v", hosts)
	}
	// Second distinct host appends.
	hosts, _ = store.AddExcludedHost(job.ID, "mbp")
	if len(hosts) != 2 {
		t.Fatalf("second add gave %v, want 2 entries", hosts)
	}
	// Durable: re-read from store reflects both.
	reread, _ := store.GetJob(job.ID)
	if !reread.HostExcluded("boringstack") || !reread.HostExcluded("mbp") {
		t.Fatalf("excluded set not persisted: %v", reread.ExcludedHosts)
	}
}

// errorClassification documents the INFRA-vs-JOB split that drives failover.
func TestRemoteAbsenceClassification(t *testing.T) {
	cases := []struct {
		name   string
		err    error
		absent bool
	}{
		{"wrapped-absent", errRemoteAbsent{err: fmt.Errorf("boom")}, true},
		{"job-4xx", fmt.Errorf("remote mac returned 400: bad model"), false},
		{"nil", nil, false},
	}
	for _, c := range cases {
		if got := isRemoteAbsence(c.err); got != c.absent {
			t.Errorf("%s: isRemoteAbsence=%v want %v", c.name, got, c.absent)
		}
	}
}

func TestStripResponseFormatRemovesOnlyThatKey(t *testing.T) {
	body := []byte(`{"model":"m","messages":[],"response_format":{"type":"json_object"},"max_tokens":8}`)
	stripped, changed := stripResponseFormat(body)
	if !changed {
		t.Fatal("expected response_format to be reported present")
	}
	var got map[string]any
	if err := json.Unmarshal(stripped, &got); err != nil {
		t.Fatalf("decode stripped body: %v", err)
	}
	if _, ok := got["response_format"]; ok {
		t.Fatalf("response_format survived the strip: %v", got)
	}
	if got["model"] != "m" || got["max_tokens"] != float64(8) {
		t.Fatalf("unrelated fields must survive: %v", got)
	}
	same, changed := stripResponseFormat([]byte(`{"model":"m"}`))
	if changed || string(same) != `{"model":"m"}` {
		t.Fatalf("body without the key must pass through unchanged: %s changed=%v", same, changed)
	}
}

func TestShouldRetryWithoutResponseFormatClassification(t *testing.T) {
	withFormat := []byte(`{"response_format":{"type":"json_schema"}}`)
	statusErr := errRemoteHTTPStatus{host: "h", code: 500, body: "packed token mask must be int32"}
	if !shouldRetryWithoutResponseFormat(statusErr, withFormat) {
		t.Fatal("status rejection with response_format present must retry")
	}
	if shouldRetryWithoutResponseFormat(statusErr, []byte(`{"model":"m"}`)) {
		t.Fatal("no response_format in the request — nothing to degrade")
	}
	if shouldRetryWithoutResponseFormat(fmt.Errorf("dial tcp: connect refused"), withFormat) {
		t.Fatal("transport errors must not trigger the degrade retry")
	}
}

func TestInferRawRetriesOnceWithoutResponseFormatOnStatusRejection(t *testing.T) {
	// The state-dependent Nativ failure: a request carrying response_format
	// 500s mid-generation ("packed token mask must be int32...") while the
	// identical payload without the field succeeds. InferRaw must degrade to a
	// single format-free retry instead of failing the job.
	var requests [][]byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		requests = append(requests, body)
		if bytes.Contains(body, []byte(`"response_format"`)) {
			w.WriteHeader(500)
			if _, err := w.Write([]byte(`{"detail":"Generation failed: packed token mask must be int32 with one complete row per token"}`)); err != nil {
				t.Errorf("write 500 body: %v", err)
			}
			return
		}
		if _, err := w.Write([]byte(`{"choices":[{"message":{"content":"{\"ok\":true}","role":"assistant"},"finish_reason":"stop"}],"usage":{"completion_tokens":3}}`)); err != nil {
			t.Errorf("write completion: %v", err)
		}
	}))
	defer server.Close()

	backend := &RemoteHTTPBackend{modelTag: "mlx-community/Qwen3.6-35B-A3B-4bit", kind: "nativ", host: "test", addr: server.URL}
	response, err := backend.InferRaw("job-1", "chat-completion", json.RawMessage(`{
		"model":"local-extract",
		"messages":[{"role":"user","content":"hi"}],
		"max_tokens":32,
		"response_format":{"type":"json_schema","json_schema":{"name":"x","strict":true,"schema":{"type":"object"}}}
	}`), "")
	if err != nil {
		t.Fatalf("InferRaw should succeed via the degrade retry: %v", err)
	}
	if response.Status != "ok" {
		t.Fatalf("status=%q, want ok", response.Status)
	}
	if len(requests) != 2 {
		t.Fatalf("expected exactly 2 upstream requests (rejected + degraded), got %d", len(requests))
	}
	if !bytes.Contains(requests[0], []byte(`"response_format"`)) {
		t.Fatal("first request must carry response_format")
	}
	if bytes.Contains(requests[1], []byte(`"response_format"`)) {
		t.Fatal("retry must not carry response_format")
	}
	var result map[string]any
	if err := json.Unmarshal(response.Result, &result); err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if result["text"] != `{"ok":true}` {
		t.Fatalf("degraded completion text=%v", result["text"])
	}
}

// TestDoChatSendsApiKeyWhenConfigured is the regression test for the second
// half of the 2026-08-29 boringstack incident: the FIRST fix only added the
// Authorization header to health/unload management calls, assuming Nativ
// chat completions itself never checks the key (true for the mlx_vlm
// version installed locally, but boringstack's deployed instance gates
// chat completions too). Without the header on the actual chat request,
// jobs kept 401ing even after the host was correctly detected as reachable.
// Proves doChat/InferRaw sends "Authorization: Bearer <key>" on every nativ
// chat request when RemoteHTTPBackend.apiKey is set.
func TestDoChatSendsApiKeyWhenConfigured(t *testing.T) {
	const key = "boringstack-secret"
	var gotAuth []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = append(gotAuth, r.Header.Get("Authorization"))
		if r.Header.Get("Authorization") != "Bearer "+key {
			w.WriteHeader(401)
			if _, err := w.Write([]byte(`{"detail":"Invalid API key"}`)); err != nil {
				t.Errorf("write 401 body: %v", err)
			}
			return
		}
		if _, err := w.Write([]byte(`{"choices":[{"message":{"content":"pong","role":"assistant"},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}`)); err != nil {
			t.Errorf("write completion: %v", err)
		}
	}))
	defer server.Close()

	backend := &RemoteHTTPBackend{modelTag: "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit", kind: "nativ", host: "boringstack", addr: server.URL, apiKey: key}
	response, err := backend.InferRaw("job-1", "chat-completion", json.RawMessage(`{
		"model":"local-chat",
		"messages":[{"role":"user","content":"ping"}],
		"max_tokens":8
	}`), "")
	if err != nil {
		t.Fatalf("InferRaw should succeed once the api key is sent: %v", err)
	}
	if response.Status != "ok" {
		t.Fatalf("status=%q, want ok", response.Status)
	}
	if len(gotAuth) == 0 || gotAuth[0] != "Bearer "+key {
		t.Fatalf("expected 'Bearer %s' Authorization header, got %v", key, gotAuth)
	}
}
