package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// localOllama is the real ollama endpoint the Phase-2 remote tests dispatch
// against. Phase 0 set up llama3.2:3b on this Mac's local ollama; it is the
// small, fast, reliable model the spec mandates (NOT gemma — that would contend
// with other work and is slow). Host-absence is simulated with an unreachable
// addr (deadOllamaAddr), so one real endpoint + one dead addr covers routing +
// failover without depending on the flaky this-MBP backup daemon.
const (
	localOllamaAddr   = "http://127.0.0.1:11434"
	localOllamaTag    = "llama3.2:3b"
	mnemosOllamaAddr  = "http://10.0.0.42:11434"
	macminiOllamaAddr = "http://10.0.0.46:11435"
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

// reachableOllama reports whether the local ollama serves the test model. Tests
// skip (not fail) when it isn't up, so the suite stays runnable on a box without
// ollama; on this dev Mac it runs for real.
func reachableOllama(t *testing.T) bool {
	t.Helper()
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(localOllamaAddr + "/api/tags")
	if err != nil {
		t.Logf("local ollama not reachable (%v) — skipping real-remote test", err)
		return false
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			t.Logf("close local ollama response: %v", err)
		}
	}()
	if resp.StatusCode != 200 {
		t.Logf("local ollama /api/tags returned %d — skipping", resp.StatusCode)
		return false
	}
	var tags struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		t.Logf("decode local ollama tags: %v", err)
		return false
	}
	for _, m := range tags.Models {
		if m.Name == localOllamaTag {
			return true
		}
	}
	t.Logf("local ollama present but %s not pulled — skipping", localOllamaTag)
	return false
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
	if !reachableOllama(t) {
		t.Skip("local ollama unavailable")
	}
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
	if !reachableOllama(t) {
		t.Skip("local ollama unavailable")
	}
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
			"llm:gemma4-26b": {
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

	job := &Job{ID: "j1", ModelID: "llm:gemma4-26b", State: "queued"}
	got, reason := mgr.PickInstanceForJobWithReason(job, true)
	if got == nil || got.host != "boringstack" {
		t.Fatalf("spark-busy pick host=%v, want boringstack", hostOf(got))
	}
	if reason != reasonSpill {
		t.Fatalf("spark-busy placement reason=%q, want %q", reason, reasonSpill)
	}

	atomic.AddInt32(&got.activeJobs, 1)
	got.setState("loaded")
	got2, reason2 := mgr.PickInstanceForJobWithReason(&Job{ID: "j2", ModelID: "llm:gemma4-26b", State: "queued"}, true)
	if got2 == nil || got2.host != "darrens-mbp" {
		t.Fatalf("boringstack-full pick host=%v, want darrens-mbp", hostOf(got2))
	}
	if reason2 != reasonSpill {
		t.Fatalf("boringstack-full placement reason=%q, want %q", reason2, reasonSpill)
	}

	atomic.AddInt32(&got2.activeJobs, 1)
	got2.setState("loaded")
	got3, reason3 := mgr.PickInstanceForJobWithReason(&Job{ID: "j3", ModelID: "llm:gemma4-26b", State: "queued"}, true)
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
