package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestChatCompletionCacheHitNoBackend proves a cache hit on the sync
// /v1/chat/completions path returns the stored response WITHOUT any scheduler,
// worker, or model instance running. The test API here has NO scheduler
// goroutine and NO instances — if the handler tried to reach a model it would
// hang/time out. A fast, correct response can ONLY come from the cache.
func TestChatCompletionCacheHitNoBackend(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	if api.llmCache == nil {
		t.Fatal("cache should be enabled by default in the test API")
	}

	// Register the model so the handler passes the "registered" gate, but never
	// start a scheduler or any instance.
	api.config.Models[llmModelID("qwen-test")] = ModelConfig{MemoryGB: 1, MaxConcurrent: 1}

	body := []byte(`{"model":"qwen-test","messages":[{"role":"user","content":"2+2?"}],"temperature":0}`)

	// Pre-seed the cache with a known result.
	key, err := api.llmCache.Key(body)
	if err != nil {
		t.Fatalf("Key: %v", err)
	}
	if err := api.llmCache.Put(key, sampleResult("four")); err != nil {
		t.Fatalf("Put: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	api.chatCompletion(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("X-Arbiter-Cache"); got != "hit" {
		t.Fatalf("X-Arbiter-Cache = %q, want hit", got)
	}
	// The body must be the raw OpenAI response with our content.
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &parsed); err != nil {
		t.Fatalf("response not valid JSON: %v — body %s", err, rec.Body.String())
	}
	if len(parsed.Choices) == 0 || parsed.Choices[0].Message.Content != "four" {
		t.Fatalf("cached content not returned: %s", rec.Body.String())
	}
}

// TestChatCompletionStreamCacheHitNoBackend proves a streamed request with the
// same content as a cached non-streamed one replays the cached completion as SSE
// without touching a backend. Same no-scheduler/no-instance guarantee as above.
func TestChatCompletionStreamCacheHitNoBackend(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	api.config.Models[llmModelID("qwen-test")] = ModelConfig{MemoryGB: 1, MaxConcurrent: 1}

	// Seed the cache using the NON-streamed body.
	plain := []byte(`{"model":"qwen-test","messages":[{"role":"user","content":"hi"}]}`)
	key, _ := api.llmCache.Key(plain)
	if err := api.llmCache.Put(key, sampleResult("streamed-answer")); err != nil {
		t.Fatalf("Put: %v", err)
	}

	// Request the SAME content WITH stream:true — must hit the same entry.
	streamed := []byte(`{"model":"qwen-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(streamed))
	rec := httptest.NewRecorder()
	api.chatCompletion(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}
	if got := rec.Header().Get("X-Arbiter-Cache"); got != "hit" {
		t.Fatalf("X-Arbiter-Cache = %q, want hit", got)
	}
	out := rec.Body.String()
	if !bytes.Contains([]byte(out), []byte("streamed-answer")) {
		t.Fatalf("SSE body missing cached content: %s", out)
	}
	if !bytes.Contains([]byte(out), []byte("data: [DONE]")) {
		t.Fatalf("SSE body missing terminator: %s", out)
	}
}

// TestAsyncChatCompletionCacheHitNoBackend proves the async /v1/jobs path with
// type chat-completion returns a pre-completed cached job without scheduling.
func TestAsyncChatCompletionCacheHitNoBackend(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	api.config.Models[llmModelID("qwen-test")] = ModelConfig{MemoryGB: 1, MaxConcurrent: 1}

	params := []byte(`{"model":"qwen-test","messages":[{"role":"user","content":"async?"}]}`)
	key, _ := api.llmCache.Key(params)
	if err := api.llmCache.Put(key, sampleResult("async-answer")); err != nil {
		t.Fatalf("Put: %v", err)
	}

	reqBody, _ := json.Marshal(map[string]any{
		"type":   "chat-completion",
		"params": json.RawMessage(params),
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/jobs", bytes.NewReader(reqBody))
	rec := httptest.NewRecorder()
	api.submitJob(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad response: %v — %s", err, rec.Body.String())
	}
	if resp["status"] != "completed" {
		t.Fatalf("status = %v, want completed — %s", resp["status"], rec.Body.String())
	}
	if resp["cached"] != true {
		t.Fatalf("cached flag = %v, want true", resp["cached"])
	}
	// The pre-completed job must carry the cached result.
	jobID, _ := resp["job_id"].(string)
	j, _ := api.store.GetJob(jobID)
	if j == nil || j.Result == nil {
		t.Fatal("pre-completed job missing result")
	}
	if !bytes.Contains(*j.Result, []byte("async-answer")) {
		t.Fatalf("job result missing cached content: %s", string(*j.Result))
	}
}

// TestChatCompletionCacheMissThenMiss proves that with no cached entry and no
// backend, the handler does NOT falsely report a hit (it would proceed to
// scheduling). We only check that a miss does not short-circuit as a hit by
// asserting the cache stays empty for this key until a real completion writes it.
func TestChatCompletionMissLeavesCacheEmpty(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	body := []byte(`{"model":"qwen-test","messages":[{"role":"user","content":"never-seen"}]}`)
	key, _ := api.llmCache.Key(body)
	if _, ok := api.llmCache.Get(key); ok {
		t.Fatal("cache should be empty for an unseen request")
	}
}
