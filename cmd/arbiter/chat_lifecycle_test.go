package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestChatCompletionDisconnectCancelsOnlyAbandonedQueuedJob(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	job, err := api.store.CreateJobWithRequestedModel("llm:test", "chat-completion", json.RawMessage(`{"model":"qwen","messages":[]}`), 0, "test")
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "http://test/v1/chat/completions", nil)
	ctx, cancel := context.WithCancel(req.Context())
	cancel()
	api.waitForChatCompletion(httptest.NewRecorder(), req.WithContext(ctx), job.ID, "", "test", "llm:test", "", time.Hour)
	stored, err := api.store.GetJob(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if stored.State != "cancelled" || stored.Error != "synchronous client disconnected" {
		t.Fatalf("state=%q error=%q", stored.State, stored.Error)
	}
}

func TestChatCompletionTimeoutCancelsAbandonedJobAndPreservesRunningOutcome(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	queued, _ := api.store.CreateJobWithRequestedModel("llm:test", "chat-completion", json.RawMessage(`{"model":"qwen","messages":[]}`), 0, "test")
	response := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "http://test/v1/chat/completions", nil)
	api.waitForChatCompletion(response, req, queued.ID, "", "test", "llm:test", "", time.Millisecond)
	if response.Code != http.StatusGatewayTimeout {
		t.Fatalf("status=%d", response.Code)
	}
	stored, _ := api.store.GetJob(queued.ID)
	if stored.State != "cancelled" || stored.Error != "synchronous response wait timed out" {
		t.Fatalf("state=%q error=%q", stored.State, stored.Error)
	}

	running, _ := api.store.CreateJobWithRequestedModel("llm:test", "chat-completion", json.RawMessage(`{"model":"qwen","messages":[]}`), 0, "test")
	if err := api.store.UpdateState(running.ID, "running", WithStartedAt(nowTS())); err != nil {
		t.Fatal(err)
	}
	api.waitForChatCompletion(httptest.NewRecorder(), req, running.ID, "", "test", "llm:test", "", time.Millisecond)
	preserved, _ := api.store.GetJob(running.ID)
	if preserved.State != "running" {
		t.Fatalf("running job changed to %q", preserved.State)
	}
}

func TestDurableCorrelatedJobSurvivesTimeoutForReconnectAndDeduplication(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	payload := json.RawMessage(`{"model":"qwen","messages":[]}`)
	job, created, conflict, err := api.store.CreateIdempotentJob("llm:test", "chat-completion", payload, 0, "test", "agentd3:test-durable", "request-hash")
	if err != nil || !created || conflict {
		t.Fatalf("create durable: created=%v conflict=%v err=%v", created, conflict, err)
	}
	req := httptest.NewRequest(http.MethodPost, "http://test/v1/chat/completions", nil)
	api.waitForChatCompletion(httptest.NewRecorder(), req, job.ID, "", "test", "llm:test", "", time.Millisecond)
	preserved, _ := api.store.GetJob(job.ID)
	if preserved.State != "queued" {
		t.Fatalf("durable job state=%q", preserved.State)
	}

	result := json.RawMessage(`{"response":{"choices":[{"message":{"content":"retrieved"}}]}}`)
	if err := api.store.UpdateState(job.ID, "completed", WithResult(result), WithFinishedAt(nowTS())); err != nil {
		t.Fatal(err)
	}
	retrieved := performRequest(api, http.MethodGet, "/v1/jobs/"+job.ID, "")
	if retrieved.Code != http.StatusOK {
		t.Fatalf("reconnect status=%d", retrieved.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(retrieved.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["status"] != "completed" || body["result"] == nil {
		t.Fatalf("reconnect body=%v", body)
	}

	duplicate, created, conflict, err := api.store.CreateIdempotentJob("llm:test", "chat-completion", payload, 0, "test", "agentd3:test-durable", "request-hash")
	if err != nil || created || conflict || duplicate.ID != job.ID {
		t.Fatalf("dedup: id=%q created=%v conflict=%v err=%v", duplicate.ID, created, conflict, err)
	}
}
