package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func openIdempotencyTestAPI(t *testing.T, root, databasePath string) (*API, func()) {
	t.Helper()
	outputDir := filepath.Join(root, "output")
	if err := os.MkdirAll(filepath.Join(outputDir, "logs"), 0o755); err != nil {
		t.Fatal(err)
	}
	store, err := NewStore(databasePath)
	if err != nil {
		t.Fatal(err)
	}
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	cfg := &Config{VRAMBudgetGB: 100, Models: map[string]ModelConfig{
		"minimax-h3": {}, "ltx2": {},
	}}
	mgr := NewInstanceManager(cfg, "python3", root)
	scheduler := NewScheduler(cfg, store, mgr, logger, outputDir)
	api := NewAPI(cfg, store, mgr, scheduler, logger, outputDir, root)
	return api, func() { logger.Close(); store.Close(); mgr.KillAll() }
}

func TestJobIdempotencyIsConcurrentConflictSafeAndDurable(t *testing.T) {
	root := t.TempDir()
	databasePath := filepath.Join(root, "arbiter.db")
	firstAPI, closeFirst := openIdempotencyTestAPI(t, root, databasePath)
	secondAPI, closeSecond := openIdempotencyTestAPI(t, root, databasePath)
	apis := []*API{firstAPI, secondAPI}
	key := "h3-concurrent-" + genID()
	body := fmt.Sprintf(`{"type":"video-generate","model":"minimax-h3","idempotency_key":%q,"params":{"resolution":"768P","duration":4,"prompt":"shot"}}`, key)

	const requests = 24
	ids := make(chan string, requests)
	errors := make(chan string, requests)
	var group sync.WaitGroup
	for requestIndex := range requests {
		group.Add(1)
		go func(api *API) {
			defer group.Done()
			response := performRequest(api, "POST", "/v1/jobs", body)
			if response.Code != 200 {
				errors <- response.Body.String()
				return
			}
			ids <- decodeObject(t, response.Body.Bytes())["job_id"].(string)
		}(apis[requestIndex%len(apis)])
	}
	group.Wait()
	close(ids)
	close(errors)
	for failure := range errors {
		t.Fatalf("concurrent keyed request failed: %s", failure)
	}
	var jobID string
	for id := range ids {
		if jobID == "" {
			jobID = id
		} else if id != jobID {
			t.Fatalf("idempotent requests returned %q and %q", jobID, id)
		}
	}
	if got := len(firstAPI.scheduler.wake) + len(secondAPI.scheduler.wake); got != 1 {
		t.Fatalf("scheduler wakes = %d, want exactly one enqueue", got)
	}
	var persistedJobID, persistedHash string
	err := firstAPI.store.db.QueryRow(
		"SELECT job_id, request_hash FROM job_idempotency WHERE idempotency_key = ?", key,
	).Scan(&persistedJobID, &persistedHash)
	if err != nil || persistedJobID != jobID || len(persistedHash) != 64 {
		t.Fatalf("persisted keyed job id=%q hash=%q error=%v", persistedJobID, persistedHash, err)
	}

	conflictBody := fmt.Sprintf(`{"type":"video-generate","model":"minimax-h3","idempotency_key":%q,"params":{"prompt":"different","duration":4,"resolution":"768P"}}`, key)
	conflict := performRequest(secondAPI, "POST", "/v1/jobs", conflictBody)
	if conflict.Code != 409 {
		t.Fatalf("conflicting reuse status=%d body=%s", conflict.Code, conflict.Body.String())
	}
	closeFirst()
	closeSecond()

	restarted, closeRestarted := openIdempotencyTestAPI(t, root, databasePath)
	defer closeRestarted()
	replayed := performRequest(restarted, "POST", "/v1/jobs", body)
	if replayed.Code != 200 || decodeObject(t, replayed.Body.Bytes())["job_id"] != jobID {
		t.Fatalf("restart replay status=%d body=%s", replayed.Code, replayed.Body.String())
	}
	if got := len(restarted.scheduler.wake); got != 0 {
		t.Fatalf("restart replay enqueued existing job %d time(s)", got)
	}
}

func TestJobIdempotencyValidatesKeyAndNormalizesJSON(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	api.config.Models["minimax-h3"] = ModelConfig{}
	api.refreshAliasModels()
	key := "normalized-" + genID()
	first := fmt.Sprintf(`{"type":"video-generate","model":"minimax-h3","idempotency_key":%q,"params":{"prompt":"shot","duration":4,"resolution":"768P"}}`, key)
	second := fmt.Sprintf(`{"params":{"resolution":"768P","duration":4,"prompt":"shot"},"idempotency_key":%q,"model":"minimax-h3","type":"video-generate"}`, key)
	firstResponse := performRequest(api, "POST", "/v1/jobs", first)
	secondResponse := performRequest(api, "POST", "/v1/jobs", second)
	if firstResponse.Code != 200 || secondResponse.Code != 200 {
		t.Fatalf("normalized submissions failed: %s / %s", firstResponse.Body.String(), secondResponse.Body.String())
	}
	if decodeObject(t, firstResponse.Body.Bytes())["job_id"] != decodeObject(t, secondResponse.Body.Bytes())["job_id"] {
		t.Fatal("JSON field order changed normalized request identity")
	}
	for _, invalid := range []any{nil, "", "   ", string(make([]byte, maxIdempotencyKeyBytes+1))} {
		encoded, _ := json.Marshal(invalid)
		response := performRequest(api, "POST", "/v1/jobs", `{"type":"video-generate","model":"minimax-h3","idempotency_key":`+string(encoded)+`,"params":{}}`)
		if response.Code != 400 {
			t.Fatalf("invalid key status=%d body=%s", response.Code, response.Body.String())
		}
	}
}
