package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// TestGetJobCanonicalResolvesArtifact proves a dedup-cache-hit job (whose
// canonical_job_id points at the original) synthesizes result_path — and
// inlines data — from the ORIGINAL job's output dir, not from its own
// never-created dir. This is the 2026-09-14 ltx25-denoise1 incident: every
// cache hit "completed" with an unresolvable result_path.
func TestGetJobCanonicalResolvesArtifact(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	orig, err := api.store.CreateJob("ltx25-denoise1", "ltx25-denoise1", json.RawMessage(`{"chunk":0}`), 1)
	if err != nil {
		t.Fatalf("create orig: %v", err)
	}
	origDir := resolveJobDir(api.config, api.outputDir, orig.ID)
	if err := os.MkdirAll(origDir, 0o755); err != nil {
		t.Fatalf("mkdir orig dir: %v", err)
	}
	payload := []byte("denoised-video-bytes")
	if err := os.WriteFile(filepath.Join(origDir, "result.mp4"), payload, 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	origResult := json.RawMessage(`{"file":"result.mp4","format":"mp4"}`)
	if err := api.store.UpdateState(orig.ID, "completed", WithResult(origResult), WithFinishedAt(nowTS())); err != nil {
		t.Fatalf("complete orig: %v", err)
	}

	hit, err := api.store.CreateJob("ltx25-denoise1", "ltx25-denoise1", json.RawMessage(`{"chunk":0}`), 0)
	if err != nil {
		t.Fatalf("create cache-hit job: %v", err)
	}
	if err := api.store.SetCanonicalJobID(hit.ID, orig.ID); err != nil {
		t.Fatalf("set canonical: %v", err)
	}
	if err := api.store.UpdateState(hit.ID, "completed", WithResult(origResult), WithFinishedAt(nowTS())); err != nil {
		t.Fatalf("complete cache-hit job: %v", err)
	}

	rec := performRequest(api, http.MethodGet, "/v1/jobs/"+hit.ID, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	resultMap, ok := body["result"].(map[string]any)
	if !ok {
		t.Fatalf("result type %T", body["result"])
	}
	resultPath, _ := resultMap["result_path"].(string)
	if resultPath == "" {
		t.Fatal("result_path missing")
	}
	if filepath.Dir(resultPath) != origDir {
		t.Fatalf("result_path dir = %q, want orig dir %q", filepath.Dir(resultPath), origDir)
	}
	data, ok := resultMap["data"].(string)
	if !ok || data == "" {
		t.Fatalf("inline data missing/empty on canonical hit: %v", resultMap["data"])
	}
}

// TestSubmitDedupSkipsStaleArtifact proves that when the original job's
// on-disk artifact has vanished, an identical submission does NOT get an
// instant bodyless cache hit — it is re-run for real. With the artifact
// present (positive control) the cache hit is served.
func TestSubmitDedupSkipsStaleArtifact(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	api.store.InitDedup()

	api.config.Models["ltx25-encode"] = ModelConfig{MemoryGB: 1, MaxConcurrent: 1}

	submit := func() map[string]any {
		t.Helper()
		reqBody, _ := json.Marshal(map[string]any{
			"type":   "ltx25-encode",
			"params": json.RawMessage(`{"chunk":0}`),
		})
		req := httptest.NewRequest(http.MethodPost, "/v1/jobs", bytes.NewReader(reqBody))
		rec := httptest.NewRecorder()
		api.submitJob(rec, req)
		if rec.Code != http.StatusOK && rec.Code != http.StatusAccepted {
			t.Fatalf("submit status = %d body = %s", rec.Code, rec.Body.String())
		}
		var resp map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode submit response: %v — %s", err, rec.Body.String())
		}
		return resp
	}

	first := submit()
	origID, _ := first["job_id"].(string)
	if origID == "" {
		t.Fatalf("no job_id in %v", first)
	}

	// Complete the original with a file-backed result and a real artifact.
	origDir := resolveJobDir(api.config, api.outputDir, origID)
	if err := os.MkdirAll(origDir, 0o755); err != nil {
		t.Fatalf("mkdir orig dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(origDir, "result.mp4"), []byte("x"), 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	origResult := json.RawMessage(`{"file":"result.mp4","format":"mp4"}`)
	if err := api.store.UpdateState(origID, "completed", WithResult(origResult), WithFinishedAt(nowTS())); err != nil {
		t.Fatalf("complete orig: %v", err)
	}

	// Positive control: artifact present → instant cache hit.
	hit := submit()
	if hit["cached"] != true {
		t.Fatalf("artifact present: cached = %v, want true — %v", hit["cached"], hit)
	}

	// Poison the entry: artifact vanishes (CIFS write lost on reconnect).
	if err := os.Remove(filepath.Join(origDir, "result.mp4")); err != nil {
		t.Fatalf("remove artifact: %v", err)
	}
	miss := submit()
	if miss["cached"] == true {
		t.Fatalf("artifact missing: got cache hit anyway — %v", miss)
	}
	if miss["status"] != "queued" {
		t.Fatalf("artifact missing: status = %v, want queued — %v", miss["status"], miss)
	}
	if miss["job_id"] == origID {
		t.Fatal("artifact missing: re-run reused the poisoned original job id")
	}
}

// TestResolveFollowersPromotesWhenOriginalArtifactMissing proves that an
// original which "completed" without its on-disk artifact demotes to the
// failure path: the oldest follower is promoted to a real queued job instead
// of inheriting a bodyless result.
func TestResolveFollowersPromotesWhenOriginalArtifactMissing(t *testing.T) {
	store, outputDir := newTestStore(t)

	payload := json.RawMessage(`{"chunk":7}`)
	orig, err := store.CreateJob("ltx25-denoise1", "ltx25-denoise1", payload, 1)
	if err != nil {
		t.Fatalf("create original: %v", err)
	}
	// Completed result referencing a file that was NEVER written (lost write).
	origResult := json.RawMessage(`{"file":"result.mp4","format":"mp4"}`)
	if err := store.UpdateState(orig.ID, "completed", WithResult(origResult), WithFinishedAt(nowTS())); err != nil {
		t.Fatalf("complete original: %v", err)
	}
	follower, err := store.CreateFollowerJob("ltx25-denoise1", "ltx25-denoise1", payload, orig.ID)
	if err != nil {
		t.Fatalf("create follower: %v", err)
	}

	store.ResolveFollowers(orig.ID, "completed", &origResult, "", outputDir)

	got, err := store.GetJob(follower.ID)
	if err != nil {
		t.Fatalf("get follower: %v", err)
	}
	if got.State != "queued" {
		t.Fatalf("follower state = %q, want queued (promoted to re-run)", got.State)
	}
	if got.Result != nil {
		t.Fatalf("promoted follower inherited a bodyless result: %s", *got.Result)
	}
}

// TestResolveFollowersRelativeSymlink proves the follower output dir is a
// RELATIVE symlink (bare original dir name). The previous absolute spark path
// was stored server-side by the macOS SMB share and failed to traverse from
// spark's own CIFS mount with EINVAL, making every follower dir unreadable.
func TestResolveFollowersRelativeSymlink(t *testing.T) {
	store, outputDir := newTestStore(t)

	payload := json.RawMessage(`{"chunk":8}`)
	orig, err := store.CreateJob("ltx25-denoise1", "ltx25-denoise1", payload, 1)
	if err != nil {
		t.Fatalf("create original: %v", err)
	}
	origDir := filepath.Join(outputDir, "jobs", orig.ID)
	if err := os.MkdirAll(origDir, 0o755); err != nil {
		t.Fatalf("mkdir original dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(origDir, "result.mp4"), []byte("x"), 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	origResult := json.RawMessage(`{"file":"result.mp4","format":"mp4"}`)
	if err := store.UpdateState(orig.ID, "completed", WithResult(origResult), WithFinishedAt(nowTS())); err != nil {
		t.Fatalf("complete original: %v", err)
	}
	follower, err := store.CreateFollowerJob("ltx25-denoise1", "ltx25-denoise1", payload, orig.ID)
	if err != nil {
		t.Fatalf("create follower: %v", err)
	}

	store.ResolveFollowers(orig.ID, "completed", &origResult, "", outputDir)

	target, err := os.Readlink(filepath.Join(outputDir, "jobs", follower.ID))
	if err != nil {
		t.Fatalf("follower dir is not a symlink: %v", err)
	}
	if target != orig.ID {
		t.Fatalf("symlink target = %q, want bare relative name %q", target, orig.ID)
	}
	if info, err := os.Stat(filepath.Join(outputDir, "jobs", follower.ID, "result.mp4")); err != nil || info.Size() == 0 {
		t.Fatalf("artifact unreadable through relative symlink: %v", err)
	}
	got, err := store.GetJob(follower.ID)
	if err != nil {
		t.Fatalf("get follower: %v", err)
	}
	if got.State != "completed" {
		t.Fatalf("follower state = %q, want completed", got.State)
	}
}
