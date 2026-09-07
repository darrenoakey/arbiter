package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestSubmitJobSourceRoundTrip verifies that caller provenance submitted with
// a job is persisted and echoed back by GET /v1/jobs/{id}.
func TestSubmitJobSourceRoundTrip(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	api.config.Models["moondream"] = ModelConfig{MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1)}

	rec := postJob(t, api, `{"type":"caption","params":{"image":"eA=="},"source":{"who":"waggler","why":"hang williams video"}}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("submit status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var submitResp struct {
		JobID string `json:"job_id"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &submitResp); err != nil {
		t.Fatalf("decode submit response: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/jobs/"+submitResp.JobID, nil)
	getRec := httptest.NewRecorder()
	api.Handler().ServeHTTP(getRec, req)
	if getRec.Code != http.StatusOK {
		t.Fatalf("get status = %d, body = %s", getRec.Code, getRec.Body.String())
	}
	var job struct {
		Source *JobSource `json:"source"`
	}
	if err := json.Unmarshal(getRec.Body.Bytes(), &job); err != nil {
		t.Fatalf("decode job: %v", err)
	}
	if job.Source == nil || job.Source.Who != "waggler" || job.Source.Why != "hang williams video" {
		t.Fatalf("job source = %+v, want who=waggler why=hang williams video", job.Source)
	}
}

// TestSubmitJobSourceValidation: bad source shapes are rejected at the door;
// blank fields collapse to "no source" rather than an error.
func TestSubmitJobSourceValidation(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	api.config.Models["moondream"] = ModelConfig{MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1)}

	for name, body := range map[string]string{
		"not an object":     `{"type":"caption","params":{"image":"eA=="},"source":"waggler"}`,
		"non-string who":    `{"type":"caption","params":{"image":"eA=="},"source":{"who":42}}`,
		"over-long who":     `{"type":"caption","params":{"image":"eA=="},"source":{"who":"` + strings.Repeat("x", 129) + `"}}`,
		"over-long why":     `{"type":"caption","params":{"image":"eA=="},"source":{"why":"` + strings.Repeat("y", 257) + `"}}`,
		"control character": "{\"type\":\"caption\",\"params\":{\"image\":\"eA==\"},\"source\":{\"who\":\"a\\u0000b\"}}",
	} {
		t.Run(name, func(t *testing.T) {
			rec := postJob(t, api, body)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
			}
		})
	}

	rec := postJob(t, api, `{"type":"caption","params":{"image":"eA=="},"source":{"who":"","why":"   "}}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("blank source status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var submitResp struct {
		JobID string `json:"job_id"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &submitResp); err != nil {
		t.Fatalf("decode submit response: %v", err)
	}
	job, err := api.store.GetJob(submitResp.JobID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if job.Source != nil {
		t.Fatalf("blank source stored as %+v, want nil", job.Source)
	}
}

// TestSourceNotPartOfDedupIdentity: identical params from two callers dedup
// (follower, not a second queued job) while each row keeps its own source.
func TestSourceNotPartOfDedupIdentity(t *testing.T) {
	store, _ := newTestStore(t)
	store.InitDedup()

	payload := json.RawMessage(`{"prompt":"same work"}`)
	first, err := store.CreateJobWithRequestedModel("ltx2", "video-generate", payload, 1, "", WithSource(&JobSource{Who: "waggler", Why: "spring song"}))
	if err != nil {
		t.Fatalf("create first: %v", err)
	}
	hash := computeJobHash("video-generate", "ltx2", payload)
	store.DedupRegister(hash, first.ID)

	// Second submission with DIFFERENT source must still dedup-hit the first.
	follower, err := store.CreateFollowerJobWithRequestedModel("ltx2", "video-generate", payload, first.ID, "", WithSource(&JobSource{Who: "beezle3", Why: "hang williams video"}))
	if err != nil {
		t.Fatalf("create follower: %v", err)
	}
	reloadedFollower, err := store.GetJob(follower.ID)
	if err != nil {
		t.Fatalf("reload follower: %v", err)
	}
	if reloadedFollower.Source == nil || reloadedFollower.Source.Who != "beezle3" {
		t.Fatalf("follower source = %+v, want who=beezle3", reloadedFollower.Source)
	}
	reloadedFirst, err := store.GetJob(first.ID)
	if err != nil {
		t.Fatalf("reload first: %v", err)
	}
	if reloadedFirst.Source == nil || reloadedFirst.Source.Who != "waggler" {
		t.Fatalf("first source = %+v, want who=waggler", reloadedFirst.Source)
	}
}

// TestIdempotentJobCarriesSource: keyed creation persists provenance and a
// key replay returns the original row with its source intact.
func TestIdempotentJobCarriesSource(t *testing.T) {
	store, _ := newTestStore(t)
	payload := json.RawMessage(`{"q":"1"}`)
	src := &JobSource{Who: "tts-service", Why: "book-reader: dune ch4"}
	job, created, conflict, err := store.CreateIdempotentJob("kokoro", "tts", payload, 1, "", "key-1", "hash-1", WithSource(src))
	if err != nil || !created || conflict {
		t.Fatalf("create idempotent job: created=%v conflict=%v err=%v", created, conflict, err)
	}
	replay, created, conflict, err := store.CreateIdempotentJob("kokoro", "tts", payload, 1, "", "key-1", "hash-1", WithSource(src))
	if err != nil || created || conflict {
		t.Fatalf("replay idempotent job: created=%v conflict=%v err=%v", created, conflict, err)
	}
	if replay.ID != job.ID {
		t.Fatalf("replay job = %s, want %s", replay.ID, job.ID)
	}
	if replay.Source == nil || replay.Source.Why != "book-reader: dune ch4" {
		t.Fatalf("replay source = %+v", replay.Source)
	}
}

// TestPSActiveJobsDetailIncludesSource: the /v1/ps live-jobs panel lists
// non-terminal jobs with their provenance so the dashboard can show who/why.
func TestPSActiveJobsDetailIncludesSource(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	api.config.Models["moondream"] = ModelConfig{MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1)}

	rec := postJob(t, api, `{"type":"caption","params":{"image":"eA=="},"source":{"who":"photo-namer","why":"relabel italy 2024"}}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("submit status = %d, body = %s", rec.Code, rec.Body.String())
	}

	api.updatePSCache()
	psReq := httptest.NewRequest(http.MethodGet, "/v1/ps", nil)
	psRec := httptest.NewRecorder()
	api.Handler().ServeHTTP(psRec, psReq)
	if psRec.Code != http.StatusOK {
		t.Fatalf("ps status = %d", psRec.Code)
	}
	var ps struct {
		ActiveJobsDetail []map[string]any `json:"active_jobs_detail"`
	}
	if err := json.Unmarshal(psRec.Body.Bytes(), &ps); err != nil {
		t.Fatalf("decode ps: %v", err)
	}
	found := false
	for _, entry := range ps.ActiveJobsDetail {
		if entry["who"] == "photo-namer" && entry["why"] == "relabel italy 2024" {
			found = true
			if entry["state"] != "queued" {
				t.Fatalf("entry state = %v, want queued", entry["state"])
			}
		}
	}
	if !found {
		t.Fatalf("active_jobs_detail missing sourced entry: %s", psRec.Body.String())
	}
}

// TestExtractChatBodySource: non-standard top-level who/why are popped from an
// OpenAI-style chat body (keeping cache/dedup keys and worker payloads clean)
// and validated into a JobSource.
func TestExtractChatBodySource(t *testing.T) {
	src, body, err := extractChatBodySource([]byte(`{"model":"local-chat","who":"beezle3","why":"hang williams video","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("extract: %v", err)
	}
	if src == nil || src.Who != "beezle3" || src.Why != "hang williams video" {
		t.Fatalf("source = %+v", src)
	}
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("stripped body not json: %v", err)
	}
	if _, present := m["who"]; present {
		t.Fatalf("who not stripped: %s", body)
	}
	if _, present := m["why"]; present {
		t.Fatalf("why not stripped: %s", body)
	}
	if m["model"] != "local-chat" {
		t.Fatalf("model clobbered: %s", body)
	}

	src, body, err = extractChatBodySource([]byte(`{"model":"local-chat","messages":[]}`))
	if err != nil || src != nil || !bytes.Equal(body, []byte(`{"model":"local-chat","messages":[]}`)) {
		t.Fatalf("no-source passthrough: src=%v body=%s err=%v", src, body, err)
	}

	if _, _, err := extractChatBodySource([]byte(`{"who":17}`)); err == nil {
		t.Fatalf("non-string who accepted")
	}
	if _, _, err := extractChatBodySource([]byte(`{"why":"` + strings.Repeat("z", 300) + `"}`)); err == nil {
		t.Fatalf("over-long why accepted")
	}
}

// TestListJobsIncludesSource: GET /v1/jobs surfaces provenance per entry.
func TestListJobsIncludesSource(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	api.config.Models["moondream"] = ModelConfig{MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1)}

	rec := postJob(t, api, `{"type":"caption","params":{"image":"eA=="},"source":{"who":"ventrilo-train","why":"voice fit eval"}}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("submit status = %d, body = %s", rec.Code, rec.Body.String())
	}
	listReq := httptest.NewRequest(http.MethodGet, "/v1/jobs?state=queued", nil)
	listRec := httptest.NewRecorder()
	api.Handler().ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list status = %d", listRec.Code)
	}
	var entries []map[string]any
	if err := json.Unmarshal(listRec.Body.Bytes(), &entries); err != nil {
		t.Fatalf("decode list: %v", err)
	}
	if len(entries) == 0 {
		t.Fatalf("no queued jobs listed")
	}
	if entries[0]["who"] != "ventrilo-train" || entries[0]["why"] != "voice fit eval" {
		t.Fatalf("first entry missing source: %+v", entries[0])
	}
}

// TestJobSourceLogFields: absent provenance logs no extra keys; present
// provenance logs exactly source_who/source_why.
func TestJobSourceLogFields(t *testing.T) {
	var nilSrc *JobSource
	if fields := nilSrc.logFields(); fields != nil {
		t.Fatalf("nil source fields = %v, want nil", fields)
	}
	fields := (&JobSource{Why: "spring song"}).logFields()
	if _, has := fields["source_who"]; has {
		t.Fatalf("empty who logged: %v", fields)
	}
	if fields["source_why"] != "spring song" {
		t.Fatalf("why missing: %v", fields)
	}
}
