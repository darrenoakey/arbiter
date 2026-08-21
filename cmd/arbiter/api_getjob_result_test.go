package main

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"testing"
)

func TestGetJobUsesAdapterFileForResultPath(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	job, err := api.store.CreateJob("ltx25-encode", "video-generate", json.RawMessage(`{}`), 1)
	if err != nil {
		t.Fatalf("CreateJob: %v", err)
	}
	jobDir := resolveJobDir(api.config, api.outputDir, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	payload := []byte("encoded-latent-bytes")
	if err := os.WriteFile(filepath.Join(jobDir, "encoded.pt"), payload, 0o644); err != nil {
		t.Fatalf("write encoded.pt: %v", err)
	}
	result := json.RawMessage(`{"file":"encoded.pt","format":"pt"}`)
	if err := api.store.UpdateState(job.ID, "completed", WithResult(result), WithFinishedAt(nowTS())); err != nil {
		t.Fatalf("UpdateState: %v", err)
	}

	response := performRequest(api, http.MethodGet, "/v1/jobs/"+job.ID, "")
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", response.Code, response.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	resultMap, ok := body["result"].(map[string]any)
	if !ok {
		t.Fatalf("result type %T", body["result"])
	}
	resultPath, ok := resultMap["result_path"].(string)
	if !ok {
		t.Fatalf("result_path type %T", resultMap["result_path"])
	}
	if filepath.Base(resultPath) != "encoded.pt" {
		t.Fatalf("result_path = %q", resultPath)
	}
	if _, err := os.Stat(resultPath); err != nil {
		t.Fatalf("result_path missing: %v", err)
	}
	encoded, ok := resultMap["data"].(string)
	if !ok {
		t.Fatalf("data type %T", resultMap["data"])
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatalf("decode data: %v", err)
	}
	if string(decoded) != string(payload) {
		t.Fatalf("data = %q", decoded)
	}

	skipped := performRequest(api, http.MethodGet, "/v1/jobs/"+job.ID+"?no_data=1", "")
	if skipped.Code != http.StatusOK {
		t.Fatalf("no_data status = %d body = %s", skipped.Code, skipped.Body.String())
	}
	var skippedBody map[string]any
	if err := json.Unmarshal(skipped.Body.Bytes(), &skippedBody); err != nil {
		t.Fatalf("decode no_data: %v", err)
	}
	skippedResult, ok := skippedBody["result"].(map[string]any)
	if !ok {
		t.Fatalf("no_data result type %T", skippedBody["result"])
	}
	if _, exists := skippedResult["data"]; exists {
		t.Fatal("no_data=1 still inlined data")
	}
}

func TestGetJobFallsBackToResultFormatFilename(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	job, err := api.store.CreateJob("local-chat", "chat", json.RawMessage(`{}`), 1)
	if err != nil {
		t.Fatalf("CreateJob: %v", err)
	}
	jobDir := resolveJobDir(api.config, api.outputDir, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	payload := []byte(`{"ok":true}`)
	if err := os.WriteFile(filepath.Join(jobDir, "result.json"), payload, 0o644); err != nil {
		t.Fatalf("write result.json: %v", err)
	}
	result := json.RawMessage(`{"format":"json"}`)
	if err := api.store.UpdateState(job.ID, "completed", WithResult(result), WithFinishedAt(nowTS())); err != nil {
		t.Fatalf("UpdateState: %v", err)
	}

	response := performRequest(api, http.MethodGet, "/v1/jobs/"+job.ID, "")
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", response.Code, response.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	resultMap, ok := body["result"].(map[string]any)
	if !ok {
		t.Fatalf("result type %T", body["result"])
	}
	resultPath, ok := resultMap["result_path"].(string)
	if !ok {
		t.Fatalf("result_path type %T", resultMap["result_path"])
	}
	if filepath.Base(resultPath) != "result.json" {
		t.Fatalf("result_path = %q", resultPath)
	}
}
