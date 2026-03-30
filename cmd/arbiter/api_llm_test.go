package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func newTestAPI(t *testing.T) (*API, func()) {
	t.Helper()

	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	if err := os.MkdirAll(filepath.Join(outputDir, "logs"), 0o755); err != nil {
		t.Fatalf("mkdir logs: %v", err)
	}
	store, err := NewStore(filepath.Join(outputDir, "arbiter.db"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	cfg := &Config{
		VRAMBudgetGB: 100,
		Host:         "127.0.0.1",
		Port:         8400,
		Models:       map[string]ModelConfig{},
	}
	mgr := NewInstanceManager(100, "python3", projectRoot)
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)
	api := NewAPI(cfg, store, mgr, sched, logger, outputDir, projectRoot)

	cleanup := func() {
		logger.Close()
		store.Close()
		mgr.KillAll()
	}
	return api, cleanup
}

func writeFakeLLMWorker(t *testing.T, path string) {
	t.Helper()
	script := `#!/usr/bin/env python3
import json
import os
import sys

capture = os.environ["CAPTURE_ENV_FILE"]
payload = {
    "argv": sys.argv,
    "env": {
        "LLM_HF_REPO": os.environ.get("LLM_HF_REPO"),
        "LLM_HF_FILE": os.environ.get("LLM_HF_FILE"),
        "LLM_MODEL_PATH": os.environ.get("LLM_MODEL_PATH"),
        "LLM_CTX_SIZE": os.environ.get("LLM_CTX_SIZE"),
        "LLM_GPU_LAYERS": os.environ.get("LLM_GPU_LAYERS"),
        "LLAMA_SERVER_BIN": os.environ.get("LLAMA_SERVER_BIN"),
        "EXTRA_FLAG": os.environ.get("EXTRA_FLAG"),
    },
}
with open(capture, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(payload) + "\n")

for line in sys.stdin:
    msg = json.loads(line)
    cmd = msg.get("cmd")
    if cmd == "load":
        print(json.dumps({"status": "ok", "vram_bytes": 0}), flush=True)
    elif cmd == "unload":
        print(json.dumps({"status": "ok"}), flush=True)
    elif cmd == "shutdown":
        print(json.dumps({"status": "ok"}), flush=True)
        break
    elif cmd == "get_port":
        print(json.dumps({"status": "ok", "result": {"port": 12345}}), flush=True)
    elif cmd == "infer":
        print(json.dumps({"status": "ok", "req_id": msg.get("req_id", ""), "result": {"format": "json"}}), flush=True)
`
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("write fake worker: %v", err)
	}
}

func readCaptureFile(t *testing.T, path string) []map[string]any {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open capture file: %v", err)
	}
	defer f.Close()

	var entries []map[string]any
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var entry map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			t.Fatalf("unmarshal capture entry: %v", err)
		}
		entries = append(entries, entry)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan capture file: %v", err)
	}
	return entries
}

func TestLLMLiveConfigMutationAndReload(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	workerPath := filepath.Join(t.TempDir(), "fake-llm-worker.py")
	capturePath := filepath.Join(t.TempDir(), "capture.jsonl")
	writeFakeLLMWorker(t, workerPath)

	registerBody := map[string]any{
		"name":             "custom-llm",
		"hf_model":         "example/custom-llm-GGUF",
		"hf_file":          "model.gguf",
		"worker_cmd":       []string{"python3", workerPath},
		"adapter_params":   map[string]string{"CAPTURE_ENV_FILE": capturePath, "EXTRA_FLAG": "one"},
		"llama_server_bin": "/tmp/llama-server-test",
	}
	raw, _ := json.Marshal(registerBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/llm/models", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	api.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("register LLM status = %d, body = %s", rec.Code, rec.Body.String())
	}

	modelReq := httptest.NewRequest(http.MethodGet, "/v1/models/custom-llm", nil)
	modelRec := httptest.NewRecorder()
	api.Handler().ServeHTTP(modelRec, modelReq)
	if modelRec.Code != http.StatusOK {
		t.Fatalf("get model status = %d, body = %s", modelRec.Code, modelRec.Body.String())
	}

	modelID := llmModelID("custom-llm")
	instances := api.mgr.GetModelInstances(modelID)
	if len(instances) != 1 {
		t.Fatalf("registered instances = %d, want 1", len(instances))
	}
	if err := instances[0].Load("cuda"); err != nil {
		t.Fatalf("initial load: %v", err)
	}
	entries := readCaptureFile(t, capturePath)
	if got := entries[0]["env"].(map[string]any)["EXTRA_FLAG"]; got != "one" {
		t.Fatalf("initial EXTRA_FLAG = %v, want one", got)
	}
	if got := entries[0]["env"].(map[string]any)["LLAMA_SERVER_BIN"]; got != "/tmp/llama-server-test" {
		t.Fatalf("initial LLAMA_SERVER_BIN = %v", got)
	}

	patchBody := map[string]any{
		"worker_cmd":     []string{"python3", workerPath},
		"adapter_params": map[string]string{"CAPTURE_ENV_FILE": capturePath, "EXTRA_FLAG": "two"},
	}
	raw, _ = json.Marshal(patchBody)
	patchReq := httptest.NewRequest(http.MethodPatch, "/v1/models/custom-llm", bytes.NewReader(raw))
	patchReq.Header.Set("Content-Type", "application/json")
	patchRec := httptest.NewRecorder()
	api.Handler().ServeHTTP(patchRec, patchReq)
	if patchRec.Code != http.StatusOK {
		t.Fatalf("patch model status = %d, body = %s", patchRec.Code, patchRec.Body.String())
	}

	reloadReq := httptest.NewRequest(http.MethodPost, "/v1/models/custom-llm/reload", nil)
	reloadRec := httptest.NewRecorder()
	api.Handler().ServeHTTP(reloadRec, reloadReq)
	if reloadRec.Code != http.StatusOK {
		t.Fatalf("reload model status = %d, body = %s", reloadRec.Code, reloadRec.Body.String())
	}

	instances = api.mgr.GetModelInstances(modelID)
	if len(instances) != 1 {
		t.Fatalf("instances after reload = %d, want 1", len(instances))
	}
	if err := instances[0].Load("cuda"); err != nil {
		t.Fatalf("load after reload: %v", err)
	}

	entries = readCaptureFile(t, capturePath)
	last := entries[len(entries)-1]["env"].(map[string]any)
	if got := last["EXTRA_FLAG"]; got != "two" {
		t.Fatalf("reloaded EXTRA_FLAG = %v, want two", got)
	}
	if got := last["LLM_HF_REPO"]; got != "example/custom-llm-GGUF" {
		t.Fatalf("reloaded LLM_HF_REPO = %v", got)
	}
}
