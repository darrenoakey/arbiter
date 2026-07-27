package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
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
	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 100}, "python3", projectRoot)
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)
	api := NewAPI(cfg, store, mgr, sched, logger, outputDir, projectRoot)

	cleanup := func() {
		logger.Close()
		store.Close()
		mgr.KillAll()
	}
	return api, cleanup
}

func writeProtocolWorker(t *testing.T, path, capturePath string) {
	t.Helper()
	script := fmt.Sprintf(`#!/usr/bin/env python3
import json
import os
import sys

capture = %q
payload = {
    "argv": sys.argv,
    "env": {
        "LLM_HF_REPO": os.environ.get("LLM_HF_REPO"),
        "LLM_HF_FILE": os.environ.get("LLM_HF_FILE"),
        "LLM_MODEL_PATH": os.environ.get("LLM_MODEL_PATH"),
        "LLM_CTX_SIZE": os.environ.get("LLM_CTX_SIZE"),
        "LLM_GPU_LAYERS": os.environ.get("LLM_GPU_LAYERS"),
        "LLAMA_SERVER_BIN": os.environ.get("LLAMA_SERVER_BIN"),
        "LLM_PARALLEL": os.environ.get("LLM_PARALLEL"),
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
`, capturePath)
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("write protocol worker: %v", err)
	}
}

func writeStreamingProtocolWorker(t *testing.T, path string, port int) {
	t.Helper()
	script := fmt.Sprintf(`#!/usr/bin/env python3
import json
import os
import sys

port = %d

for line in sys.stdin:
    msg = json.loads(line)
    cmd = msg.get("cmd")
    req_id = msg.get("req_id", "_default")
    if cmd == "load":
        print(json.dumps({"status": "ok", "req_id": req_id}), flush=True)
    elif cmd == "unload":
        print(json.dumps({"status": "ok", "req_id": req_id}), flush=True)
    elif cmd == "shutdown":
        print(json.dumps({"status": "ok", "req_id": req_id}), flush=True)
        break
    elif cmd == "get_port":
        print(json.dumps({"status": "ok", "req_id": req_id, "result": {"port": port}}), flush=True)
`, port)
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("write streaming protocol worker: %v", err)
	}
}

func readCaptureFile(t *testing.T, path string) []map[string]any {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open capture file: %v", err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			t.Errorf("close capture file: %v", err)
		}
	}()

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

	workerPath := filepath.Join(api.projectRoot, "llm-worker")
	capturePath := filepath.Join(t.TempDir(), "capture.jsonl")
	writeProtocolWorker(t, workerPath, capturePath)
	llamaServerPath := filepath.Join(api.projectRoot, "local", "bin", "llama-server")
	if err := os.MkdirAll(filepath.Dir(llamaServerPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(llamaServerPath, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	registerBody := map[string]any{
		"name":             "custom-llm",
		"hf_model":         "example/custom-llm-GGUF",
		"hf_file":          "model.gguf",
		"worker_cmd":       []string{workerPath},
		"adapter_params":   map[string]string{"LLM_PARALLEL": "2"},
		"llama_server_bin": llamaServerPath,
	}
	raw, _ := json.Marshal(registerBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/llm/models", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	api.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
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
	if got := entries[0]["env"].(map[string]any)["LLM_PARALLEL"]; got != "2" {
		t.Fatalf("initial LLM_PARALLEL = %v, want 2", got)
	}
	if got := entries[0]["env"].(map[string]any)["LLAMA_SERVER_BIN"]; got != llamaServerPath {
		t.Fatalf("initial LLAMA_SERVER_BIN = %v", got)
	}

	patchBody := map[string]any{
		"worker_cmd":     []string{workerPath},
		"adapter_params": map[string]string{"LLM_PARALLEL": "3"},
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
	if got := last["LLM_PARALLEL"]; got != "3" {
		t.Fatalf("reloaded LLM_PARALLEL = %v, want 3", got)
	}
	if got := last["LLM_HF_REPO"]; got != "example/custom-llm-GGUF" {
		t.Fatalf("reloaded LLM_HF_REPO = %v", got)
	}
}

func TestLLMModelConfigPatchNoRemoteSpillRoundTrip(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	workerPath := filepath.Join(api.projectRoot, "llm-worker")
	capturePath := filepath.Join(t.TempDir(), "capture.jsonl")
	writeProtocolWorker(t, workerPath, capturePath)
	llamaServerPath := filepath.Join(api.projectRoot, "local", "bin", "llama-server")
	if err := os.MkdirAll(filepath.Dir(llamaServerPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(llamaServerPath, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	registerBody := map[string]any{
		"name":             "no-spill-llm",
		"hf_model":         "example/no-spill-llm-GGUF",
		"hf_file":          "model.gguf",
		"worker_cmd":       []string{workerPath},
		"llama_server_bin": llamaServerPath,
		"placements":       []string{"spark"},
	}
	raw, _ := json.Marshal(registerBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/llm/models", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	api.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("register LLM status = %d, body = %s", rec.Code, rec.Body.String())
	}
	modelID := llmModelID("no-spill-llm")

	// Initially NoRemoteSpill should be false.
	getReq := httptest.NewRequest(http.MethodGet, "/v1/models/no-spill-llm", nil)
	getRec := httptest.NewRecorder()
	api.Handler().ServeHTTP(getRec, getReq)
	if getRec.Code != http.StatusOK {
		t.Fatalf("get model status = %d, body = %s", getRec.Code, getRec.Body.String())
	}
	var getResp map[string]any
	if err := json.Unmarshal(getRec.Body.Bytes(), &getResp); err != nil {
		t.Fatalf("unmarshal get response: %v", err)
	}
	if got, ok := getResp["no_remote_spill"].(bool); !ok || got {
		t.Fatalf("initial no_remote_spill = %v, want false", getResp["no_remote_spill"])
	}

	// Patch no_remote_spill to true.
	patchBody := map[string]any{"no_remote_spill": true}
	raw, _ = json.Marshal(patchBody)
	patchReq := httptest.NewRequest(http.MethodPatch, "/v1/models/no-spill-llm", bytes.NewReader(raw))
	patchReq.Header.Set("Content-Type", "application/json")
	patchRec := httptest.NewRecorder()
	api.Handler().ServeHTTP(patchRec, patchReq)
	if patchRec.Code != http.StatusOK {
		t.Fatalf("patch no_remote_spill status = %d, body = %s", patchRec.Code, patchRec.Body.String())
	}

	// Verify GET reflects the change and the live config has it.
	getRec = httptest.NewRecorder()
	api.Handler().ServeHTTP(getRec, getReq)
	if getRec.Code != http.StatusOK {
		t.Fatalf("get model after patch status = %d, body = %s", getRec.Code, getRec.Body.String())
	}
	if err := json.Unmarshal(getRec.Body.Bytes(), &getResp); err != nil {
		t.Fatalf("unmarshal get response after patch: %v", err)
	}
	if got, ok := getResp["no_remote_spill"].(bool); !ok || !got {
		t.Fatalf("patched no_remote_spill = %v, want true", getResp["no_remote_spill"])
	}
	if !api.config.Models[modelID].NoRemoteSpillOrDefault() {
		t.Fatalf("live config NoRemoteSpill=false, want true")
	}
}

func TestLLMRegistrationRejectsInjectedAndNestedAdapterParamsBeforePersistence(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	bodies := []string{
		`{"name":"attack","hf_model":"org/model-GGUF","adapter_params":{"LD_PRELOAD":"/tmp/payload.so"}}`,
		`{"name":"attack","hf_model":"org/model-GGUF","adapter_params":{"DYLD_INSERT_LIBRARIES":"/tmp/payload.dylib"}}`,
		`{"name":"attack","hf_model":"org/model-GGUF","adapter_params":{"PYTHONPATH":"/tmp/module"}}`,
		`{"name":"attack","hf_model":"org/model-GGUF","adapter_params":{"PATH":"/tmp/bin"}}`,
		`{"name":"attack","hf_model":"org/model-GGUF","adapter_params":{"SHELL":"/bin/sh"}}`,
		`{"name":"attack","hf_model":"org/model-GGUF","adapter_params":{"LLM_CTX_SIZE":{"value":"8192"}}}`,
	}
	for _, body := range bodies {
		request := httptest.NewRequest(http.MethodPost, "/v1/llm/models", strings.NewReader(body))
		response := httptest.NewRecorder()
		api.Handler().ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Errorf("body %s status = %d, response = %s", body, response.Code, response.Body.String())
		}
	}
	if len(api.config.Models) != 0 {
		t.Fatalf("rejected model entered runtime config: %+v", api.config.Models)
	}
	if raw, err := os.ReadFile(filepath.Join(api.projectRoot, "local", "config.json")); err == nil && strings.Contains(string(raw), "PRELOAD") {
		t.Fatalf("rejected adapter params persisted: %s", raw)
	}
}

func TestModelUpdateAndReloadRejectInjectedAdapterParams(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	modelID := "llm:secured"
	config := repositoryWorkerConfig(api.projectRoot, "llm-worker", "llamacpp")
	config.MemoryGB = 1
	config.MaxConcurrent = 1
	config.MaxInstances = intPtr(1)
	api.config.Models[modelID] = config
	api.mgr.ScaleModel(modelID, 1, config)

	patch := `{"adapter_params":{"BASH_ENV":"/tmp/startup.sh"},"reload_workers":true}`
	request := httptest.NewRequest(http.MethodPatch, "/v1/models/secured", strings.NewReader(patch))
	response := httptest.NewRecorder()
	api.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("injected update status = %d, body = %s", response.Code, response.Body.String())
	}
	if _, ok := api.config.Models[modelID].AdapterParams["BASH_ENV"]; ok {
		t.Fatal("rejected update mutated runtime config")
	}

	mutated := api.config.Models[modelID]
	mutated.AdapterParams["LD_PRELOAD"] = "/tmp/payload.so"
	api.config.Models[modelID] = mutated
	reload := httptest.NewRequest(http.MethodPost, "/v1/models/secured/reload", nil)
	reloadResponse := httptest.NewRecorder()
	api.Handler().ServeHTTP(reloadResponse, reload)
	if reloadResponse.Code != http.StatusBadRequest {
		t.Fatalf("mutated reload status = %d, body = %s", reloadResponse.Code, reloadResponse.Body.String())
	}
}

func TestLLMRegistrationRejectsFreeFormVllmFlags(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	body := `{"name":"attack","backend":"vllm","vllm_model":"org/model","vllm_extra_args":"--served-model-name innocent; sh -c id"}`
	request := httptest.NewRequest(http.MethodPost, "/v1/llm/models", strings.NewReader(body))
	response := httptest.NewRecorder()
	api.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "vllm_extra_args is disabled") {
		t.Fatalf("free-form flags response = %d %s", response.Code, response.Body.String())
	}
}

func TestModelAndLLMNumericBoundsRejectWithoutMutation(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	tests := []struct{ path, body string }{
		{path: "/v1/models", body: `{"model_id":"m1","memory_gb":0}`},
		{path: "/v1/models", body: `{"model_id":"m2","memory_gb":-1}`},
		{path: "/v1/models", body: `{"model_id":"m3","memory_gb":101}`},
		{path: "/v1/models", body: `{"model_id":"m4","memory_gb":1,"max_instances":-1}`},
		{path: "/v1/models", body: `{"model_id":"m5","memory_gb":1,"max_instances":129}`},
		{path: "/v1/models", body: `{"model_id":"m6","memory_gb":1,"max_concurrent":0}`},
		{path: "/v1/models", body: `{"model_id":"m7","memory_gb":1,"max_concurrent":1025}`},
		{path: "/v1/models", body: `{"model_id":"m8","memory_gb":1,"max_runtime_seconds":0}`},
		{path: "/v1/models", body: `{"model_id":"m9","memory_gb":1,"max_runtime_seconds":604801}`},
		{path: "/v1/models", body: `{"model_id":"m10","memory_gb":1,"pressure_index":-0.1}`},
		{path: "/v1/models", body: `{"model_id":"m11","memory_gb":1,"pressure_index":1.1}`},
		{path: "/v1/models", body: `{"model_id":"m12","memory_gb":1,"avg_inference_ms":-1}`},
		{path: "/v1/models", body: `{"model_id":"m13","memory_gb":1,"load_ms":604800001}`},
		{path: "/v1/models", body: `{"model_id":"m13b","memory_gb":1,"group_priority":1000001}`},
		{path: "/v1/models", body: `{"model_id":"m14","memory_gb":NaN}`},
		{path: "/v1/models", body: `{"model_id":"m15","memory_gb":1e309}`},
		{path: "/v1/models", body: `{"model_id":"m16","memory_gb":1,"max_instances":999999999999999999999}`},
		{path: "/v1/llm/models", body: `{"name":"l1","hf_model":"org/model-GGUF","ctx_size":127}`},
		{path: "/v1/llm/models", body: `{"name":"l0","hf_model":"org/model-GGUF","ctx_size":0}`},
		{path: "/v1/llm/models", body: `{"name":"l2","hf_model":"org/model-GGUF","ctx_size":1048577}`},
		{path: "/v1/llm/models", body: `{"name":"l3","hf_model":"org/model-GGUF","gpu_layers":-2}`},
		{path: "/v1/llm/models", body: `{"name":"l4","hf_model":"org/model-GGUF","gpu_layers":10001}`},
		{path: "/v1/llm/models", body: `{"name":"l5","hf_model":"org/model-GGUF","adapter_params":{"LLM_PARALLEL":"0"}}`},
		{path: "/v1/llm/models", body: `{"name":"l6","hf_model":"org/model-GGUF","adapter_params":{"LLM_PARALLEL":"1025"}}`},
		{path: "/v1/llm/models", body: `{"name":"l7","hf_model":"org/model-GGUF","adapter_params":{"LLM_PARALLEL":"999999999999999999999"}}`},
		{path: "/v1/llm/models", body: `{"name":"l8","hf_model":"org/model-GGUF","memory_gb":Infinity}`},
		{path: "/v1/llm/models", body: `{"name":"l9","hf_model":"org/model-GGUF","memory_gb":0}`},
		{path: "/v1/llm/models", body: `{"name":"l10","hf_model":"org/model-GGUF","memory_gb":-1}`},
		{path: "/v1/llm/models", body: `{"name":"l11","hf_model":"org/model-GGUF","memory_gb":101}`},
	}
	for _, test := range tests {
		request := httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body))
		response := httptest.NewRecorder()
		api.Handler().ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Errorf("body %s status = %d, response = %s", test.body, response.Code, response.Body.String())
		}
	}
	if len(api.config.Models) != 0 || len(api.mgr.byModel) != 0 {
		t.Fatalf("invalid numeric input changed runtime: models=%v runtime=%v", api.config.Models, api.mgr.byModel)
	}
	configPath := filepath.Join(api.projectRoot, "local", "config.json")
	if _, err := os.Stat(configPath); !os.IsNotExist(err) {
		t.Fatalf("invalid numeric input changed persisted state: %v", err)
	}
}

func TestModelAPIAllowsExtendedRuntimeOnlyForExactLatentSyncID(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	create := httptest.NewRequest(http.MethodPost, "/v1/models", strings.NewReader(
		`{"model_id":"latentsync","memory_gb":1,"max_instances":0,"max_runtime_seconds":4000000}`))
	createResponse := httptest.NewRecorder()
	api.Handler().ServeHTTP(createResponse, create)
	if createResponse.Code != http.StatusOK {
		t.Fatalf("latentsync create status = %d, body = %s", createResponse.Code, createResponse.Body.String())
	}

	neighbor := httptest.NewRequest(http.MethodPost, "/v1/models", strings.NewReader(
		`{"model_id":"latentsync-copy","memory_gb":1,"max_instances":0,"max_runtime_seconds":4000000}`))
	neighborResponse := httptest.NewRecorder()
	api.Handler().ServeHTTP(neighborResponse, neighbor)
	if neighborResponse.Code != http.StatusBadRequest || !strings.Contains(neighborResponse.Body.String(), "604800") {
		t.Fatalf("neighbor create status = %d, body = %s", neighborResponse.Code, neighborResponse.Body.String())
	}

	overflow := httptest.NewRequest(http.MethodPatch, "/v1/models/latentsync", strings.NewReader(
		`{"max_runtime_seconds":4000001}`))
	overflowResponse := httptest.NewRecorder()
	api.Handler().ServeHTTP(overflowResponse, overflow)
	if overflowResponse.Code != http.StatusBadRequest || !strings.Contains(overflowResponse.Body.String(), "4000000") {
		t.Fatalf("latentsync overflow patch status = %d, body = %s", overflowResponse.Code, overflowResponse.Body.String())
	}

	accepted := httptest.NewRequest(http.MethodPatch, "/v1/models/latentsync", strings.NewReader(
		`{"max_runtime_seconds":4000000}`))
	acceptedResponse := httptest.NewRecorder()
	api.Handler().ServeHTTP(acceptedResponse, accepted)
	if acceptedResponse.Code != http.StatusOK {
		t.Fatalf("latentsync exact patch status = %d, body = %s", acceptedResponse.Code, acceptedResponse.Body.String())
	}
}

func TestLLMUpdateNumericBoundsPreserveDiskAndLiveRuntime(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	modelID := "llm:numeric-update"
	workerPath := filepath.Join(api.projectRoot, "llm-worker")
	writeProtocolWorker(t, workerPath, filepath.Join(t.TempDir(), "capture.jsonl"))
	current := repositoryWorkerConfig(api.projectRoot, "llm-worker", "llamacpp")
	api.config.Models[modelID] = current
	api.mgr.ScaleModel(modelID, 1, current)
	instance := api.mgr.GetModelInstances(modelID)[0]
	if err := instance.Load("cuda"); err != nil {
		t.Fatal(err)
	}
	processID := instance.cmd.Process.Pid
	if err := SaveModelConfig(api.projectRoot, modelID, current); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(api.projectRoot, "local", "config.json")
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	bodies := []string{
		`{"memory_gb":0}`, `{"memory_gb":101}`, `{"memory_gb":1e309}`,
		`{"max_instances":-1}`, `{"max_instances":129}`,
		`{"max_concurrent":0}`, `{"max_concurrent":1025}`,
		`{"max_runtime_seconds":0}`, `{"max_runtime_seconds":604801}`,
		`{"keep_alive_seconds":-1}`, `{"keep_alive_seconds":604801}`,
		`{"avg_inference_ms":NaN}`, `{"load_ms":604800001}`,
		`{"pressure_index":-0.1}`, `{"pressure_index":1.1}`,
		`{"adapter_params":{"LLM_CTX_SIZE":"0"},"reload_workers":true}`,
		`{"adapter_params":{"LLM_PARALLEL":"1025"},"reload_workers":true}`,
	}
	for _, body := range bodies {
		request := httptest.NewRequest(http.MethodPatch, "/v1/models/numeric-update", strings.NewReader(body))
		response := httptest.NewRecorder()
		api.Handler().ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Errorf("body %s status = %d, response = %s", body, response.Code, response.Body.String())
		}
	}
	after, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(before, after) {
		t.Fatalf("invalid update changed persisted state: equal=%v error=%v", bytes.Equal(before, after), err)
	}
	instances := api.mgr.GetModelInstances(modelID)
	if len(instances) != 1 || instances[0] != instance || instance.cmd.Process.Pid != processID || instance.State() != "loaded" {
		t.Fatal("invalid update changed live process runtime")
	}
	if got := api.config.Models[modelID]; got.MemoryGB != current.MemoryGB || got.MaxConcurrent != current.MaxConcurrent || *got.MaxInstances != *current.MaxInstances {
		t.Fatalf("invalid update changed in-memory config: %+v", got)
	}
}

func TestModelRegistrationPersistenceFailureLeavesRuntimeUnchanged(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	restore := makeConfigStorageUnwritable(t, api.projectRoot)
	defer restore()

	request := httptest.NewRequest(http.MethodPost, "/v1/models", strings.NewReader(`{"model_id":"birefnet","memory_gb":1}`))
	response := httptest.NewRecorder()
	api.Handler().ServeHTTP(response, request)
	if response.Code < 500 {
		t.Fatalf("registration status = %d, body = %s", response.Code, response.Body.String())
	}
	if _, exists := api.config.Models["birefnet"]; exists {
		t.Fatal("failed registration mutated in-memory config")
	}
	if _, exists := api.mgr.byModel["birefnet"]; exists {
		t.Fatal("failed registration created runtime model")
	}
}

func TestModelUpdatePersistenceFailureLeavesProcessAndConfigUnchanged(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	modelID := "llm:unchanged"
	workerPath := filepath.Join(api.projectRoot, "llm-worker")
	capturePath := filepath.Join(t.TempDir(), "capture.jsonl")
	writeProtocolWorker(t, workerPath, capturePath)
	current := repositoryWorkerConfig(api.projectRoot, "llm-worker", "llamacpp")
	current.MemoryGB = 1
	current.MaxConcurrent = 1
	current.MaxInstances = intPtr(1)
	api.config.Models[modelID] = current
	api.mgr.ScaleModel(modelID, 1, current)
	instance := api.mgr.GetModelInstances(modelID)[0]
	if err := instance.Load("cuda"); err != nil {
		t.Fatal(err)
	}
	pid := instance.cmd.Process.Pid
	if err := SaveModelConfig(api.projectRoot, modelID, current); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(filepath.Join(api.projectRoot, "local", "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	restore := makeConfigStorageUnwritable(t, api.projectRoot)
	defer restore()

	request := httptest.NewRequest(http.MethodPatch, "/v1/models/unchanged", strings.NewReader(`{"max_instances":2,"max_concurrent":2}`))
	response := httptest.NewRecorder()
	api.Handler().ServeHTTP(response, request)
	if response.Code < 500 {
		t.Fatalf("update status = %d, body = %s", response.Code, response.Body.String())
	}
	if got := api.config.Models[modelID]; got.MaxConcurrent != 1 || *got.MaxInstances != 1 {
		t.Fatalf("runtime config changed after persistence failure: %+v", got)
	}
	instances := api.mgr.GetModelInstances(modelID)
	if len(instances) != 1 || instances[0] != instance || instances[0].cmd.Process.Pid != pid || instances[0].State() != "loaded" {
		t.Fatal("loaded process lifecycle changed before persistence succeeded")
	}
	restore()
	after, err := os.ReadFile(filepath.Join(api.projectRoot, "local", "config.json"))
	if err != nil || !bytes.Equal(before, after) {
		t.Fatalf("persisted config changed: equal=%v error=%v", bytes.Equal(before, after), err)
	}
}

func TestLLMRegistrationPersistenceFailureReturnsErrorAndLeavesNoRuntime(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	workerPath := filepath.Join(api.projectRoot, "llm-worker")
	writeProtocolWorker(t, workerPath, filepath.Join(t.TempDir(), "capture.jsonl"))
	restore := makeConfigStorageUnwritable(t, api.projectRoot)
	defer restore()

	body := fmt.Sprintf(`{"name":"not-persisted","hf_model":"org/model-GGUF","worker_cmd":[%q]}`, workerPath)
	request := httptest.NewRequest(http.MethodPost, "/v1/llm/models", strings.NewReader(body))
	response := httptest.NewRecorder()
	api.Handler().ServeHTTP(response, request)
	if response.Code < 500 {
		t.Fatalf("LLM registration status = %d, body = %s", response.Code, response.Body.String())
	}
	modelID := llmModelID("not-persisted")
	if _, exists := api.config.Models[modelID]; exists {
		t.Fatal("failed LLM registration mutated in-memory config")
	}
	if _, exists := api.mgr.byModel[modelID]; exists {
		t.Fatal("failed LLM registration created runtime model")
	}
	if _, exists := JobTypeToModel["chat-completion:not-persisted"]; exists {
		t.Fatal("failed LLM registration created job routing")
	}
}

func TestPersistModelConfigTransactionRollsBackExactBytesAndLiveProcess(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	modelID := "llm:rollback"
	workerPath := filepath.Join(api.projectRoot, "llm-worker")
	writeProtocolWorker(t, workerPath, filepath.Join(t.TempDir(), "capture.jsonl"))
	current := repositoryWorkerConfig(api.projectRoot, "llm-worker", "llamacpp")
	current.MemoryGB = 1
	current.MaxConcurrent = 1
	current.MaxInstances = intPtr(1)
	api.config.Models[modelID] = current
	api.mgr.ScaleModel(modelID, 1, current)
	original := api.mgr.GetModelInstances(modelID)[0]
	if err := original.Load("cuda"); err != nil {
		t.Fatal(err)
	}
	originalPID := original.cmd.Process.Pid
	if err := SaveModelConfig(api.projectRoot, modelID, current); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(api.projectRoot, "local", "config.json")
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	updated := current
	updated.MaxConcurrent = 2
	updated.MaxInstances = intPtr(2)
	err = persistModelConfigTransaction(api.projectRoot, modelID, updated, api.config.VRAMBudgetGB, func() error {
		if _, applyErr := api.applyUpdatedModelRuntime(modelID, current, updated, false); applyErr != nil {
			return applyErr
		}
		return errors.New("real lifecycle verification failure")
	}, func() error {
		return api.rollbackUpdatedModelRuntime(modelID, current, false)
	})
	if err == nil {
		t.Fatal("runtime failure was swallowed")
	}
	after, readErr := os.ReadFile(path)
	if readErr != nil || !bytes.Equal(before, after) {
		t.Fatalf("rollback did not restore exact config bytes: equal=%v error=%v", bytes.Equal(before, after), readErr)
	}
	instances := api.mgr.GetModelInstances(modelID)
	if len(instances) != 1 || instances[0] != original || original.cmd.Process.Pid != originalPID || original.State() != "loaded" {
		t.Fatal("rollback did not preserve original loaded worker")
	}
	err = persistModelConfigTransaction(api.projectRoot, modelID, updated, api.config.VRAMBudgetGB, func() error {
		if _, applyErr := api.applyUpdatedModelRuntime(modelID, current, updated, false); applyErr != nil {
			return applyErr
		}
		panic("unexpected lifecycle panic")
	}, func() error {
		return api.rollbackUpdatedModelRuntime(modelID, current, false)
	})
	if err == nil || !strings.Contains(err.Error(), "unexpected lifecycle panic") {
		t.Fatalf("runtime panic was not converted to an error: %v", err)
	}
	after, readErr = os.ReadFile(path)
	if readErr != nil || !bytes.Equal(before, after) {
		t.Fatalf("panic rollback did not restore exact config bytes: equal=%v error=%v", bytes.Equal(before, after), readErr)
	}
	instances = api.mgr.GetModelInstances(modelID)
	if len(instances) != 1 || instances[0] != original || original.cmd.Process.Pid != originalPID || original.State() != "loaded" {
		t.Fatal("panic rollback did not preserve original loaded worker")
	}
}

func makeConfigStorageUnwritable(t *testing.T, projectRoot string) func() {
	t.Helper()
	directory := filepath.Join(projectRoot, "local")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(directory, 0o555); err != nil {
		t.Fatal(err)
	}
	restored := false
	return func() {
		if restored {
			return
		}
		restored = true
		if err := os.Chmod(directory, 0o755); err != nil {
			t.Errorf("restore config directory permissions: %v", err)
		}
	}
}

// TestRegisterLLMSetsDefaultMaxRuntimeSec verifies that POST /v1/llm/models
// without an explicit max_runtime_seconds sets a non-zero default (600s) so
// the watchdog will never immediately kill the job due to maxSec==0.
func TestRegisterLLMSetsDefaultMaxRuntimeSec(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	workerPath := filepath.Join(api.projectRoot, "llm-worker")
	script := `#!/usr/bin/env python3
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    cmd = msg.get("cmd")
    req_id = msg.get("req_id", "_default")
    if cmd == "load":
        print(json.dumps({"status": "ok", "req_id": req_id}), flush=True)
    elif cmd in ("unload", "shutdown"):
        print(json.dumps({"status": "ok", "req_id": req_id}), flush=True)
        break
`
	if err := os.WriteFile(workerPath, []byte(script), 0o755); err != nil {
		t.Fatalf("write worker: %v", err)
	}

	// Register without specifying max_runtime_seconds.
	body := map[string]any{
		"name":       "my-dynamic-llm",
		"hf_model":   "example/my-dynamic-llm-GGUF",
		"worker_cmd": []string{workerPath},
	}
	raw, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/v1/llm/models", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	api.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("register LLM status = %d, body = %s", rec.Code, rec.Body.String())
	}

	modelID := llmModelID("my-dynamic-llm")
	cfg, ok := api.config.Models[modelID]
	if !ok {
		t.Fatalf("model %q not found in config after registration", modelID)
	}
	if cfg.MaxRuntimeSec == 0 {
		t.Fatal("MaxRuntimeSec must not be 0 for dynamically-registered LLM; watchdog would kill all jobs immediately")
	}
	// Default should be the 600s we set.
	if cfg.MaxRuntimeSec != 600 {
		t.Fatalf("MaxRuntimeSec = %d, want 600 (default for dynamic LLMs)", cfg.MaxRuntimeSec)
	}
}

// TestRegisterLLMRespectsExplicitMaxRuntimeSec verifies that when the caller
// provides max_runtime_seconds the value is honoured.
func TestRegisterLLMRespectsExplicitMaxRuntimeSec(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	workerPath := filepath.Join(api.projectRoot, "llm-worker")
	script := `#!/usr/bin/env python3
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    cmd = msg.get("cmd")
    req_id = msg.get("req_id", "_default")
    if cmd == "load":
        print(json.dumps({"status": "ok", "req_id": req_id}), flush=True)
    elif cmd in ("unload", "shutdown"):
        print(json.dumps({"status": "ok", "req_id": req_id}), flush=True)
        break
`
	if err := os.WriteFile(workerPath, []byte(script), 0o755); err != nil {
		t.Fatalf("write worker: %v", err)
	}

	maxSec := 1800
	body := map[string]any{
		"name":                "my-custom-timeout-llm",
		"hf_model":            "example/my-custom-timeout-llm-GGUF",
		"worker_cmd":          []string{workerPath},
		"max_runtime_seconds": maxSec,
	}
	raw, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/v1/llm/models", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	api.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("register LLM status = %d, body = %s", rec.Code, rec.Body.String())
	}

	modelID := llmModelID("my-custom-timeout-llm")
	cfg, ok := api.config.Models[modelID]
	if !ok {
		t.Fatalf("model %q not found in config after registration", modelID)
	}
	if cfg.MaxRuntimeSec != maxSec {
		t.Fatalf("MaxRuntimeSec = %d, want %d (explicit value must be preserved)", cfg.MaxRuntimeSec, maxSec)
	}
}

func TestChatCompletionStreamReservesInstanceWhileStreaming(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	// Streaming chat now goes through the queue like everything else, so the
	// scheduler must be running to dispatch the job and hand the picked
	// instance back to the API handler via the stream-handoff registry. It is
	// started AFTER the model config map is populated below — sched.Run reads
	// config.Models (rescoreAll) on a separate goroutine, so writing the map
	// after launching it would be a concurrent map write (data race).
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	upstreamDone := make(chan struct{})
	upstreamRelease := make(chan struct{})
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer func() {
		if err := listener.Close(); err != nil && !errors.Is(err, net.ErrClosed) {
			t.Errorf("close listener: %v", err)
		}
	}()

	upstream := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			close(upstreamDone)
			w.Header().Set("Content-Type", "text/event-stream")
			flusher, _ := w.(http.Flusher)
			_, _ = io.WriteString(w, "data: hello\n\n")
			if flusher != nil {
				flusher.Flush()
			}
			<-upstreamRelease
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}),
	}
	defer func() {
		if err := upstream.Close(); err != nil {
			t.Errorf("close upstream: %v", err)
		}
	}()
	go func() {
		if err := upstream.Serve(listener); err != nil && err != http.ErrServerClosed {
			t.Errorf("serve upstream: %v", err)
		}
	}()

	streamPort := listener.Addr().(*net.TCPAddr).Port
	workerPath := filepath.Join(api.projectRoot, "llm-worker")
	writeStreamingProtocolWorker(t, workerPath, streamPort)

	pressure := 0.5
	api.config.Models["llm:test-stream"] = ModelConfig{
		MemoryGB:      1,
		MaxConcurrent: 1,
		MaxInstances:  intPtr(1),
		PressureIndex: &pressure,
		WorkerCmd:     []string{workerPath},
		AdapterParams: map[string]string{"LLM_BACKEND": "llamacpp"},
	}
	api.mgr.ScaleModel("llm:test-stream", 1, api.config.Models["llm:test-stream"])

	// Config map is fully populated — now it's safe to start the scheduler
	// goroutine that reads it.
	go api.scheduler.Run(ctx)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{"stream":true}`)))
	rec := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		api.chatCompletionStream(rec, req, "llm:test-stream", []byte(`{"stream":true}`))
		close(done)
	}()

	select {
	case <-upstreamDone:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for upstream stream to start")
	}

	insts := api.mgr.GetModelInstances("llm:test-stream")
	if len(insts) != 1 {
		t.Fatalf("instances = %d, want 1", len(insts))
	}
	if got := insts[0].ActiveJobs(); got != 1 {
		t.Fatalf("active jobs during stream = %d, want 1", got)
	}

	close(upstreamRelease)

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for stream handler to finish")
	}
	streamJobs, err := api.store.ListJobs("", "llm:test-stream", 10)
	if err != nil || len(streamJobs) != 1 || streamJobs[0].RequestedModel != "test-stream" {
		t.Fatalf("stream requested_model persistence: jobs=%+v error=%v", streamJobs, err)
	}

	// Wait briefly for the scheduler's dispatch goroutine to run its deferred
	// ReleaseAndCheck (decrementing activeJobs).
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if insts[0].ActiveJobs() == 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if got := insts[0].ActiveJobs(); got != 0 {
		t.Fatalf("active jobs after stream = %d, want 0", got)
	}
}
