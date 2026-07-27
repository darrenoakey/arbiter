package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func postJob(t *testing.T, api *API, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/jobs", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()
	api.Handler().ServeHTTP(rec, req)
	return rec
}

func TestSubmitJobRejectsDisabledStillImageTypesAndOverrides(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	for _, id := range []string{"flux2", "flux-schnell", "z-image-turbo", "FLUX_KONTEXT-lora"} {
		api.config.Models[id] = ModelConfig{MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1)}
	}

	cases := map[string]string{
		"generate type":       `{"type":"image-generate","params":{"prompt":"test"}}`,
		"edit type":           `{"type":"image-edit","params":{"image":"x"}}`,
		"top-level flux2":     `{"type":"background-remove","model":"flux2","params":{"image":"x"}}`,
		"nested flux schnell": `{"type":"background-remove","params":{"model":"flux-schnell","image":"x"}}`,
		"unconfigured alias":  `{"type":"caption","model":"black-forest-labs/FLUX.2-klein-9B","params":{"image":"x"}}`,
		"prefix and LoRA":     `{"type":"background-remove","params":{"model":"FLUX_KONTEXT-lora","image":"x"}}`,
		"z-image alias":       `{"type":"query","model":"Tongyi-MAI/Z_Image_Turbo","params":{"image":"x","question":"?"}}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			rec := postJob(t, api, body)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), stillImageDisabledMessage) {
				t.Fatalf("body does not state owner policy: %s", rec.Body.String())
			}
		})
	}
}

func TestSubmitJobPreservesBiRefNetAndLTX2Variants(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	pressure := 1.0
	for _, id := range []string{"birefnet", "birefnet-v2", "ltx2", "ltx2-dev", "ltx2-dev-denoise2"} {
		api.config.Models[id] = ModelConfig{
			MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1), PressureIndex: &pressure,
		}
	}
	api.refreshAliasModels()

	for name, body := range map[string]string{
		"birefnet":         `{"type":"background-remove","params":{"image":"eA=="}}`,
		"birefnet variant": `{"type":"background-remove","model":"birefnet-v2","params":{"image":"eA=="}}`,
		"ltx2":             `{"type":"video-generate","model":"ltx2","params":{}}`,
		"ltx2 variant":     `{"type":"video-generate","model":"ltx2-dev","params":{}}`,
		"ltx2 dev denoise": `{"type":"video-denoise2","model":"ltx2-dev-denoise2","params":{}}`,
	} {
		t.Run(name, func(t *testing.T) {
			rec := postJob(t, api, body)
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
			}
		})
	}
}

func TestSubmitJobRejectsIncompatibleNonImageOverride(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	api.config.Models["birefnet"] = ModelConfig{MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1)}
	api.refreshAliasModels()
	rec := postJob(t, api, `{"type":"video-generate","model":"birefnet","params":{}}`)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "not compatible") {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
}

func TestSubmitJobKeepsSemanticVoiceAndModelFieldsOutOfRouting(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	for _, modelID := range []string{"rvc-convert", "tts-custom", "embed-text"} {
		api.config.Models[modelID] = ModelConfig{MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1)}
	}
	for name, body := range map[string]string{
		"rvc flora voice": `{"type":"rvc-convert","params":{"model":"flora"}}`,
		"rvc flux voice":  `{"type":"rvc-convert","params":{"model":"flux2"}}`,
		"tts flora model": `{"type":"tts-custom","params":{"model":"floral-voice"}}`,
		"embedding model": `{"type":"embed-text","params":{"model":"sentence-transformers/flora"}}`,
	} {
		t.Run(name, func(t *testing.T) {
			response := postJob(t, api, body)
			if response.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
			}
		})
	}
}

func TestModelRegistrationAndReloadRejectStillGenerators(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	for name, body := range map[string]string{
		"id":        `{"model_id":"flux2","memory_gb":1}`,
		"hf repo":   `{"model_id":"opaque","memory_gb":1,"auto_download":"black-forest-labs/FLUX.1-schnell"}`,
		"worker":    `{"model_id":"opaque-worker","memory_gb":1,"worker_cmd":["python","flux_server.py"]}`,
		"LoRA path": `{"model_id":"opaque-lora","memory_gb":1,"model_path":"/models/product-lora.safetensors"}`,
	} {
		t.Run(name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/models", bytes.NewBufferString(body))
			rec := httptest.NewRecorder()
			api.Handler().ServeHTTP(rec, req)
			if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), stillImageDisabledMessage) {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
			}
		})
	}

	api.config.Models["flux2"] = ModelConfig{MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(0)}
	reloadReq := httptest.NewRequest(http.MethodPost, "/v1/models/flux2/reload", nil)
	reloadRec := httptest.NewRecorder()
	api.Handler().ServeHTTP(reloadRec, reloadReq)
	if reloadRec.Code != http.StatusBadRequest || !strings.Contains(reloadRec.Body.String(), stillImageDisabledMessage) {
		t.Fatalf("reload status = %d, body = %s", reloadRec.Code, reloadRec.Body.String())
	}
}

func TestModelRegistrationAndReloadRejectUntrustedWorkers(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	req := httptest.NewRequest(http.MethodPost, "/v1/models", bytes.NewBufferString(
		`{"model_id":"birefnet","memory_gb":1,"worker_cmd":["sh","-c","true"]}`))
	rec := httptest.NewRecorder()
	api.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), untrustedWorkerCommandMessage) {
		t.Fatalf("registration status = %d, body = %s", rec.Code, rec.Body.String())
	}

	api.config.Models["birefnet"] = ModelConfig{
		MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(0), WorkerCmd: []string{"/tmp/llm-worker"},
	}
	reloadReq := httptest.NewRequest(http.MethodPost, "/v1/models/birefnet/reload", nil)
	reloadRec := httptest.NewRecorder()
	api.Handler().ServeHTTP(reloadRec, reloadReq)
	if reloadRec.Code != http.StatusBadRequest || !strings.Contains(reloadRec.Body.String(), untrustedWorkerCommandMessage) {
		t.Fatalf("reload status = %d, body = %s", reloadRec.Code, reloadRec.Body.String())
	}
}
