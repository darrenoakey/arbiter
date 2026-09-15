package main

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The reference editor is the ONE sanctioned still-image exception: a
// reference-conditioned local editor that renders targets for other
// pipelines. These tests pin that the carve-out is exactly that narrow —
// its own job type and model id only — and that every other still-image
// path stays disabled.

func referenceImageEditConfig(root string) ModelConfig {
	return ModelConfig{
		MemoryGB:      36,
		MaxConcurrent: 1,
		MaxInstances:  intPtr(1),
		AutoDownload:  "black-forest-labs/FLUX.2-klein-9B",
		WorkerCmd: []string{
			filepath.Join(root, "venvs", "flux2", "bin", "python"),
			"-m", "arbiter.worker_main", referenceImageEditModel,
		},
	}
}

func TestReferenceImageEditIsAllowedByPolicy(t *testing.T) {
	if isDisabledStillImageModel(referenceImageEditModel) {
		t.Fatal("reference-image-edit must not be classified as a disabled still-image model")
	}
	root := t.TempDir()
	cfg := referenceImageEditConfig(root)
	if disabledStillImageConfig(referenceImageEditModel, cfg) {
		t.Fatal("reference-image-edit config naming its FLUX checkpoint and flux2 venv was refused")
	}
	if err := validateModelWorkerPolicy(root, referenceImageEditModel, cfg, true); err != nil {
		t.Fatalf("reference-image-edit worker policy rejected: %v", err)
	}
	if err := rejectDisabledStillImage("reference-image-edit", referenceImageEditModel); err != nil {
		t.Fatalf("reference-image-edit job type rejected: %v", err)
	}
	if err := validateJobModelCompatibility("reference-image-edit", referenceImageEditModel); err != nil {
		t.Fatalf("reference-image-edit routing rejected: %v", err)
	}
}

func TestReferenceImageEditExceptionIsNarrow(t *testing.T) {
	root := t.TempDir()
	// The same checkpoint under any other model id stays disabled.
	legacy := referenceImageEditConfig(root)
	legacy.WorkerCmd[3] = "flux2"
	if !disabledStillImageConfig("flux2", legacy) {
		t.Fatal("legacy flux2 model must remain disabled")
	}
	if err := validateModelWorkerPolicy(root, "flux2", legacy, true); err == nil ||
		!strings.Contains(err.Error(), stillImageDisabledMessage) {
		t.Fatalf("legacy flux2 worker policy = %v, want still-image policy error", err)
	}
	// The reference model cannot be smuggled in under the legacy job types
	// or any other job type.
	for _, jobType := range []string{"image-edit", "image-generate", "background-remove", "caption"} {
		if err := rejectDisabledStillImage(jobType, referenceImageEditModel); err == nil {
			t.Fatalf("job type %q accepted the reference editor as an override", jobType)
		}
	}
	// And the reference job type cannot be redirected to another adapter.
	for _, modelID := range []string{"flux2", "flux-schnell", "birefnet"} {
		if err := validateJobModelCompatibility("reference-image-edit", modelID); err == nil {
			t.Fatalf("reference-image-edit routed to %q", modelID)
		}
	}
	// A worker command that selects a different adapter is still untrusted.
	mismatched := referenceImageEditConfig(root)
	mismatched.WorkerCmd[3] = "flux2"
	if err := validateModelWorkerPolicy(root, referenceImageEditModel, mismatched, true); err == nil {
		t.Fatal("reference-image-edit accepted a worker command selecting the legacy flux2 adapter")
	}
}

func TestLoadConfigKeepsReferenceImageEditButDropsLegacyFlux(t *testing.T) {
	root := t.TempDir()
	localDir := filepath.Join(root, "local")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatal(err)
	}
	python := filepath.Join(root, "venvs", "flux2", "bin", "python")
	body := `{"models":{
		"flux2":{"memory_gb":36,"auto_download":"black-forest-labs/FLUX.2-klein-9B"},
		"reference-image-edit":{"memory_gb":36,"auto_download":"black-forest-labs/FLUX.2-klein-9B",
			"worker_cmd":["` + python + `","-m","arbiter.worker_main","reference-image-edit"]},
		"birefnet":{"memory_gb":1}
	}}`
	if err := os.WriteFile(filepath.Join(localDir, "config.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := cfg.Models["flux2"]; ok {
		t.Fatal("legacy flux2 survived startup config filtering")
	}
	if _, ok := cfg.Models[referenceImageEditModel]; !ok {
		t.Fatal("reference-image-edit was removed by startup config filtering")
	}
	if _, ok := cfg.Models["birefnet"]; !ok {
		t.Fatal("birefnet was removed")
	}
}

func TestSubmitReferenceImageEditJob(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	pressure := 1.0
	cfg := referenceImageEditConfig(api.projectRoot)
	cfg.PressureIndex = &pressure
	api.config.Models[referenceImageEditModel] = cfg
	api.refreshAliasModels()

	rec := postJob(t, api, `{"type":"reference-image-edit","params":{"prompt":"pro dslr","image":"eA=="}}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"model":"reference-image-edit"`) {
		t.Fatalf("job did not route to the reference editor: %s", rec.Body.String())
	}

	for name, body := range map[string]string{
		"legacy edit type":      `{"type":"image-edit","model":"reference-image-edit","params":{"image":"eA=="}}`,
		"legacy generate type":  `{"type":"image-generate","model":"reference-image-edit","params":{"prompt":"x"}}`,
		"override to flux2":     `{"type":"reference-image-edit","model":"flux2","params":{"image":"eA=="}}`,
		"smuggled under remove": `{"type":"background-remove","model":"reference-image-edit","params":{"image":"eA=="}}`,
	} {
		t.Run(name, func(t *testing.T) {
			rec := postJob(t, api, body)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
			}
		})
	}
}
