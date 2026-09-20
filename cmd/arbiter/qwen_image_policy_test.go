package main

import (
	"path/filepath"
	"strings"
	"testing"
)

// The Qwen unified adapter is sanctioned exception #2 (owner decision,
// 2026-09-21): a local text-to-image + reference-editing model exposed only
// through the `qwen-image` job type. These tests pin that the carve-out is
// exactly that narrow — its own job type and model id only — and that every
// other still-image path stays disabled.

func qwenImage21Config(root string) ModelConfig {
	return ModelConfig{
		MemoryGB:      44,
		MaxConcurrent: 1,
		MaxInstances:  intPtr(1),
		AutoDownload:  "Qwen/Qwen-Image-2.1",
		WorkerCmd: []string{
			filepath.Join(root, "venvs", "qwenimage", "bin", "python"),
			"-m", "arbiter.worker_main", qwenImage21Model,
		},
	}
}

func TestQwenImage21IsAllowedByPolicy(t *testing.T) {
	if isDisabledStillImageModel(qwenImage21Model) {
		t.Fatal("qwen-image-2.1 must not be classified as a disabled still-image model")
	}
	root := t.TempDir()
	// The venv python path does not exist in tests; bypass the existence
	// probing the same way the reference editor test does by validating the
	// policy layers that matter here.
	cfg := qwenImage21Config(root)
	if disabledStillImageConfig(qwenImage21Model, cfg) {
		t.Fatal("qwen-image-2.1 config naming its Qwen checkpoint and venv was refused")
	}
	if err := validatePythonWorkerCommand(root, qwenImage21Model, cfg.WorkerCmd); err != nil {
		t.Fatalf("qwen-image-2.1 worker command rejected: %v", err)
	}
	if err := rejectDisabledStillImage(qwenImageJobType, qwenImage21Model); err != nil {
		t.Fatalf("qwen-image job type rejected: %v", err)
	}
	if err := validateJobModelCompatibility(qwenImageJobType, qwenImage21Model); err != nil {
		t.Fatalf("qwen-image routing rejected: %v", err)
	}
	if got := JobTypeToModel[qwenImageJobType]; got != qwenImage21Model {
		t.Fatalf("JobTypeToModel[qwen-image] = %q, want %q", got, qwenImage21Model)
	}
	if venv, ok := trustedPythonAdapters[qwenImage21Model]; !ok || venv != "qwenimage" {
		t.Fatalf("trustedPythonAdapters[qwen-image-2.1] = %q ok=%v, want qwenimage", venv, ok)
	}
}

func TestQwenImage21ExceptionIsNarrow(t *testing.T) {
	root := t.TempDir()
	// Near-neighbour aliases under any id stay disabled.
	for _, modelID := range []string{"qwen-image-2.1-lora", "qwen-image", "Qwen/Qwen-Image-2.1", "qwen-image-edit"} {
		if !isDisabledStillImageModel(modelID) {
			t.Fatalf("%q was not classified as a disabled still-image model", modelID)
		}
	}
	// The sanctioned model cannot be smuggled in under other job types —
	// including the legacy image markers.
	for _, jobType := range []string{"image-edit", "image-generate", "background-remove", "reference-image-edit"} {
		if err := rejectDisabledStillImage(jobType, qwenImage21Model); err == nil {
			t.Fatalf("job type %q accepted the qwen adapter as an override", jobType)
		}
	}
	// And the qwen job type cannot be redirected to another adapter.
	for _, modelID := range []string{"flux2", "reference-image-edit", "birefnet"} {
		if err := validateJobModelCompatibility(qwenImageJobType, modelID); err == nil {
			t.Fatalf("qwen-image routed to %q", modelID)
		}
	}
	// A worker command that selects a different adapter is still untrusted.
	mismatched := qwenImage21Config(root)
	mismatched.WorkerCmd[3] = "flux2"
	if err := validatePythonWorkerCommand(root, qwenImage21Model, mismatched.WorkerCmd); err == nil ||
		!strings.Contains(err.Error(), untrustedWorkerCommandMessage) {
		t.Fatalf("qwen-image-2.1 accepted a worker command selecting the flux2 adapter: %v", err)
	}
	// The config cannot be smuggled in under a non-whitelisted model id: the
	// auto_download marker trips the still-image config check.
	disguised := qwenImage21Config(root)
	if !disabledStillImageConfig("qwen-image-2.1-lora", disguised) {
		t.Fatal("qwen config under a lora alias survived the still-image config check")
	}
}

func TestQwenImage21ModelRegistrationRoundTrip(t *testing.T) {
	// POST /v1/models and the persisted config both go through
	// ApplyModelConfig; the sanctioned shape must survive worker policy.
	root := t.TempDir()
	cfg := qwenImage21Config(root)
	if err := validateModelWorkerPolicy(root, qwenImage21Model, cfg, true); err != nil {
		t.Fatalf("qwen-image-2.1 model registration rejected: %v", err)
	}
}
