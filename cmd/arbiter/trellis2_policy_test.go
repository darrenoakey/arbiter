package main

import (
	"path/filepath"
	"strings"
	"testing"
)

const trellis2Model = "trellis2"
const imageTo3DJobType = "image-to-3d"

func trellis2Config(root string) ModelConfig {
	return ModelConfig{
		MemoryGB:      48,
		MaxConcurrent: 1,
		MaxInstances:  intPtr(1),
		AutoDownload:  "microsoft/TRELLIS.2-4B",
		AdapterParams: map[string]string{
			"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
		},
		WorkerCmd: []string{
			filepath.Join(root, "venvs", "trellis2", "bin", "python"),
			"-m", "arbiter.worker_main", trellis2Model,
		},
	}
}

func TestTrellis2IsAllowedByPolicy(t *testing.T) {
	if isDisabledStillImageModel(trellis2Model) {
		t.Fatal("trellis2 must not be classified as a disabled still-image model")
	}
	root := t.TempDir()
	cfg := trellis2Config(root)
	if disabledStillImageConfig(trellis2Model, cfg) {
		t.Fatal("trellis2 config naming microsoft/TRELLIS.2-4B was refused")
	}
	if err := validatePythonWorkerCommand(root, trellis2Model, cfg.WorkerCmd); err != nil {
		t.Fatalf("trellis2 worker command rejected: %v", err)
	}
	if err := rejectDisabledStillImage(imageTo3DJobType, trellis2Model); err != nil {
		t.Fatalf("image-to-3d job type rejected: %v", err)
	}
	if err := validateJobModelCompatibility(imageTo3DJobType, trellis2Model); err != nil {
		t.Fatalf("image-to-3d routing rejected: %v", err)
	}
	if got := JobTypeToModel[imageTo3DJobType]; got != trellis2Model {
		t.Fatalf("JobTypeToModel[image-to-3d] = %q, want %q", got, trellis2Model)
	}
	if venv, ok := trustedPythonAdapters[trellis2Model]; !ok || venv != "trellis2" {
		t.Fatalf("trustedPythonAdapters[trellis2] = %q ok=%v, want trellis2", venv, ok)
	}
}

func TestTrellis2JobTypeIsExact(t *testing.T) {
	root := t.TempDir()
	for _, modelID := range []string{"flux2", "reference-image-edit", "qwen-image-2.1", "birefnet"} {
		if err := validateJobModelCompatibility(imageTo3DJobType, modelID); err == nil {
			t.Fatalf("image-to-3d routed to %q", modelID)
		}
	}
	mismatched := trellis2Config(root)
	mismatched.WorkerCmd[3] = "qwen-image-2.1"
	if err := validatePythonWorkerCommand(root, trellis2Model, mismatched.WorkerCmd); err == nil ||
		!strings.Contains(err.Error(), untrustedWorkerCommandMessage) {
		t.Fatalf("trellis2 accepted a worker command selecting another adapter: %v", err)
	}
	wrongVenv := trellis2Config(root)
	wrongVenv.WorkerCmd[0] = filepath.Join(root, "venvs", "qwenimage", "bin", "python")
	if err := validatePythonWorkerCommand(root, trellis2Model, wrongVenv.WorkerCmd); err == nil {
		t.Fatal("trellis2 accepted the qwenimage interpreter")
	}
}

func TestTrellis2ModelRegistrationRoundTrip(t *testing.T) {
	root := t.TempDir()
	cfg := trellis2Config(root)
	if err := validateModelWorkerPolicy(root, trellis2Model, cfg, true); err != nil {
		t.Fatalf("trellis2 model registration rejected: %v", err)
	}
}
