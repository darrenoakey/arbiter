package main

import (
	"path/filepath"
	"testing"
)

func TestMiniAgiIsASanctionedTrainer(t *testing.T) {
	const modelID = "mini-agi"
	const jobType = "mini-agi-read"
	if isDisabledStillImageModel(modelID) {
		t.Fatal("mini-agi must not be classified as a disabled still-image model")
	}
	root := t.TempDir()
	cfg := ModelConfig{MemoryGB: 16, MaxRuntimeSec: 2100, ModelPath: filepath.Join(root, "training", "mini-agi")}
	if err := validateModelWorkerPolicy(root, modelID, cfg, true); err != nil {
		t.Fatalf("mini-agi default worker rejected: %v", err)
	}
	if err := validateJobModelCompatibility(jobType, modelID); err != nil {
		t.Fatalf("mini-agi job rejected: %v", err)
	}
	if got := JobTypeToModel[jobType]; got != modelID {
		t.Fatalf("JobTypeToModel[%s] = %q", jobType, got)
	}
	if venv, ok := trustedPythonAdapters[modelID]; !ok || venv != "" {
		t.Fatalf("trustedPythonAdapters[%s] = %q ok=%v; want the main venv", modelID, venv, ok)
	}
}
