package main

import (
	"path/filepath"
	"testing"
)

func TestRoutingDecideIsASanctionedTrainer(t *testing.T) {
	const modelID = "routing-decide"
	const jobType = "routing-decide-train"
	if isDisabledStillImageModel(modelID) {
		t.Fatal("routing-decide must not be classified as a disabled still-image model")
	}
	root := t.TempDir()
	cfg := ModelConfig{
		MemoryGB:      24,
		MaxRuntimeSec: 14400,
		WorkerCmd: []string{
			filepath.Join(root, "venvs", "routing-decide", "bin", "python"),
			"-m", "arbiter.worker_main", modelID,
		},
	}
	if err := validatePythonWorkerCommand(root, modelID, cfg.WorkerCmd); err != nil {
		t.Fatalf("routing-decide worker command rejected: %v", err)
	}
	if err := validateJobModelCompatibility(jobType, modelID); err != nil {
		t.Fatalf("routing-decide job rejected: %v", err)
	}
	if got := JobTypeToModel[jobType]; got != modelID {
		t.Fatalf("JobTypeToModel[%s] = %q", jobType, got)
	}
	if venv, ok := trustedPythonAdapters[modelID]; !ok || venv != modelID {
		t.Fatalf("trustedPythonAdapters[%s] = %q ok=%v", modelID, venv, ok)
	}
}
