package main

import (
	"path/filepath"
	"testing"
)

func TestNimbleScoreUsesSanctionedCudaWorker(t *testing.T) {
	const modelID = "nimble-scorer"
	const jobType = "nimble-score"
	if isDisabledStillImageModel(modelID) {
		t.Fatal("nimble-scorer must not be classified as a disabled still-image model")
	}
	root := t.TempDir()
	command := []string{
		filepath.Join(root, "venvs", "nimble-scorer", "bin", "python"),
		"-m", "arbiter.worker_main", modelID,
	}
	if err := validatePythonWorkerCommand(root, modelID, command); err != nil {
		t.Fatalf("nimble-scorer worker command rejected: %v", err)
	}
	if err := validateJobModelCompatibility(jobType, modelID); err != nil {
		t.Fatalf("nimble-score job rejected: %v", err)
	}
	if got := JobTypeToModel[jobType]; got != modelID {
		t.Fatalf("JobTypeToModel[%s] = %q, want %q", jobType, got, modelID)
	}
	if venv, ok := trustedPythonAdapters[modelID]; !ok || venv != modelID {
		t.Fatalf("trustedPythonAdapters[%s] = %q ok=%v, want %q", modelID, venv, ok, modelID)
	}
}
