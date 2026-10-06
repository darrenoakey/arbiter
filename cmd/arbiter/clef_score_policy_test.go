package main

import (
	"path/filepath"
	"testing"
)

func TestClefScoreUsesSanctionedCudaWorker(t *testing.T) {
	root := t.TempDir()
	for jobType, modelID := range map[string]string{"clef-score": "clef-scorer", "clef-flash-score": "clef-flash-scorer"} {
		if isDisabledStillImageModel(modelID) {
			t.Fatalf("%s must not be classified as a disabled still-image model", modelID)
		}
		command := []string{
			filepath.Join(root, "venvs", "clef-scorer", "bin", "python"),
			"-m", "arbiter.worker_main", modelID,
		}
		if err := validatePythonWorkerCommand(root, modelID, command); err != nil {
			t.Fatalf("%s worker command rejected: %v", modelID, err)
		}
		if err := validateJobModelCompatibility(jobType, modelID); err != nil {
			t.Fatalf("%s job rejected: %v", jobType, err)
		}
		if got := JobTypeToModel[jobType]; got != modelID {
			t.Fatalf("JobTypeToModel[%s] = %q, want %q", jobType, got, modelID)
		}
	}
}
