package main

import (
	"path/filepath"
	"testing"
)

// The qlabs noise trainer is a training job: it must resolve to its own
// model id, ride the parent's isolated qlabsnoise venv inside the audited
// capped worker, and never be classified as a disabled still-image model.
func TestQlabsNoiseTrainUsesSanctionedCudaWorker(t *testing.T) {
	root := t.TempDir()
	const modelID = "qlabs-noise-train"
	if isDisabledStillImageModel(modelID) {
		t.Fatalf("%s must not be classified as a disabled still-image model", modelID)
	}
	command := []string{
		filepath.Join(root, "venvs", "qlabsnoise", "bin", "python"),
		"-m", "arbiter.worker_main", modelID,
	}
	if err := validatePythonWorkerCommand(root, modelID, command); err != nil {
		t.Fatalf("%s worker command rejected: %v", modelID, err)
	}
	if err := validateJobModelCompatibility("qlabs-noise-train", modelID); err != nil {
		t.Fatalf("qlabs-noise-train job rejected: %v", err)
	}
	if got := JobTypeToModel["qlabs-noise-train"]; got != modelID {
		t.Fatalf("JobTypeToModel[qlabs-noise-train] = %q, want %q", got, modelID)
	}
	if venv := trustedPythonAdapters[modelID]; venv != "qlabsnoise" {
		t.Fatalf("trustedPythonAdapters[%s] = %q, want %q", modelID, venv, "qlabsnoise")
	}
}

// The training model must not accept arbitrary worker configurations: only
// the exact sanctioned venv identity passes, and the declared memory cap
// stays at the agreed 8GB initial reservation.
func TestQlabsNoiseTrainRejectsUntrustedWorkerConfigurations(t *testing.T) {
	root := t.TempDir()
	const modelID = "qlabs-noise-train"
	for name, command := range map[string][]string{
		"wrong venv": {
			filepath.Join(root, "venvs", "other", "bin", "python"),
			"-m", "arbiter.worker_main", modelID,
		},
		"arbitrary module": {
			filepath.Join(root, "venvs", "qlabsnoise", "bin", "python"),
			"-m", "arbiter.worker_main", "mini-agi",
		},
		"script execution": {
			filepath.Join(root, "venvs", "qlabsnoise", "bin", "python"),
			filepath.Join(root, "train.py"),
		},
	} {
		if err := validateWorkerCommand(root, modelID, ModelConfig{WorkerCmd: command}); err == nil {
			t.Fatalf("%s worker command accepted: %v", name, command)
		}
	}
	if err := validateJobModelCompatibility("qlabs-noise-train", "mini-agi"); err == nil {
		t.Fatal("qlabs-noise-train job must not be routable to a different adapter")
	}
}
