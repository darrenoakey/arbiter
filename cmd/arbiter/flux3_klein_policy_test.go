package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// FLUX 3 Image Klein is an owner-directed benchmark (2026-10-16): three
// exact precision ids, each reachable only through its own job type from
// venvs/flux3. These tests pin that the opening is exactly that narrow and
// every other flux* path stays denied.

var flux3KleinIDs = []string{flux3KleinBF16Model, flux3KleinFP8Model, flux3KleinNVFP4Model}

func flux3KleinConfig(root, modelID, weightsRoot string) ModelConfig {
	return ModelConfig{
		MemoryGB:      36,
		MaxConcurrent: 1,
		MaxInstances:  intPtr(1),
		ModelPath:     "/home/darren/nvme-models/flux3/flux-3-image-klein",
		Placements:    []string{"spark"},
		AdapterParams: map[string]string{
			flux3WeightsRootParam:     weightsRoot,
			"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
		},
		WorkerCmd: []string{
			filepath.Join(root, "venvs", "flux3", "bin", "python"),
			"-m", "arbiter.worker_main", modelID,
		},
	}
}

func TestFlux3KleinExactVariantsAreAllowed(t *testing.T) {
	root := t.TempDir()
	weights := t.TempDir()
	for _, modelID := range flux3KleinIDs {
		if isDisabledStillImageModel(modelID) {
			t.Fatalf("%s classified as a disabled still-image model", modelID)
		}
		cfg := flux3KleinConfig(root, modelID, weights)
		if disabledStillImageConfig(modelID, cfg) {
			t.Fatalf("%s config naming its flux3 weights and venv was refused", modelID)
		}
		if err := validateModelWorkerPolicy(root, modelID, cfg, true); err != nil {
			t.Fatalf("%s registration rejected: %v", modelID, err)
		}
		if err := validateJobModelCompatibility(modelID, modelID); err != nil {
			t.Fatalf("%s routing rejected: %v", modelID, err)
		}
		if got := JobTypeToModel[modelID]; got != modelID {
			t.Fatalf("JobTypeToModel[%s] = %q", modelID, got)
		}
		if venv := trustedPythonAdapters[modelID]; venv != "flux3" {
			t.Fatalf("trustedPythonAdapters[%s] = %q, want flux3", modelID, venv)
		}
	}
}

func TestFlux3KleinVariantsCannotCrossOrBeSmuggled(t *testing.T) {
	root := t.TempDir()
	weights := t.TempDir()
	for _, modelID := range flux3KleinIDs {
		// A variant cannot serve another variant's (or any other) job type.
		for _, jobType := range append([]string{"image-generate", "image-edit", "qwen-image", "reference-image-edit", "background-remove", "flux3", "flux3-klein"}, flux3KleinIDs...) {
			if jobType == modelID {
				continue
			}
			if err := rejectDisabledStillImage(jobType, modelID); err == nil {
				t.Fatalf("job type %q accepted %s", jobType, modelID)
			}
			if err := validateJobModelCompatibility(jobType, modelID); err == nil {
				t.Fatalf("job type %q routed to %s", jobType, modelID)
			}
		}
		// The job type cannot be redirected to another adapter.
		for _, other := range []string{"flux2", "qwen-image-2.1", "reference-image-edit", "birefnet", modelID + "-lora"} {
			if err := validateJobModelCompatibility(modelID, other); err == nil {
				t.Fatalf("%s job routed to %q", modelID, other)
			}
		}
		// A worker command selecting a different adapter or venv is untrusted.
		for _, mutate := range []func([]string){
			func(c []string) { c[3] = "flux2" },
			func(c []string) { c[0] = filepath.Join(root, "venvs", "flux2", "bin", "python") },
		} {
			cfg := flux3KleinConfig(root, modelID, weights)
			mutate(cfg.WorkerCmd)
			if err := validatePythonWorkerCommand(root, modelID, cfg.WorkerCmd); err == nil ||
				!strings.Contains(err.Error(), untrustedWorkerCommandMessage) {
				t.Fatalf("%s accepted mutated worker %v: %v", modelID, cfg.WorkerCmd, err)
			}
		}
		for _, other := range flux3KleinIDs {
			if other == modelID {
				continue
			}
			cfg := flux3KleinConfig(root, other, weights)
			if err := validatePythonWorkerCommand(root, modelID, cfg.WorkerCmd); err == nil {
				t.Fatalf("%s accepted the %s worker command", modelID, other)
			}
		}
	}
	// Aliases and near neighbours stay denied, including the flux3 config
	// under a non-exact id.
	for _, alias := range []string{"flux3", "flux3-klein", "flux3-klein-base", "flux3-klein-int8", "FLUX3-KLEIN-BF16-lora", "black-forest-labs/flux-3-image-klein", "flux-3-image-klein"} {
		if !isDisabledStillImageModel(alias) {
			t.Fatalf("%q was not denied", alias)
		}
		if !disabledStillImageConfig(alias, flux3KleinConfig(root, flux3KleinBF16Model, weights)) {
			t.Fatalf("flux3 config survived under alias %q", alias)
		}
		if _, ok := JobTypeToModel[alias]; ok {
			t.Fatalf("alias %q is a job type", alias)
		}
	}
}

func TestFlux3WeightsRootParamIsScopedAndValidated(t *testing.T) {
	root := t.TempDir()
	weights := t.TempDir()
	file := filepath.Join(weights, "file")
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(t.TempDir(), "link")
	if err := os.Symlink(weights, link); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{"relative/flux3", weights + "/../" + filepath.Base(weights), link, file, filepath.Join(weights, "missing")} {
		cfg := flux3KleinConfig(root, flux3KleinFP8Model, bad)
		if err := validateAdapterParams(root, flux3KleinFP8Model, cfg); err == nil ||
			!strings.Contains(err.Error(), untrustedAdapterParamsMessage) {
			t.Fatalf("FLUX3_WEIGHTS_ROOT %q accepted: %v", bad, err)
		}
	}
	// No other torch worker may receive the flux3 weights root.
	for _, other := range []string{"qwen-image-2.1", "iris-3b", "birefnet"} {
		cfg := ModelConfig{AdapterParams: map[string]string{flux3WeightsRootParam: weights}}
		if err := validateAdapterParams(root, other, cfg); err == nil {
			t.Fatalf("%s accepted %s", other, flux3WeightsRootParam)
		}
	}
	// And flux3 still rejects unsanctioned keys.
	cfg := flux3KleinConfig(root, flux3KleinNVFP4Model, weights)
	cfg.AdapterParams["WEIGHTS_DIR"] = weights
	if err := validateAdapterParams(root, flux3KleinNVFP4Model, cfg); err == nil {
		t.Fatal("flux3 accepted WEIGHTS_DIR")
	}
}
