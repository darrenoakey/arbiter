package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func minimaxH3ImageConfig(root string) ModelConfig {
	return ModelConfig{
		MemoryGB: 75, MaxConcurrent: 1, MaxInstances: intPtr(1),
		ModelPath: "/home/darren/nvme-models/minimax-h3-image/MiniMax-H3-Image", Placements: []string{"spark"},
		AdapterParams: map[string]string{"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
		WorkerCmd:     []string{filepath.Join(root, "venvs", "minimaxh3image", "bin", "python"), "-m", "arbiter.worker_main", minimaxH3ImageModel},
	}
}

func TestMiniMaxH3ImageExactLocalComparisonIsAllowed(t *testing.T) {
	root := t.TempDir()
	cfg := minimaxH3ImageConfig(root)
	if isDisabledStillImageModel(minimaxH3ImageModel) || disabledStillImageConfig(minimaxH3ImageModel, cfg) {
		t.Fatal("local MiniMax H3 still-image comparison was refused")
	}
	if err := validateModelWorkerPolicy(root, minimaxH3ImageModel, cfg, true); err != nil {
		t.Fatalf("registration rejected: %v", err)
	}
	if err := validateJobModelCompatibility(minimaxH3ImageJobType, minimaxH3ImageModel); err != nil {
		t.Fatalf("exact route rejected: %v", err)
	}
	if got := JobTypeToModel[minimaxH3ImageJobType]; got != minimaxH3ImageModel {
		t.Fatalf("route = %q", got)
	}
	if trustedPythonAdapters[minimaxH3ImageModel] != "minimaxh3image" {
		t.Fatal("wrong venv binding")
	}
}

func TestMiniMaxH3ImageCannotBeCrossedOrSmuggled(t *testing.T) {
	root := t.TempDir()
	for _, jobType := range []string{"video-generate-h3", "image-generate", "qwen-image", "moss-image", "p-h3-image-comfy"} {
		if err := rejectDisabledStillImage(jobType, minimaxH3ImageModel); err == nil {
			t.Fatalf("%s accepted MiniMax image model", jobType)
		}
		if err := validateJobModelCompatibility(jobType, minimaxH3ImageModel); err == nil {
			t.Fatalf("%s routed to MiniMax image model", jobType)
		}
	}
	for _, model := range []string{"minimax-h3-local", "minimax-fast-h3", "qwen-image-2.1", "minimax-h3-image-bf16"} {
		if err := validateJobModelCompatibility(minimaxH3ImageJobType, model); err == nil {
			t.Fatalf("MiniMax image job accepted %s", model)
		}
	}
	for _, mutate := range []func([]string){
		func(c []string) { c[3] = "minimax-h3-local" },
		func(c []string) { c[0] = filepath.Join(root, "venvs", "ph3comfy", "bin", "python") },
	} {
		cfg := minimaxH3ImageConfig(root)
		mutate(cfg.WorkerCmd)
		if err := validatePythonWorkerCommand(root, minimaxH3ImageModel, cfg.WorkerCmd); err == nil || !strings.Contains(err.Error(), untrustedWorkerCommandMessage) {
			t.Fatalf("mutated worker accepted: %v", err)
		}
	}
	for _, alias := range []string{"MiniMaxAI/MiniMax-H3-Image", "minimax-h3-image-bf16", "minimax-h3-image-nvfp4"} {
		if !isDisabledStillImageModel(alias) || !disabledStillImageConfig(alias, minimaxH3ImageConfig(root)) {
			t.Fatalf("alias accepted: %s", alias)
		}
	}
}
