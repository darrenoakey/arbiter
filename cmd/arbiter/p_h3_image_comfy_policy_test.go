package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func ph3ImageComfyConfig(root string) ModelConfig {
	return ModelConfig{
		MemoryGB: 28, MaxConcurrent: 1, MaxInstances: intPtr(1),
		ModelPath: "/mnt/t9/models/p-h3-image", Placements: []string{"spark"},
		AdapterParams: map[string]string{"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
		WorkerCmd:     []string{filepath.Join(root, "venvs", "ph3comfy", "bin", "python"), "-m", "arbiter.worker_main", ph3ImageComfyW4A8Model},
	}
}

func TestPH3ComfyExactW4A8ComparisonIsAllowed(t *testing.T) {
	root := t.TempDir()
	cfg := ph3ImageComfyConfig(root)
	if isDisabledStillImageModel(ph3ImageComfyW4A8Model) || disabledStillImageConfig(ph3ImageComfyW4A8Model, cfg) {
		t.Fatal("official P-H3 Comfy W4A8 comparison was refused")
	}
	if err := validateModelWorkerPolicy(root, ph3ImageComfyW4A8Model, cfg, true); err != nil {
		t.Fatalf("registration rejected: %v", err)
	}
	if err := validateJobModelCompatibility(ph3ImageComfyJobType, ph3ImageComfyW4A8Model); err != nil {
		t.Fatalf("exact route rejected: %v", err)
	}
	if got := JobTypeToModel[ph3ImageComfyJobType]; got != ph3ImageComfyW4A8Model {
		t.Fatalf("route = %q", got)
	}
	if trustedPythonAdapters[ph3ImageComfyW4A8Model] != "ph3comfy" {
		t.Fatal("wrong venv binding")
	}
}

func TestPH3ComfyCannotBeCrossedOrSmuggled(t *testing.T) {
	root := t.TempDir()
	for _, jobType := range []string{"p-h3-image-comfy-w4a8", "image-generate", "qwen-image", "moss-image", "flux3-klein-bf16"} {
		if err := rejectDisabledStillImage(jobType, ph3ImageComfyW4A8Model); err == nil {
			t.Fatalf("%s accepted P-H3 model", jobType)
		}
		if err := validateJobModelCompatibility(jobType, ph3ImageComfyW4A8Model); err == nil {
			t.Fatalf("%s routed to P-H3 model", jobType)
		}
	}
	for _, model := range []string{"qwen-image-2.1", "moss-image-sft", "p-h3-image-comfy-w4a8-w4a8"} {
		if err := validateJobModelCompatibility(ph3ImageComfyJobType, model); err == nil {
			t.Fatalf("P-H3 job accepted %s", model)
		}
	}
	for _, mutate := range []func([]string){
		func(c []string) { c[3] = "moss-image-sft" },
		func(c []string) { c[0] = filepath.Join(root, "venvs", "minimax-h3", "bin", "python") },
	} {
		cfg := ph3ImageComfyConfig(root)
		mutate(cfg.WorkerCmd)
		if err := validatePythonWorkerCommand(root, ph3ImageComfyW4A8Model, cfg.WorkerCmd); err == nil || !strings.Contains(err.Error(), untrustedWorkerCommandMessage) {
			t.Fatalf("mutated worker accepted: %v", err)
		}
	}
	for _, alias := range []string{"p-h3-image-comfy", "PrunaAI/P-H3-Image", "p-h3-image-w4a8", "p-h3-image-comfy", "minimax-h3-image"} {
		if !isDisabledStillImageModel(alias) {
			t.Fatalf("alias accepted: %s", alias)
		}
		if !disabledStillImageConfig(alias, ph3ImageComfyConfig(root)) {
			t.Fatalf("config smuggled as: %s", alias)
		}
	}
}
