package main

import (
	"path/filepath"
	"strings"
	"testing"
)

// MOSS-Image-SFT (OpenMOSS, ~5B DiT + Qwen3.5 text encoder, Apache 2.0) is an
// owner-requested comparison adapter (2026-10-17): the exact id moss-image-sft
// reachable only through the moss-image job type from venvs/mossimage. These
// tests pin that the opening is exactly that narrow and every other moss
// path (MOSS-Image-Base, quantized community builds, LoRA variants, renamed
// copies) stays denied.

func mossImageConfig(root string) ModelConfig {
	return ModelConfig{
		MemoryGB:      40,
		MaxConcurrent: 1,
		MaxInstances:  intPtr(1),
		ModelPath:     "/home/darren/nvme-models/moss-image/MOSS-Image-SFT",
		Placements:    []string{"spark"},
		AdapterParams: map[string]string{
			"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
		},
		WorkerCmd: []string{
			filepath.Join(root, "venvs", "mossimage", "bin", "python"),
			"-m", "arbiter.worker_main", mossImageSftModel,
		},
	}
}

func TestMossImageSftExactIDIsAllowed(t *testing.T) {
	root := t.TempDir()
	if isDisabledStillImageModel(mossImageSftModel) {
		t.Fatalf("%s classified as a disabled still-image model", mossImageSftModel)
	}
	cfg := mossImageConfig(root)
	if disabledStillImageConfig(mossImageSftModel, cfg) {
		t.Fatalf("%s config naming its MOSS weights and venv was refused", mossImageSftModel)
	}
	if err := validateModelWorkerPolicy(root, mossImageSftModel, cfg, true); err != nil {
		t.Fatalf("%s registration rejected: %v", mossImageSftModel, err)
	}
	if err := validateJobModelCompatibility(mossImageJobType, mossImageSftModel); err != nil {
		t.Fatalf("moss-image routing rejected: %v", err)
	}
	if got := JobTypeToModel[mossImageJobType]; got != mossImageSftModel {
		t.Fatalf("JobTypeToModel[%s] = %q", mossImageJobType, got)
	}
	if venv := trustedPythonAdapters[mossImageSftModel]; venv != "mossimage" {
		t.Fatalf("trustedPythonAdapters[%s] = %q, want mossimage", mossImageSftModel, venv)
	}
}

func TestMossImageSftCannotCrossOrBeSmuggled(t *testing.T) {
	root := t.TempDir()
	// The adapter cannot serve any other job type, and the job type cannot
	// be redirected to another adapter.
	for _, jobType := range []string{"image-generate", "image-edit", "qwen-image", "reference-image-edit", "flux3-klein-bf16", "moss-image-sft", "moss-image-base"} {
		if err := rejectDisabledStillImage(jobType, mossImageSftModel); err == nil {
			t.Fatalf("job type %q accepted %s", jobType, mossImageSftModel)
		}
		if err := validateJobModelCompatibility(jobType, mossImageSftModel); err == nil {
			t.Fatalf("job type %q routed to %s", jobType, mossImageSftModel)
		}
	}
	for _, other := range []string{"flux2", "qwen-image-2.1", "reference-image-edit", "birefnet", mossImageSftModel + "-lora"} {
		if err := validateJobModelCompatibility(mossImageJobType, other); err == nil {
			t.Fatalf("moss-image job routed to %q", other)
		}
	}
	// A worker command selecting a different adapter or venv is untrusted.
	for _, mutate := range []func([]string){
		func(c []string) { c[3] = "qwen-image-2.1" },
		func(c []string) { c[0] = filepath.Join(root, "venvs", "qwenimage", "bin", "python") },
		func(c []string) { c[0] = filepath.Join(root, "venvs", "moss-image", "bin", "python") },
	} {
		cfg := mossImageConfig(root)
		mutate(cfg.WorkerCmd)
		if err := validatePythonWorkerCommand(root, mossImageSftModel, cfg.WorkerCmd); err == nil ||
			!strings.Contains(err.Error(), untrustedWorkerCommandMessage) {
			t.Fatalf("%s accepted mutated worker %v: %v", mossImageSftModel, cfg.WorkerCmd, err)
		}
	}
	// Aliases and near neighbours stay denied. (The exact job type
	// `moss-image` is of course registered — it maps to moss-image-sft.)
	for _, alias := range []string{
		"moss-image-base", "moss-image-sft-lora",
		"moss-image-sft-int8", "MOSSImageSFT", "mossimage", "OpenMOSS-Team/MOSS-Image-SFT",
		"OpenMOSS-Team/MOSS-Image-Base", "moss-image-edit",
	} {
		if !isDisabledStillImageModel(alias) {
			t.Fatalf("%q was not denied", alias)
		}
		if !disabledStillImageConfig(alias, mossImageConfig(root)) {
			t.Fatalf("moss config survived under alias %q", alias)
		}
		if _, ok := JobTypeToModel[alias]; ok {
			t.Fatalf("alias %q is a job type", alias)
		}
	}
}
