package main

import (
	"path/filepath"
	"strings"
	"testing"
)

// The Qwen unified adapter is sanctioned exception #2 (owner decision,
// 2026-09-21): a local text-to-image + reference-editing model exposed only
// through the `qwen-image` job type. These tests pin that the carve-out is
// exactly that narrow — its own job type and model id only — and that every
// other still-image path stays disabled.

func qwenImage21Config(root string) ModelConfig {
	return ModelConfig{
		MemoryGB:      44,
		MaxConcurrent: 1,
		MaxInstances:  intPtr(1),
		AutoDownload:  "Qwen/Qwen-Image-2.1",
		WorkerCmd: []string{
			filepath.Join(root, "venvs", "qwenimage", "bin", "python"),
			"-m", "arbiter.worker_main", qwenImage21Model,
		},
	}
}

func TestQwenImage21IsAllowedByPolicy(t *testing.T) {
	if isDisabledStillImageModel(qwenImage21Model) {
		t.Fatal("qwen-image-2.1 must not be classified as a disabled still-image model")
	}
	root := t.TempDir()
	// The venv python path does not exist in tests; bypass the existence
	// probing the same way the reference editor test does by validating the
	// policy layers that matter here.
	cfg := qwenImage21Config(root)
	if disabledStillImageConfig(qwenImage21Model, cfg) {
		t.Fatal("qwen-image-2.1 config naming its Qwen checkpoint and venv was refused")
	}
	if err := validatePythonWorkerCommand(root, qwenImage21Model, cfg.WorkerCmd); err != nil {
		t.Fatalf("qwen-image-2.1 worker command rejected: %v", err)
	}
	if err := rejectDisabledStillImage(qwenImageJobType, qwenImage21Model); err != nil {
		t.Fatalf("qwen-image job type rejected: %v", err)
	}
	if err := validateJobModelCompatibility(qwenImageJobType, qwenImage21Model); err != nil {
		t.Fatalf("qwen-image routing rejected: %v", err)
	}
	if got := JobTypeToModel[qwenImageJobType]; got != qwenImage21Model {
		t.Fatalf("JobTypeToModel[qwen-image] = %q, want %q", got, qwenImage21Model)
	}
	if venv, ok := trustedPythonAdapters[qwenImage21Model]; !ok || venv != "qwenimage" {
		t.Fatalf("trustedPythonAdapters[qwen-image-2.1] = %q ok=%v, want qwenimage", venv, ok)
	}
}

func TestQwenImage21ExceptionIsNarrow(t *testing.T) {
	root := t.TempDir()
	// Near-neighbour aliases under any id stay disabled.
	for _, modelID := range []string{"qwen-image-2.1-lora", "qwen-image", "Qwen/Qwen-Image-2.1", "qwen-image-edit"} {
		if !isDisabledStillImageModel(modelID) {
			t.Fatalf("%q was not classified as a disabled still-image model", modelID)
		}
	}
	// The sanctioned model cannot be smuggled in under other job types —
	// including the legacy image markers.
	for _, jobType := range []string{"image-edit", "image-generate", "background-remove", "reference-image-edit"} {
		if err := rejectDisabledStillImage(jobType, qwenImage21Model); err == nil {
			t.Fatalf("job type %q accepted the qwen adapter as an override", jobType)
		}
	}
	// And the qwen job type cannot be redirected to another adapter.
	for _, modelID := range []string{"flux2", "reference-image-edit", "birefnet"} {
		if err := validateJobModelCompatibility(qwenImageJobType, modelID); err == nil {
			t.Fatalf("qwen-image routed to %q", modelID)
		}
	}
	// A worker command that selects a different adapter is still untrusted.
	mismatched := qwenImage21Config(root)
	mismatched.WorkerCmd[3] = "flux2"
	if err := validatePythonWorkerCommand(root, qwenImage21Model, mismatched.WorkerCmd); err == nil ||
		!strings.Contains(err.Error(), untrustedWorkerCommandMessage) {
		t.Fatalf("qwen-image-2.1 accepted a worker command selecting the flux2 adapter: %v", err)
	}
	// The config cannot be smuggled in under a non-whitelisted model id: the
	// auto_download marker trips the still-image config check.
	disguised := qwenImage21Config(root)
	if !disabledStillImageConfig("qwen-image-2.1-lora", disguised) {
		t.Fatal("qwen config under a lora alias survived the still-image config check")
	}
}

func TestQwenImage21ModelRegistrationRoundTrip(t *testing.T) {
	// POST /v1/models and the persisted config both go through
	// ApplyModelConfig; the sanctioned shape must survive worker policy.
	root := t.TempDir()
	cfg := qwenImage21Config(root)
	if err := validateModelWorkerPolicy(root, qwenImage21Model, cfg, true); err != nil {
		t.Fatalf("qwen-image-2.1 model registration rejected: %v", err)
	}
}

// --- Sanctioned exception #3: qwen-image-2.1-heretic -------------------------

func qwenImage21HereticConfig(root string) ModelConfig {
	return ModelConfig{
		MemoryGB:      52,
		MaxConcurrent: 1,
		MaxInstances:  intPtr(1),
		AutoDownload:  "",
		ModelPath:     "/mnt/t9/models/qwen-image-2.1-heretic",
		WorkerCmd: []string{
			filepath.Join(root, "venvs", "qwenimage", "bin", "python"),
			"-m", "arbiter.worker_main", qwenImage21HereticModel,
		},
	}
}

func TestQwenImage21HereticIsAllowedByPolicy(t *testing.T) {
	if isDisabledStillImageModel(qwenImage21HereticModel) {
		t.Fatal("qwen-image-2.1-heretic must not be classified as a disabled still-image model")
	}
	root := t.TempDir()
	cfg := qwenImage21HereticConfig(root)
	if disabledStillImageConfig(qwenImage21HereticModel, cfg) {
		t.Fatal("qwen-image-2.1-heretic config naming the merged local dir and venv was refused")
	}
	if err := validatePythonWorkerCommand(root, qwenImage21HereticModel, cfg.WorkerCmd); err != nil {
		t.Fatalf("qwen-image-2.1-heretic worker command rejected: %v", err)
	}
	if err := rejectDisabledStillImage(qwenImageHereticJobType, qwenImage21HereticModel); err != nil {
		t.Fatalf("qwen-image-heretic job type rejected: %v", err)
	}
	if err := validateJobModelCompatibility(qwenImageHereticJobType, qwenImage21HereticModel); err != nil {
		t.Fatalf("qwen-image-heretic routing rejected: %v", err)
	}
	if got := JobTypeToModel[qwenImageHereticJobType]; got != qwenImage21HereticModel {
		t.Fatalf("JobTypeToModel[qwen-image-heretic] = %q, want %q", got, qwenImage21HereticModel)
	}
	if venv, ok := trustedPythonAdapters[qwenImage21HereticModel]; !ok || venv != "qwenimage" {
		t.Fatalf("trustedPythonAdapters[qwen-image-2.1-heretic] = %q ok=%v, want qwenimage", venv, ok)
	}
}

func TestQwenImage21HereticExceptionIsNarrow(t *testing.T) {
	root := t.TempDir()
	// Near-neighbour aliases stay disabled.
	for _, modelID := range []string{"qwen-image-2.1-heretic-lora", "qwen-image-heretic", "qwen-image-2.1-uncensored"} {
		if !isDisabledStillImageModel(modelID) {
			t.Fatalf("%q was not classified as a disabled still-image model", modelID)
		}
	}
	// The heretic adapter cannot be smuggled in under other job types,
	// including the stock qwen-image job type and legacy image markers.
	for _, jobType := range []string{"qwen-image", "qwen-image-turbo", "image-edit", "image-generate", "reference-image-edit"} {
		if err := rejectDisabledStillImage(jobType, qwenImage21HereticModel); err == nil {
			t.Fatalf("job type %q accepted the heretic adapter as an override", jobType)
		}
	}
	// The stock qwen adapter cannot serve the heretic job type, and the heretic
	// job type cannot be redirected to another adapter.
	if err := rejectDisabledStillImage(qwenImageHereticJobType, qwenImage21Model); err == nil {
		t.Fatal("qwen-image-heretic accepted the stock qwen adapter as an override")
	}
	for _, modelID := range []string{"flux2", "reference-image-edit", "birefnet", "qwen-image-2.1"} {
		if err := validateJobModelCompatibility(qwenImageHereticJobType, modelID); err == nil {
			t.Fatalf("qwen-image-heretic routed to %q", modelID)
		}
	}
	// A worker command that selects a different adapter is still untrusted.
	mismatched := qwenImage21HereticConfig(root)
	mismatched.WorkerCmd[3] = "flux2"
	if err := validatePythonWorkerCommand(root, qwenImage21HereticModel, mismatched.WorkerCmd); err == nil ||
		!strings.Contains(err.Error(), untrustedWorkerCommandMessage) {
		t.Fatalf("qwen-image-2.1-heretic accepted a worker command selecting the flux2 adapter: %v", err)
	}
}

func qwenImage21SglangConfig(root string) ModelConfig {
	return ModelConfig{
		MemoryGB:      52,
		MaxConcurrent: 1,
		MaxInstances:  intPtr(1),
		AutoDownload:  "Qwen/Qwen-Image-2.1",
		WorkerCmd: []string{
			filepath.Join(root, "venvs", "qwenimage-sglang", "bin", "python"),
			"-m", "arbiter.worker_main", qwenImage21SglangModel,
		},
	}
}

func TestQwenImage21SglangIsAllowedByPolicy(t *testing.T) {
	if isDisabledStillImageModel(qwenImage21SglangModel) {
		t.Fatal("qwen-image-2.1-sglang must not be classified as a disabled still-image model")
	}
	root := t.TempDir()
	cfg := qwenImage21SglangConfig(root)
	if disabledStillImageConfig(qwenImage21SglangModel, cfg) {
		t.Fatal("qwen-image-2.1-sglang config naming its Qwen checkpoint and venv was refused")
	}
	if err := validatePythonWorkerCommand(root, qwenImage21SglangModel, cfg.WorkerCmd); err != nil {
		t.Fatalf("qwen-image-2.1-sglang worker command rejected: %v", err)
	}
	if err := rejectDisabledStillImage(qwenImageSglangJobType, qwenImage21SglangModel); err != nil {
		t.Fatalf("qwen-image-sglang job type rejected: %v", err)
	}
	if err := validateJobModelCompatibility(qwenImageSglangJobType, qwenImage21SglangModel); err != nil {
		t.Fatalf("qwen-image-sglang routing rejected: %v", err)
	}
	if got := JobTypeToModel[qwenImageSglangJobType]; got != qwenImage21SglangModel {
		t.Fatalf("JobTypeToModel[qwen-image-sglang] = %q, want %q", got, qwenImage21SglangModel)
	}
	if venv, ok := trustedPythonAdapters[qwenImage21SglangModel]; !ok || venv != "qwenimage-sglang" {
		t.Fatalf("trustedPythonAdapters[qwen-image-2.1-sglang] = %q ok=%v, want qwenimage-sglang", venv, ok)
	}
}

func TestQwenImage21SglangExceptionIsNarrow(t *testing.T) {
	root := t.TempDir()
	for _, modelID := range []string{"qwen-image-2.1-sglang-lora", "qwen-image-sglang"} {
		if !isDisabledStillImageModel(modelID) {
			t.Fatalf("%q was not classified as a disabled still-image model", modelID)
		}
	}
	for _, jobType := range []string{"qwen-image", "qwen-image-heretic", "qwen-image-turbo", "image-generate", "reference-image-edit"} {
		if err := rejectDisabledStillImage(jobType, qwenImage21SglangModel); err == nil {
			t.Fatalf("job type %q accepted the sglang adapter as an override", jobType)
		}
	}
	if err := rejectDisabledStillImage(qwenImageSglangJobType, qwenImage21Model); err == nil {
		t.Fatal("qwen-image-sglang accepted the diffusers adapter as an override")
	}
	mismatched := qwenImage21SglangConfig(root)
	mismatched.WorkerCmd[3] = "flux2"
	if err := validatePythonWorkerCommand(root, qwenImage21SglangModel, mismatched.WorkerCmd); err == nil ||
		!strings.Contains(err.Error(), untrustedWorkerCommandMessage) {
		t.Fatalf("qwen-image-2.1-sglang accepted a worker command selecting flux2: %v", err)
	}
}

func qwenImage21TurboConfig(root string) ModelConfig {
	return ModelConfig{
		MemoryGB:      52,
		MaxConcurrent: 1,
		MaxInstances:  intPtr(1),
		ModelPath:     "/home/darren/nvme-models/Qwen-Image-2.1-Turbo",
		WorkerCmd: []string{
			filepath.Join(root, "venvs", "qwenimage-turbo", "bin", "python"),
			"-m", "arbiter.worker_main", qwenImage21TurboModel,
		},
	}
}

func TestQwenImage21TurboIsAllowedByPolicy(t *testing.T) {
	if isDisabledStillImageModel(qwenImage21TurboModel) {
		t.Fatal("qwen-image-2.1-turbo must not be classified as a disabled still-image model")
	}
	root := t.TempDir()
	cfg := qwenImage21TurboConfig(root)
	if disabledStillImageConfig(qwenImage21TurboModel, cfg) {
		t.Fatal("qwen-image-2.1-turbo config naming the NVMe checkpoint and venv was refused")
	}
	if err := validatePythonWorkerCommand(root, qwenImage21TurboModel, cfg.WorkerCmd); err != nil {
		t.Fatalf("qwen-image-2.1-turbo worker command rejected: %v", err)
	}
	if err := rejectDisabledStillImage(qwenImageTurboJobType, qwenImage21TurboModel); err != nil {
		t.Fatalf("qwen-image-turbo job type rejected: %v", err)
	}
	if err := validateJobModelCompatibility(qwenImageTurboJobType, qwenImage21TurboModel); err != nil {
		t.Fatalf("qwen-image-turbo routing rejected: %v", err)
	}
	if got := JobTypeToModel[qwenImageTurboJobType]; got != qwenImage21TurboModel {
		t.Fatalf("JobTypeToModel[qwen-image-turbo] = %q, want %q", got, qwenImage21TurboModel)
	}
	if venv, ok := trustedPythonAdapters[qwenImage21TurboModel]; !ok || venv != "qwenimage-turbo" {
		t.Fatalf("trustedPythonAdapters[qwen-image-2.1-turbo] = %q ok=%v, want qwenimage-turbo", venv, ok)
	}
}

func TestQwenImage21TurboExceptionIsNarrow(t *testing.T) {
	root := t.TempDir()
	for _, modelID := range []string{"qwen-image-2.1-turbo-lora", "qwen-image-turbo", "Qwen/Qwen-Image-2.1-Turbo"} {
		if !isDisabledStillImageModel(modelID) {
			t.Fatalf("%q was not classified as a disabled still-image model", modelID)
		}
	}
	for _, jobType := range []string{"qwen-image", "qwen-image-heretic", "qwen-image-sglang", "image-generate", "reference-image-edit"} {
		if err := rejectDisabledStillImage(jobType, qwenImage21TurboModel); err == nil {
			t.Fatalf("job type %q accepted the turbo adapter as an override", jobType)
		}
	}
	for _, modelID := range []string{qwenImage21Model, qwenImage21HereticModel, qwenImage21SglangModel} {
		if err := rejectDisabledStillImage(qwenImageTurboJobType, modelID); err == nil {
			t.Fatalf("qwen-image-turbo accepted %q as an override", modelID)
		}
	}
	mismatched := qwenImage21TurboConfig(root)
	mismatched.WorkerCmd[3] = "flux2"
	if err := validatePythonWorkerCommand(root, qwenImage21TurboModel, mismatched.WorkerCmd); err == nil ||
		!strings.Contains(err.Error(), untrustedWorkerCommandMessage) {
		t.Fatalf("qwen-image-2.1-turbo accepted a worker command selecting flux2: %v", err)
	}
}

func TestQwenImage21TurboModelRegistrationRoundTrip(t *testing.T) {
	root := t.TempDir()
	cfg := qwenImage21TurboConfig(root)
	if err := validateModelWorkerPolicy(root, qwenImage21TurboModel, cfg, true); err != nil {
		t.Fatalf("qwen-image-2.1-turbo model registration rejected: %v", err)
	}
}

func TestQwenImage21HereticModelRegistrationRoundTrip(t *testing.T) {
	root := t.TempDir()
	cfg := qwenImage21HereticConfig(root)
	if err := validateModelWorkerPolicy(root, qwenImage21HereticModel, cfg, true); err != nil {
		t.Fatalf("qwen-image-2.1-heretic model registration rejected: %v", err)
	}
}
