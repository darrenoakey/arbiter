package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func minimaxH3ImageSglangConfig(root string) ModelConfig {
	return ModelConfig{
		MemoryGB: 64, MaxConcurrent: 1, MaxInstances: intPtr(1),
		ModelPath: "/home/darren/nvme-models/minimax-h3-image-sglang/MiniMax-H3-Image", Placements: []string{"spark"},
		AdapterParams: map[string]string{"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
		WorkerCmd:     []string{filepath.Join(root, "venvs", "h3image-sglang", "bin", "python"), "-m", "arbiter.worker_main", minimaxH3ImageSglangMXFP8Model},
	}
}

func TestMiniMaxH3ImageSglangExactLocalComparisonIsAllowed(t *testing.T) {
	root := t.TempDir()
	cfg := minimaxH3ImageSglangConfig(root)
	if isDisabledStillImageModel(minimaxH3ImageSglangMXFP8Model) || disabledStillImageConfig(minimaxH3ImageSglangMXFP8Model, cfg) {
		t.Fatal("local MiniMax H3 SGLang still-image comparison was refused")
	}
	if err := validateModelWorkerPolicy(root, minimaxH3ImageSglangMXFP8Model, cfg, true); err != nil {
		t.Fatalf("registration rejected: %v", err)
	}
	if err := validateJobModelCompatibility(minimaxH3ImageSglangMXFP8JobType, minimaxH3ImageSglangMXFP8Model); err != nil {
		t.Fatalf("exact route rejected: %v", err)
	}
	if got := JobTypeToModel[minimaxH3ImageSglangMXFP8JobType]; got != minimaxH3ImageSglangMXFP8Model {
		t.Fatalf("route = %q", got)
	}
	if trustedPythonAdapters[minimaxH3ImageSglangMXFP8Model] != "h3image-sglang" {
		t.Fatal("wrong venv binding")
	}
}

func TestMiniMaxH3ImageRuntimesCannotBeCrossed(t *testing.T) {
	root := t.TempDir()
	models := []string{minimaxH3ImageModel, minimaxH3ImageOrbitQuantModel, minimaxH3ImageSglangMXFP8Model}
	for _, model := range models {
		for _, jobType := range []string{minimaxH3ImageJobType, minimaxH3ImageOrbitQuantJobType, minimaxH3ImageSglangMXFP8JobType, "image-generate", "video-generate-h3"} {
			err := validateJobModelCompatibility(jobType, model)
			if want := minimaxH3ImageJobTypes[model]; jobType == want {
				if err != nil {
					t.Fatalf("%s rejected its own job type: %v", model, err)
				}
			} else if err == nil {
				t.Fatalf("%s job type accepted model %s", jobType, model)
			}
		}
	}
	cfg := minimaxH3ImageSglangConfig(root)
	cfg.WorkerCmd[0] = filepath.Join(root, "venvs", "qwenimage-sglang", "bin", "python")
	if err := validatePythonWorkerCommand(root, minimaxH3ImageSglangMXFP8Model, cfg.WorkerCmd); err == nil || !strings.Contains(err.Error(), untrustedWorkerCommandMessage) {
		t.Fatalf("foreign sglang venv accepted: %v", err)
	}
}
