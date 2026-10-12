package main

import (
	"path/filepath"
	"testing"
)

func TestIrisAndMingComparisonAdaptersArePinned(t *testing.T) {
	root := t.TempDir()
	if isDisabledStillImageModel(iris3BModel) {
		t.Fatal("iris-3b must not be classified as a disabled still-image model")
	}
	if isDisabledStillImageModel(mingImageDesignModel) {
		t.Fatal("ming-image-0.1-design must not be classified as a disabled still-image model")
	}
	if !isDisabledStillImageModel("iris-3b-lora") || !isDisabledStillImageModel("ming-image-0.1-design-lora") {
		t.Fatal("lora aliases of the comparison adapters must stay disabled")
	}

	iris := ModelConfig{
		MemoryGB:  48,
		ModelPath: "/home/darren/nvme-models/iris-3b",
		WorkerCmd: []string{
			filepath.Join(root, "venvs", "iris3b", "bin", "python"),
			"-m", "arbiter.worker_main", iris3BModel,
		},
	}
	if err := validatePythonWorkerCommand(root, iris3BModel, iris.WorkerCmd); err != nil {
		t.Fatalf("iris worker: %v", err)
	}
	if err := rejectDisabledStillImage(iris3BJobType, iris3BModel); err != nil {
		t.Fatalf("iris job type: %v", err)
	}
	if err := rejectDisabledStillImage("qwen-image", iris3BModel); err == nil {
		t.Fatal("iris accepted the qwen job type")
	}
	if got := JobTypeToModel[iris3BJobType]; got != iris3BModel {
		t.Fatalf("JobTypeToModel[iris-3b] = %q", got)
	}

	ming := ModelConfig{
		MemoryGB:  80,
		ModelPath: "/home/darren/nvme-models/Ming-Image-0.1-Design",
		WorkerCmd: []string{
			filepath.Join(root, "venvs", "mingimage", "bin", "python"),
			"-m", "arbiter.worker_main", mingImageDesignModel,
		},
	}
	if err := validatePythonWorkerCommand(root, mingImageDesignModel, ming.WorkerCmd); err != nil {
		t.Fatalf("ming worker: %v", err)
	}
	if err := rejectDisabledStillImage(mingImageDesignJobType, mingImageDesignModel); err != nil {
		t.Fatalf("ming job type: %v", err)
	}
	if got := JobTypeToModel[mingImageDesignJobType]; got != mingImageDesignModel {
		t.Fatalf("JobTypeToModel[ming-image-design] = %q", got)
	}
}
