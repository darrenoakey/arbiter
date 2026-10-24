package main

import "testing"

// TestIDLoraRouting pins the new video route to its isolated audited worker.
func TestIDLoraRouting(t *testing.T) {
	if isDisabledStillImageModel("id-lora") || !isDisabledStillImageModel("id-lora-custom") {
		t.Fatal("ID-LoRA exception must be exact")
	}
	if disabledStillImageConfig("id-lora", ModelConfig{WorkerCmd: []string{"/home/darren/src/arbiter/venvs/id-lora/bin/python", "-m", "arbiter.worker_main", "id-lora"}}) {
		t.Fatal("audited video worker must pass model policy")
	}
	if got := JobTypeToModel["video-id-lora"]; got != "id-lora" {
		t.Fatalf("video-id-lora maps to %q", got)
	}
	if got, ok := trustedPythonAdapters["id-lora"]; !ok || got != "id-lora" {
		t.Fatalf("id-lora worker venv = %q, exists = %v", got, ok)
	}
	if _, ok := trustedPythonAdapters["id-lora-custom"]; ok {
		t.Fatal("unregistered ID-LoRA aliases must not be trusted")
	}
}
