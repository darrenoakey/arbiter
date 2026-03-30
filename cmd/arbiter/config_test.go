package main

import (
	"path/filepath"
	"testing"
)

func intPtr(n int) *int {
	return &n
}

func TestSaveAndDeleteModelConfig(t *testing.T) {
	projectRoot := t.TempDir()
	cfg := ModelConfig{
		MemoryGB:       12,
		MaxConcurrent:  2,
		MaxInstances:   intPtr(3),
		KeepAliveSec:   900,
		AvgInferenceMs: 1500,
		LoadMs:         2500,
		WorkerCmd:      []string{"worker-bin", "--serve"},
		AdapterParams: map[string]string{
			"FOO": "bar",
		},
	}

	if err := SaveModelConfig(projectRoot, "demo-model", cfg); err != nil {
		t.Fatalf("SaveModelConfig() error = %v", err)
	}

	loaded, err := LoadConfig(projectRoot)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}
	saved, ok := loaded.Models["demo-model"]
	if !ok {
		t.Fatalf("saved model missing from config.json at %s", filepath.Join(projectRoot, "local", "config.json"))
	}
	if saved.MemoryGB != cfg.MemoryGB || saved.MaxConcurrent != cfg.MaxConcurrent {
		t.Fatalf("saved config mismatch: got %+v want %+v", saved, cfg)
	}
	if saved.MaxInstances == nil || *saved.MaxInstances != *cfg.MaxInstances {
		t.Fatalf("saved max_instances mismatch: got %+v want %+v", saved.MaxInstances, cfg.MaxInstances)
	}

	if err := DeleteModelConfig(projectRoot, "demo-model"); err != nil {
		t.Fatalf("DeleteModelConfig() error = %v", err)
	}

	loaded, err = LoadConfig(projectRoot)
	if err != nil {
		t.Fatalf("LoadConfig() after delete error = %v", err)
	}
	if _, ok := loaded.Models["demo-model"]; ok {
		t.Fatalf("model still present after delete")
	}
}
