package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
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

// writeTestConfig writes a config.json under projectRoot/local and returns the
// loaded Config. Used by the multi-machine placement/host tests.
func writeTestConfig(t *testing.T, projectRoot, body string) *Config {
	t.Helper()
	localDir := filepath.Join(projectRoot, "local")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatalf("mkdir local: %v", err)
	}
	if err := os.WriteFile(filepath.Join(localDir, "config.json"), []byte(body), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	cfg, err := LoadConfig(projectRoot)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	return cfg
}

// A model with no placements MUST default to the local host ("spark") and
// remote-enabled — i.e. behave exactly as before the multi-machine seam.
func TestPlacementsDefaultToLocalSpark(t *testing.T) {
	cfg := writeTestConfig(t, t.TempDir(), `{
		"models": {
			"flux2": {"memory_gb": 31, "max_concurrent": 1}
		}
	}`)
	m := cfg.Models["flux2"]
	if got := m.PlacementsOrDefault(); len(got) != 1 || got[0] != LocalHost {
		t.Fatalf("default placements = %v, want [%q]", got, LocalHost)
	}
	if !m.RemoteEnabledOrDefault() {
		t.Fatalf("RemoteEnabledOrDefault() = false, want true (nil default)")
	}
	if len(cfg.Hosts) != 0 {
		t.Fatalf("expected no hosts, got %v", cfg.Hosts)
	}
}

// A model with explicit hosts + placements + remote_enabled parses correctly.
func TestHostsAndPlacementsParse(t *testing.T) {
	cfg := writeTestConfig(t, t.TempDir(), `{
		"hosts": {
			"boringstack": {"addr": "10.0.0.42:11434", "kind": "mlx", "budget_gb": 96}
		},
		"models": {
			"gemma4-26b": {
				"memory_gb": 20,
				"placements": ["boringstack", "spark"],
				"remote_enabled": false
			}
		}
	}`)

	h, ok := cfg.Hosts["boringstack"]
	if !ok {
		t.Fatalf("host boringstack missing")
	}
	if h.Addr != "10.0.0.42:11434" || h.Kind != "mlx" || h.BudgetGB != 96 {
		t.Fatalf("host parsed wrong: %+v", h)
	}
	if cfg.HostIsLocal("boringstack") {
		t.Fatalf("boringstack should be remote")
	}
	if !cfg.HostIsLocal("spark") || !cfg.HostIsLocal("") {
		t.Fatalf("spark / empty should be local")
	}

	m := cfg.Models["gemma4-26b"]
	if got := m.PlacementsOrDefault(); len(got) != 2 || got[0] != "boringstack" || got[1] != "spark" {
		t.Fatalf("placements = %v, want [boringstack spark]", got)
	}
	if m.RemoteEnabledOrDefault() {
		t.Fatalf("remote_enabled=false should report disabled")
	}
}

// A remote instance must NEVER contribute to the audited local VRAM ledger.
func TestRemoteInstanceExcludedFromUsedGB(t *testing.T) {
	cfg := &Config{
		VRAMBudgetGB: 100,
		Models:       map[string]ModelConfig{},
		Hosts: map[string]HostConfig{
			"boringstack": {Addr: "10.0.0.42:11434", Kind: "mlx", BudgetGB: 96},
		},
	}
	mgr := NewInstanceManager(cfg, "python3", ".")

	// A local instance reserves audited VRAM.
	local := NewInstance("flux2", "flux2", 1, 30, "python3", ".")
	mgr.Register(local)
	if !mgr.ReserveMemoryFor(local, 30) {
		t.Fatalf("local reserve failed")
	}

	// A remote instance reserves nothing locally.
	remote := NewInstance("gemma4-26b", "gemma4-26b", 1, 20, "python3", ".")
	remote.host = "boringstack"
	if !remote.isRemote() {
		t.Fatalf("expected remote instance")
	}
	mgr.Register(remote)
	if !mgr.ReserveMemoryFor(remote, 20) {
		t.Fatalf("remote reserve should report success")
	}

	if mgr.usedGB != 30 {
		t.Fatalf("usedGB = %v, want 30 (remote excluded)", mgr.usedGB)
	}
	if remote.vramHeld {
		t.Fatalf("remote instance must not hold audited VRAM")
	}

	// Audit and reconcile must agree with the local-only ledger.
	mgr.AuditVRAMConsistency("test")
	if freed := mgr.ReconcileFromInstances(); freed != 0 {
		t.Fatalf("ReconcileFromInstances freed %v, expected 0 (no drift)", freed)
	}

	// Releasing the remote instance is a no-op locally; the local one frees 30.
	if freed := mgr.ReleaseMemoryFor(remote); freed != 0 {
		t.Fatalf("remote release freed %v, want 0", freed)
	}
	if freed := mgr.ReleaseMemoryFor(local); freed != 30 {
		t.Fatalf("local release freed %v, want 30", freed)
	}
	if mgr.usedGB != 0 {
		t.Fatalf("usedGB = %v, want 0 after releases", mgr.usedGB)
	}

	// The advisory remote-host budget exists and is separate from usedGB.
	rb := mgr.RemoteHostBudget("boringstack")
	if rb == nil {
		t.Fatalf("expected advisory budget for boringstack")
	}
	if rb.FreeGB() != 96 {
		t.Fatalf("remote FreeGB = %v, want 96", rb.FreeGB())
	}
}

func TestSaveModelConfigConcurrentWritesRemainValid(t *testing.T) {
	projectRoot := t.TempDir()
	const n = 24
	var wg sync.WaitGroup
	errs := make(chan error, n)

	for i := 0; i < n; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			cfg := ModelConfig{
				MemoryGB:       float64(10 + i),
				MaxConcurrent:  1,
				MaxInstances:   intPtr(1),
				KeepAliveSec:   900,
				AvgInferenceMs: 1500,
				LoadMs:         2500,
			}
			errs <- SaveModelConfig(projectRoot, fmt.Sprintf("model-%02d", i), cfg)
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("SaveModelConfig() error = %v", err)
		}
	}

	loaded, err := LoadConfig(projectRoot)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("model-%02d", i)
		if _, ok := loaded.Models[id]; !ok {
			t.Fatalf("saved model %q missing after concurrent writes", id)
		}
	}
}
