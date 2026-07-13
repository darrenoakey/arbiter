package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// readEventLines returns each JSON line in the event log directory whose
// "event" field matches eventName.
func readEventLines(t *testing.T, logDir, eventName string) []map[string]any {
	t.Helper()
	entries, err := os.ReadDir(logDir)
	if err != nil {
		return nil
	}
	var out []map[string]any
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(logDir, e.Name()))
		if err != nil {
			continue
		}
		for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
			if line == "" {
				continue
			}
			var m map[string]any
			if err := json.Unmarshal([]byte(line), &m); err != nil {
				continue
			}
			if m["event"] == eventName {
				out = append(out, m)
			}
		}
	}
	return out
}

func TestPatchModelMemoryGB_PreservesOtherFields(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "local"), 0o755); err != nil {
		t.Fatal(err)
	}
	original := map[string]any{
		"vram_budget_gb": 100.0,
		"models": map[string]any{
			"flux-schnell": map[string]any{
				"memory_gb":      24.0,
				"max_concurrent": 1.0,
				"adapter_params": map[string]any{"hand_edited": "yes"},
			},
		},
	}
	raw, _ := json.MarshalIndent(original, "", "  ")
	if err := os.WriteFile(filepath.Join(root, "local", "config.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := PatchModelMemoryGB(root, "flux-schnell", 32.0); err != nil {
		t.Fatalf("patch: %v", err)
	}
	got, _ := os.ReadFile(filepath.Join(root, "local", "config.json"))
	var parsed map[string]any
	if err := json.Unmarshal(got, &parsed); err != nil {
		t.Fatal(err)
	}
	model := parsed["models"].(map[string]any)["flux-schnell"].(map[string]any)
	if model["memory_gb"] != 32.0 {
		t.Fatalf("memory_gb not patched: %v", model["memory_gb"])
	}
	// Other fields preserved
	if model["max_concurrent"] != 1.0 {
		t.Fatalf("max_concurrent lost: %v", model["max_concurrent"])
	}
	ap, ok := model["adapter_params"].(map[string]any)
	if !ok || ap["hand_edited"] != "yes" {
		t.Fatalf("adapter_params lost: %v", model["adapter_params"])
	}
}

func TestMemoryWatchdog_DriftEmitsEventAndPatches(t *testing.T) {
	root := t.TempDir()
	logDir := filepath.Join(root, "logs")
	if err := os.MkdirAll(filepath.Join(root, "local"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		t.Fatal(err)
	}
	configured := map[string]any{
		"vram_budget_gb": 100.0,
		"models": map[string]any{
			"test-model": map[string]any{"memory_gb": 4.0},
		},
	}
	raw, _ := json.MarshalIndent(configured, "", "  ")
	if err := os.WriteFile(filepath.Join(root, "local", "config.json"), raw, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg := &Config{
		VRAMBudgetGB: 100,
		Models: map[string]ModelConfig{
			"test-model": {MemoryGB: 4},
		},
	}
	mgr := NewInstanceManager(cfg, "python3", root)
	logger := NewEventLogger(logDir)
	w := NewMemoryWatchdog(cfg, mgr, logger, root)

	// Synthesise an instance memory snapshot showing severe drift.
	snap := instanceMemSnapshot{
		InstanceID:   "test-model",
		ModelID:      "test-model",
		PID:          1,
		TreeVRAMGB:   12, // 3x configured
		TreeRSSGB:    8,
		ConfiguredGB: 4,
	}
	// Drift requires `driftConsecutive` ticks. Simulate by calling checkDrift twice.
	w.checkDrift(snap)
	w.checkDrift(snap)

	events := readEventLines(t, logDir, "model.memory_drift")
	if len(events) == 0 {
		t.Fatalf("expected model.memory_drift event after 2 sustained ticks")
	}
	last := events[len(events)-1]
	if last["model_id"] != "test-model" {
		t.Fatalf("wrong model in event: %v", last)
	}
	if last["writeback_applied"] != true {
		t.Fatalf("expected writeback_applied=true, got %v", last["writeback_applied"])
	}

	// Config file should now have a larger memory_gb.
	got, err := os.ReadFile(filepath.Join(root, "local", "config.json"))
	if err != nil {
		t.Fatalf("read updated config: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(got, &parsed); err != nil {
		t.Fatalf("decode updated config: %v", err)
	}
	model := parsed["models"].(map[string]any)["test-model"].(map[string]any)
	newGB := model["memory_gb"].(float64)
	if newGB <= 4.0 {
		t.Fatalf("memory_gb not increased: %v", newGB)
	}
	// In-memory config also updated
	if cfg.Models["test-model"].MemoryGB != newGB {
		t.Fatalf("in-memory cfg not updated: %v vs %v", cfg.Models["test-model"].MemoryGB, newGB)
	}
}

func TestMemoryWatchdog_DriftCappedByBudget(t *testing.T) {
	root := t.TempDir()
	logDir := filepath.Join(root, "logs")
	if err := os.MkdirAll(filepath.Join(root, "local"), 0o755); err != nil {
		t.Fatalf("create local dir: %v", err)
	}
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		t.Fatalf("create log dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "local", "config.json"), []byte(`{"models":{"m":{"memory_gb":4.0}}}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg := &Config{
		VRAMBudgetGB: 100,
		Models:       map[string]ModelConfig{"m": {MemoryGB: 4}},
	}
	mgr := NewInstanceManager(cfg, "python3", root)
	logger := NewEventLogger(logDir)
	w := NewMemoryWatchdog(cfg, mgr, logger, root)

	// Wildly inflated observation should be capped at VRAMBudgetGB * driftWritebackCap.
	snap := instanceMemSnapshot{
		ModelID:      "m",
		TreeVRAMGB:   500,
		TreeRSSGB:    0,
		ConfiguredGB: 4,
	}
	w.checkDrift(snap)
	w.checkDrift(snap)
	if cfg.Models["m"].MemoryGB > cfg.VRAMBudgetGB*driftWritebackCap+0.01 {
		t.Fatalf("memory_gb exceeded cap: got %v, cap %v", cfg.Models["m"].MemoryGB, cfg.VRAMBudgetGB*driftWritebackCap)
	}
}

func TestMemoryWatchdog_NoDriftBelowThreshold(t *testing.T) {
	root := t.TempDir()
	logDir := filepath.Join(root, "logs")
	if err := os.MkdirAll(filepath.Join(root, "local"), 0o755); err != nil {
		t.Fatalf("create local dir: %v", err)
	}
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		t.Fatalf("create log dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "local", "config.json"), []byte(`{"models":{"m":{"memory_gb":4.0}}}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg := &Config{VRAMBudgetGB: 100, Models: map[string]ModelConfig{"m": {MemoryGB: 4}}}
	mgr := NewInstanceManager(cfg, "python3", root)
	logger := NewEventLogger(logDir)
	w := NewMemoryWatchdog(cfg, mgr, logger, root)

	// 1.4x is under driftRatio=1.5 — should never fire.
	snap := instanceMemSnapshot{ModelID: "m", TreeVRAMGB: 5.6, ConfiguredGB: 4}
	for i := 0; i < 5; i++ {
		w.checkDrift(snap)
	}
	if got := readEventLines(t, logDir, "model.memory_drift"); len(got) != 0 {
		t.Fatalf("expected no drift events under threshold, got %d", len(got))
	}
}

func TestMemoryWatchdog_SystemPressureEventEmitted(t *testing.T) {
	root := t.TempDir()
	logDir := filepath.Join(root, "logs")
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		t.Fatalf("create log dir: %v", err)
	}

	cfg := &Config{SystemRAMBudgetGB: 100, Models: map[string]ModelConfig{}}
	mgr := NewInstanceManager(cfg, "python3", root)
	logger := NewEventLogger(logDir)
	w := NewMemoryWatchdog(cfg, mgr, logger, root)

	w.checkSystemPressure(98) // > 100 * 0.95
	events := readEventLines(t, logDir, "system.memory_pressure")
	if len(events) == 0 {
		t.Fatalf("expected system.memory_pressure event for 98GB on 100GB budget")
	}
}

func TestMemoryWatchdog_SystemPressureDisabledWhenBudgetZero(t *testing.T) {
	root := t.TempDir()
	logDir := filepath.Join(root, "logs")
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		t.Fatalf("create log dir: %v", err)
	}

	cfg := &Config{SystemRAMBudgetGB: 0, Models: map[string]ModelConfig{}}
	mgr := NewInstanceManager(cfg, "python3", root)
	logger := NewEventLogger(logDir)
	w := NewMemoryWatchdog(cfg, mgr, logger, root)

	w.checkSystemPressure(500) // huge but disabled
	if got := readEventLines(t, logDir, "system.memory_pressure"); len(got) != 0 {
		t.Fatalf("expected no events when budget=0, got %d", len(got))
	}
}
