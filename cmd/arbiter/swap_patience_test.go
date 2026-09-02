package main

import (
	"encoding/json"
	"path/filepath"
	"testing"
	"time"
)

// TestMarkSubprocessExitedReleasesReservation covers the VRAM-accounting race
// behind the 2026-06-10 swap-thrash livelock: markSubprocessExited used to
// clear vramHeld without decrementing usedGB, so the eviction path's paired
// ReleaseMemoryFor no-opped and the challenger's reservation failed on phantom
// memory until the 15s reconciler.
func TestMarkSubprocessExitedReleasesReservation(t *testing.T) {
	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 100}, "python3", t.TempDir())
	inst := NewInstance("big-model", "big-model#0", 1, 56, "python3", ".")
	mgr.Register(inst)

	if !mgr.ReserveMemoryFor(inst, 56) {
		t.Fatal("initial reservation failed")
	}
	if free := mgr.FreeGB(); free != 44 {
		t.Fatalf("FreeGB after reserve = %.1f, want 44", free)
	}

	// Simulate the worker process dying (readLoop EOF path).
	inst.markSubprocessExited()

	if free := mgr.FreeGB(); free != 100 {
		t.Fatalf("FreeGB after subprocess exit = %.1f, want 100 — reservation not released", free)
	}
	// The paired ReleaseMemoryFor from the eviction path must now no-op,
	// not double-free.
	if gb := mgr.ReleaseMemoryFor(inst); gb != 0 {
		t.Fatalf("ReleaseMemoryFor after exit freed %.1fGB, want 0 (idempotent)", gb)
	}
	if free := mgr.FreeGB(); free != 100 {
		t.Fatalf("FreeGB after double release = %.1f, want 100", free)
	}
}

// TestEvictForGBWithQueueInfoAllowQueuedFilter verifies the swap-patience hook:
// instances of models with queued work are only evictable when the filter says
// so; nil filter preserves the old behavior.
func TestEvictForGBWithQueueInfoAllowQueuedFilter(t *testing.T) {
	makeLoaded := func(mgr *InstanceManager, model string, gb float64) *Instance {
		inst := NewInstance(model, model, 1, gb, "python3", ".")
		inst.mu.Lock()
		inst.state = "loaded"
		inst.lastActive = time.Now().Add(-time.Minute)
		inst.mu.Unlock()
		mgr.Register(inst)
		mgr.ReserveMemoryFor(inst, gb)
		return inst
	}

	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 100}, "python3", ".")
	protected := makeLoaded(mgr, "expensive-with-queue", 56)
	sacrificial := makeLoaded(mgr, "cheap-with-queue", 30)

	queued := map[string]int{"expensive-with-queue": 5, "cheap-with-queue": 2}
	allow := func(modelID string) bool { return modelID == "cheap-with-queue" }

	freed, err := mgr.EvictForGBWithQueueInfo(25, queued, allow)
	if err != nil {
		t.Fatalf("eviction failed: %v", err)
	}
	if freed < 25 {
		t.Fatalf("freed %.1f, want >= 25", freed)
	}
	if protected.State() != "loaded" {
		t.Fatalf("protected model evicted despite filter denial (state=%s)", protected.State())
	}
	if sacrificial.State() != "stopped" {
		t.Fatalf("allowed model not evicted (state=%s)", sacrificial.State())
	}

	// Filter denying everything: eviction must fail rather than evict protected.
	mgr2 := NewInstanceManager(&Config{VRAMBudgetGB: 100}, "python3", ".")
	p2 := makeLoaded(mgr2, "expensive-with-queue", 56)
	deny := func(string) bool { return false }
	if _, err := mgr2.EvictForGBWithQueueInfo(25, map[string]int{"expensive-with-queue": 5}, deny); err == nil {
		t.Fatal("expected eviction to fail when filter denies the only candidate")
	}
	if p2.State() != "loaded" {
		t.Fatalf("protected model evicted under full denial (state=%s)", p2.State())
	}
}

// TestCanEvictForSwap covers the drain-before-swap policy itself.
func TestCanEvictForSwap(t *testing.T) {
	dir := t.TempDir()
	store, err := NewStore(filepath.Join(dir, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()

	cfg := &Config{
		VRAMBudgetGB: 100,
		Models: map[string]ModelConfig{
			"denoise":    {MemoryGB: 56, LoadMs: 420000, AvgInferenceMs: 120000}, // 7-min load
			"challenger": {MemoryGB: 46, LoadMs: 120000, AvgInferenceMs: 5000},
			"nocost":     {MemoryGB: 10}, // no LoadMs configured -> 60s floor
		},
	}
	logger := NewEventLogger(filepath.Join(dir, "logs"))
	defer logger.Close()
	mgr := NewInstanceManager(cfg, "python3", dir)
	sched := NewScheduler(cfg, store, mgr, logger, dir)

	payload := json.RawMessage(`{}`)

	// Victim with no pending work: always evictable.
	if !sched.canEvictForSwap("denoise", "challenger") {
		t.Fatal("victim with empty queue must be evictable")
	}

	// Give the victim queued work and the challenger a fresh job:
	// guard must hold (challenger wait 0s << 7min*2).
	if _, err := store.CreateJob("denoise", "video-denoise2", payload, 1); err != nil {
		t.Fatalf("create denoise job: %v", err)
	}
	if _, err := store.CreateJob("challenger", "chat-completion", payload, 1); err != nil {
		t.Fatalf("create challenger job: %v", err)
	}
	if sched.canEvictForSwap("denoise", "challenger") {
		t.Fatal("young challenger must NOT evict a victim with queued work")
	}

	// Backdate the challenger's job past load_ms * patience (7min*2=840s):
	// guard must release.
	if _, err := store.db.Exec(
		"UPDATE jobs SET created_at = created_at - 900 WHERE model_id = 'challenger'"); err != nil {
		t.Fatalf("backdate challenger job: %v", err)
	}
	if !sched.canEvictForSwap("denoise", "challenger") {
		t.Fatal("starved challenger (900s wait > 840s threshold) must be allowed to evict")
	}

	// Negative patience disables the guard entirely.
	if _, err := store.db.Exec(
		"UPDATE jobs SET created_at = created_at + 900 WHERE model_id = 'challenger'"); err != nil {
		t.Fatalf("restore challenger job age: %v", err)
	}
	cfg.SwapPatience = -1
	if !sched.canEvictForSwap("denoise", "challenger") {
		t.Fatal("negative swap_patience must disable the guard")
	}
	cfg.SwapPatience = 0 // back to default 2.0

	// Unconfigured LoadMs falls back to the 60s floor: a 130s-old challenger
	// clears 60*2=120s.
	if _, err := store.CreateJob("nocost", "embed-text", payload, 1); err != nil {
		t.Fatalf("create nocost job: %v", err)
	}
	if sched.canEvictForSwap("nocost", "challenger") {
		t.Fatal("fresh challenger must not evict nocost victim (60s floor * 2)")
	}
	if _, err := store.db.Exec(
		"UPDATE jobs SET created_at = created_at - 130 WHERE model_id = 'challenger'"); err != nil {
		t.Fatalf("backdate challenger job: %v", err)
	}
	if !sched.canEvictForSwap("nocost", "challenger") {
		t.Fatal("130s challenger must clear the 120s floor threshold")
	}
}
