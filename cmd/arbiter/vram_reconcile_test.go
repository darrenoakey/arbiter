package main

import (
	"testing"
)

// makeLoadedInstance creates a registered instance pretending to hold the
// given GB of VRAM. Used by reconciliation tests.
func makeLoadedInstance(t *testing.T, mgr *InstanceManager, modelID string, gb float64) *Instance {
	t.Helper()
	inst := NewInstance(modelID, modelID, 1, gb, "python3", ".")
	inst.mu.Lock()
	inst.state = "loaded"
	inst.mu.Unlock()
	mgr.Register(inst)
	if !mgr.ReserveMemoryFor(inst, gb) {
		t.Fatalf("could not reserve %fGB for %s (budget too small?)", gb, modelID)
	}
	return inst
}

func TestReleaseMemoryFor_IsIdempotent(t *testing.T) {
	cfg := &Config{VRAMBudgetGB: 100, Models: map[string]ModelConfig{}}
	mgr := NewInstanceManager(cfg, "python3", ".")
	inst := makeLoadedInstance(t, mgr, "m", 30)
	if mgr.usedGB != 30 {
		t.Fatalf("expected 30GB used, got %v", mgr.usedGB)
	}
	freed1 := mgr.ReleaseMemoryFor(inst)
	if freed1 != 30 {
		t.Fatalf("first release should free 30GB, got %v", freed1)
	}
	if mgr.usedGB != 0 {
		t.Fatalf("expected 0GB used after release, got %v", mgr.usedGB)
	}
	freed2 := mgr.ReleaseMemoryFor(inst)
	if freed2 != 0 {
		t.Fatalf("second release should free 0GB (idempotent), got %v", freed2)
	}
	if mgr.usedGB != 0 {
		t.Fatalf("usedGB should remain 0 after double release, got %v", mgr.usedGB)
	}
}

func TestReconcileFromInstances_NoLeak_ReturnsZero(t *testing.T) {
	cfg := &Config{VRAMBudgetGB: 100, Models: map[string]ModelConfig{}}
	mgr := NewInstanceManager(cfg, "python3", ".")
	makeLoadedInstance(t, mgr, "a", 20)
	makeLoadedInstance(t, mgr, "b", 30)
	// Bookkeeping (50) matches sum of vramHeld instances (50). No leak.
	if freed := mgr.ReconcileFromInstances(); freed != 0 {
		t.Fatalf("expected 0 freed when bookkeeping is correct, got %v", freed)
	}
	if mgr.usedGB != 50 {
		t.Fatalf("usedGB drifted: %v", mgr.usedGB)
	}
}

func TestReconcileFromInstances_ReclaimsOrphanReservation(t *testing.T) {
	cfg := &Config{VRAMBudgetGB: 100, Models: map[string]ModelConfig{}}
	mgr := NewInstanceManager(cfg, "python3", ".")
	instA := makeLoadedInstance(t, mgr, "a", 20)
	instB := makeLoadedInstance(t, mgr, "b", 30)
	// Simulate the bug: instB's worker died unexpectedly. readLoop sets
	// state=error and clears vramHeld. usedGB still reflects 50GB.
	instB.mu.Lock()
	instB.state = "error"
	instB.vramHeld = false
	instB.mu.Unlock()

	freed := mgr.ReconcileFromInstances()
	if freed < 30-0.01 || freed > 30+0.01 {
		t.Fatalf("expected ~30GB reclaimed from dead instance, got %v", freed)
	}
	if mgr.usedGB < 20-0.01 || mgr.usedGB > 20+0.01 {
		t.Fatalf("usedGB should be 20 (instA only) after reconcile, got %v", mgr.usedGB)
	}
	// Sanity: instA's reservation untouched
	if !instA.vramHeld {
		t.Fatalf("instA.vramHeld should still be true")
	}
}

func TestReconcileFromInstances_ToleratesFloatDrift(t *testing.T) {
	cfg := &Config{VRAMBudgetGB: 100, Models: map[string]ModelConfig{}}
	mgr := NewInstanceManager(cfg, "python3", ".")
	makeLoadedInstance(t, mgr, "a", 20)
	// Synthetic 0.1GB drift inside tolerance — should NOT trigger reclaim.
	mgr.mu.Lock()
	mgr.usedGB += 0.1
	mgr.mu.Unlock()
	if freed := mgr.ReconcileFromInstances(); freed != 0 {
		t.Fatalf("expected no reclaim within tolerance, got freed=%v", freed)
	}
}

// Regression test for the leak that exhausted the VRAM budget after the
// inference-timeout watchdog (scheduler.go) called inst.Kill() without pairing
// it with ReleaseMemoryFor. Kill set state="stopped", so the readLoop's
// death-path treated the exit as deliberate and (before the fix) skipped
// clearing vramHeld. ReconcileFromInstances then summed the dead instance's
// memoryGB into `expected`, hiding the phantom forever.
//
// The fix (strengthened 2026-06-10): markSubprocessExited always clears
// vramHeld AND releases the reservation immediately — not merely leaving the
// phantom for the 15s reconciler. The delayed-release window was itself a bug:
// an eviction's challenger failed its reservation on the phantom GB, took the
// 30s insufficient-memory cooldown, and the displaced model reloaded — the
// swap-thrash livelock. usedGB must be correct the instant the process dies.
func TestMarkSubprocessExited_ClearsVRAMOnDeliberateShutdown(t *testing.T) {
	cfg := &Config{VRAMBudgetGB: 100, Models: map[string]ModelConfig{}}
	mgr := NewInstanceManager(cfg, "python3", ".")
	inst := makeLoadedInstance(t, mgr, "doomed", 80)

	// Simulate Kill(): state -> stopped, no ReleaseMemoryFor called.
	inst.mu.Lock()
	inst.state = "stopped"
	inst.mu.Unlock()

	// Simulate the readLoop noticing the subprocess is gone.
	inst.markSubprocessExited()

	if inst.vramHeld {
		t.Fatalf("vramHeld must be cleared after subprocess exit even on deliberate shutdown")
	}
	if inst.state != "stopped" {
		t.Fatalf("deliberate shutdown should preserve state=stopped, got %q", inst.state)
	}

	// The release is immediate — usedGB is already correct, no phantom left
	// for the reconciler to find.
	if mgr.usedGB > 0.01 {
		t.Fatalf("usedGB should be ~0 immediately after subprocess exit, got %v", mgr.usedGB)
	}
	if freed := mgr.ReconcileFromInstances(); freed > 0.01 {
		t.Fatalf("reconcile found %vGB to reclaim — release was not immediate", freed)
	}
}

// markSubprocessExited on an unexpected death (state was loaded/loading) must
// flip state to "error" AND clear vramHeld.
func TestMarkSubprocessExited_ClearsVRAMOnUnexpectedDeath(t *testing.T) {
	cfg := &Config{VRAMBudgetGB: 100, Models: map[string]ModelConfig{}}
	mgr := NewInstanceManager(cfg, "python3", ".")
	inst := makeLoadedInstance(t, mgr, "crashed", 50)
	// state is already "loaded" from makeLoadedInstance.
	inst.markSubprocessExited()
	if inst.vramHeld {
		t.Fatalf("vramHeld must be cleared after subprocess exit")
	}
	if inst.state != "error" {
		t.Fatalf("unexpected death should flip state to error, got %q", inst.state)
	}
}

func TestReserveMemoryFor_FailsAtBudget(t *testing.T) {
	cfg := &Config{VRAMBudgetGB: 50, Models: map[string]ModelConfig{}}
	mgr := NewInstanceManager(cfg, "python3", ".")
	inst1 := NewInstance("a", "a", 1, 30, "python3", ".")
	mgr.Register(inst1)
	if !mgr.ReserveMemoryFor(inst1, 30) {
		t.Fatal("first reserve should succeed")
	}
	if !inst1.vramHeld {
		t.Fatal("inst1.vramHeld should be true after reserve")
	}
	inst2 := NewInstance("b", "b", 1, 30, "python3", ".")
	mgr.Register(inst2)
	if mgr.ReserveMemoryFor(inst2, 30) {
		t.Fatal("second reserve should fail (60 > 50 budget)")
	}
	if inst2.vramHeld {
		t.Fatal("inst2.vramHeld must NOT be set when reserve fails — would create a phantom reservation")
	}
}
