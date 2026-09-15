package main

import (
	"testing"
	"time"
)

// These tests pin the fix for a real outage (2026-09-15): ltx25-denoise1 needs
// 82GB of a 90GB budget. After a redeploy it lost residency and could never get
// it back — every time enough VRAM freed, a 23.7GB vision model or a 32GB LLM
// took the window within milliseconds, the big load failed, the job cooled down
// 30s, and the cycle repeated for over 100 minutes with a full queue and an
// idle GPU. A starved model must be able to hold the memory it needs.

func newClaimTestManager(t *testing.T, budgetGB float64) *InstanceManager {
	t.Helper()
	return NewInstanceManager(&Config{VRAMBudgetGB: budgetGB}, "python3", t.TempDir())
}

func TestStarvationClaimWithholdsMemoryFromOtherModels(t *testing.T) {
	mgr := newClaimTestManager(t, 90)

	if !mgr.ClaimVRAMForStarvedModel("ltx25-denoise1", 82, time.Minute) {
		t.Fatal("first claim was refused")
	}
	if holder, gb := mgr.VRAMClaimHolder(); holder != "ltx25-denoise1" || gb != 82 {
		t.Fatalf("VRAMClaimHolder() = (%q, %v), want (ltx25-denoise1, 82)", holder, gb)
	}

	// A small model must NOT be able to take memory the claimant needs.
	if mgr.reserveMemoryForModel("moondream", 23.7) {
		t.Fatal("small model reserved VRAM while a starvation claim was held")
	}
	if got := mgr.FreeGBFor("moondream"); got > 8.0001 {
		t.Fatalf("FreeGBFor(other) = %v, want <= 8 (90 - 82 claimed)", got)
	}

	// The claimant itself sees the memory and can take it.
	if got := mgr.FreeGBFor("ltx25-denoise1"); got < 89.9999 {
		t.Fatalf("FreeGBFor(claimant) = %v, want full 90", got)
	}
	if !mgr.reserveMemoryForModel("ltx25-denoise1", 82) {
		t.Fatal("claimant could not reserve the memory it claimed")
	}

	// Reserving retires the claim so the fleet returns to normal rules.
	if holder, _ := mgr.VRAMClaimHolder(); holder != "" {
		t.Fatalf("claim still held by %q after the claimant reserved", holder)
	}
}

func TestStarvationClaimExpires(t *testing.T) {
	mgr := newClaimTestManager(t, 90)
	if !mgr.ClaimVRAMForStarvedModel("ltx25-denoise1", 82, 150*time.Millisecond) {
		t.Fatal("claim was refused")
	}
	if mgr.reserveMemoryForModel("moondream", 23.7) {
		t.Fatal("reservation succeeded while the claim was live")
	}

	time.Sleep(250 * time.Millisecond)

	if holder, _ := mgr.VRAMClaimHolder(); holder != "" {
		t.Fatalf("claim %q outlived its TTL — a model that can never load would stall the fleet", holder)
	}
	if !mgr.reserveMemoryForModel("moondream", 23.7) {
		t.Fatal("small model still blocked after the claim expired")
	}
}

func TestOnlyOneStarvationClaimAtATime(t *testing.T) {
	mgr := newClaimTestManager(t, 90)
	if !mgr.ClaimVRAMForStarvedModel("ltx25-denoise1", 82, time.Minute) {
		t.Fatal("first claim refused")
	}
	if mgr.ClaimVRAMForStarvedModel("ltx2", 74, time.Minute) {
		t.Fatal("a second model took a competing claim — claims must be exclusive")
	}
	// Renewal by the SAME model is how the scheduler keeps the hold alive.
	if !mgr.ClaimVRAMForStarvedModel("ltx25-denoise1", 82, time.Minute) {
		t.Fatal("claimant could not renew its own claim")
	}
	mgr.ReleaseVRAMClaim("ltx25-denoise1")
	if !mgr.ClaimVRAMForStarvedModel("ltx2", 74, time.Minute) {
		t.Fatal("claim not available after release")
	}
}

func TestClaimDoesNotBlockWhenMemoryStillFits(t *testing.T) {
	mgr := newClaimTestManager(t, 90)
	mgr.ClaimVRAMForStarvedModel("ltx25-denoise1", 82, time.Minute)
	// 8GB is still free beyond the claim, so a small model may proceed —
	// the claim withholds only what the starved model actually needs.
	if !mgr.reserveMemoryForModel("aesthetic-scorer", 1) {
		t.Fatal("a model that fits alongside the claim was wrongly blocked")
	}
}

// TestSchedulerClaimsOnlyAfterSustainedStarvation proves the threshold: a model
// that has merely been queued briefly must not seize memory from everyone else.
func TestSchedulerClaimsOnlyAfterSustainedStarvation(t *testing.T) {
	mgr := newClaimTestManager(t, 90)
	sched := &Scheduler{mgr: mgr}

	sched.claimVRAMIfStarving("ltx25-denoise1", 82, vramStarvationSeconds-1)
	if holder, _ := mgr.VRAMClaimHolder(); holder != "" {
		t.Fatalf("claimed after only a short wait (holder=%q)", holder)
	}

	sched.claimVRAMIfStarving("ltx25-denoise1", 82, vramStarvationSeconds+1)
	if holder, _ := mgr.VRAMClaimHolder(); holder != "ltx25-denoise1" {
		t.Fatalf("no claim after sustained starvation (holder=%q)", holder)
	}
}

// TestSchedulerNeverClaimsForAnImpossibleModel: a model bigger than the whole
// budget can never load, so claiming for it would stall every other model.
func TestSchedulerNeverClaimsForAnImpossibleModel(t *testing.T) {
	mgr := newClaimTestManager(t, 90)
	sched := &Scheduler{mgr: mgr}
	sched.claimVRAMIfStarving("too-big", 120, vramStarvationSeconds*10)
	if holder, _ := mgr.VRAMClaimHolder(); holder != "" {
		t.Fatalf("claimed for an unloadable model (holder=%q)", holder)
	}
}
