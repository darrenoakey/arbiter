package main

import (
	"path/filepath"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

// Tests for the remote-placement-starvation fixes (Fix 1-7). Each asserts the
// NEW behavioural contract; together they would fail red against the pre-fix
// binary. They reuse the existing newRemoteTestScheduler / markLoaded / setupInstances
// scaffolding and the in-package ability to set inst.state / inst.lastActive directly.

// ---- Config builder shared by the InstanceManager-only tests ----

func remotePlacementConfig() *Config {
	return &Config{
		VRAMBudgetGB: 100,
		Hosts: map[string]HostConfig{
			"h1":          {Addr: "http://10.255.255.1:11434", Kind: "mlx", BudgetGB: 64},
			"h2":          {Addr: "http://10.255.255.2:11434", Kind: "mlx", BudgetGB: 64},
			"boringstack": {Addr: "http://10.255.255.3:11434", Kind: "mlx", BudgetGB: 96},
		},
		Models: map[string]ModelConfig{},
	}
}

// markLoadedIdle sets an instance loaded + idle with a non-zero lastActive so it
// is a candidate for the eviction / reclaimable paths.
func markLoadedIdle(t *testing.T, inst *Instance) {
	t.Helper()
	inst.mu.Lock()
	inst.state = "loaded"
	inst.lastActive = time.Now().Add(-time.Minute)
	inst.mu.Unlock()
}

// T1 — a loaded idle REMOTE instance survives a queue-priority sweep triggered
// by another model's backlog, while a loaded idle LOCAL one is evicted. (Fix 1 / Defect 3)
func TestEvictIdleNoQueueModelsSkipsRemoteInstances(t *testing.T) {
	cfg := remotePlacementConfig()
	one := 1
	cfg.Models["rm"] = ModelConfig{MemoryGB: 10, MaxConcurrent: 1, MaxInstances: &one, PressureIndex: pi(), Placements: []string{"h1"}}
	cfg.Models["lm"] = ModelConfig{MemoryGB: 10, MaxConcurrent: 1, MaxInstances: &one, PressureIndex: pi(), Placements: []string{"spark"}}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())
	mgr.SetReachabilityFunc(func(string) bool { return true })

	remote := mgr.GetModelInstances("rm")[0]
	local := mgr.GetModelInstances("lm")[0]
	markLoadedIdle(t, remote)
	markLoadedIdle(t, local)

	evicted, err := mgr.EvictIdleNoQueueModels(map[string]int{"backlog": 1})
	if err != nil {
		t.Fatalf("EvictIdleNoQueueModels: %v", err)
	}
	if remote.State() == "loaded" {
		// good: remote survived
	} else {
		t.Fatalf("remote idle instance was evicted (state=%s); remotes must never be local-VRAM candidates", remote.State())
	}
	if local.State() == "loaded" {
		t.Fatalf("local idle instance was NOT evicted; it must be reclaimed to make room for the backlog")
	}
	if evicted != 1 {
		t.Fatalf("evicted=%d, want 1 (local only; remote skipped)", evicted)
	}
}

// T2 — EvictForGB evicts NOTHING when only remote instances are loaded idle
// (ReleaseMemoryFor returns 0 for remotes); remotes are never killed for 0 GB. (Fix 1)
func TestEvictForGBIgnoresRemoteCandidates(t *testing.T) {
	cfg := remotePlacementConfig()
	one := 1
	cfg.Models["rm"] = ModelConfig{MemoryGB: 10, MaxConcurrent: 1, MaxInstances: &one, PressureIndex: pi(), Placements: []string{"h1", "h2"}}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())
	mgr.SetReachabilityFunc(func(string) bool { return true })
	for _, inst := range mgr.GetModelInstances("rm") {
		markLoadedIdle(t, inst)
	}
	if err := mgr.EvictForGB(10); err == nil {
		t.Fatalf("EvictForGB unexpectedly freed memory from remote-only idle set; remotes hold zero spark VRAM")
	}
	for _, inst := range mgr.GetModelInstances("rm") {
		if inst.State() != "loaded" {
			t.Fatalf("remote instance %s was evicted (state=%s) for 0 GB; must survive", inst.InstanceID, inst.State())
		}
	}
}

// T3 — an idle remote instance does NOT inflate ReclaimableIdleGB. (Fix 1)
func TestReclaimableIdleGBExcludesRemote(t *testing.T) {
	cfg := remotePlacementConfig()
	one := 1
	cfg.Models["rm"] = ModelConfig{MemoryGB: 10, MaxConcurrent: 1, MaxInstances: &one, PressureIndex: pi(), Placements: []string{"h1"}}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())
	mgr.SetReachabilityFunc(func(string) bool { return true })
	markLoadedIdle(t, mgr.GetModelInstances("rm")[0])
	if got := mgr.ReclaimableIdleGB("other"); got != 0 {
		t.Fatalf("ReclaimableIdleGB=%v, want 0 (remote residency is not spark-reclaimable)", got)
	}
}

// T8 — a remote-warm model with no pending work survives a tick where another
// model has backlog. (Fix 1 / Defect 3 — the "local backlog cools a remote-warm
// model" failure mode.)
func TestRemoteWarmSurvivesLocalBacklogTick(t *testing.T) {
	cfg := remotePlacementConfig()
	one := 1
	cfg.Models["qm"] = ModelConfig{MemoryGB: 10, MaxConcurrent: 1, MaxInstances: &one, PressureIndex: pi(), Placements: []string{"h1"}}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())
	mgr.SetReachabilityFunc(func(string) bool { return true })
	remote := mgr.GetModelInstances("qm")[0]
	markLoadedIdle(t, remote)
	// LTX has backlog; qm has none → pre-fix the idle remote would be queue-evicted.
	if _, err := mgr.EvictIdleNoQueueModels(map[string]int{"ltx2-dev-denoise1": 5, "qm": 0}); err != nil {
		t.Fatalf("evict: %v", err)
	}
	if remote.State() != "loaded" {
		t.Fatalf("remote-warm qwen cooled by local backlog (state=%s); must stay loaded", remote.State())
	}
}

// T4 — ClearExcludedHostForActiveJobs: an AGED exclusion is cleared on recovery,
// a FRESH (<minAge) one survives. (Fix 2)
func TestExclusionClearedOnHostRecovery(t *testing.T) {
	store, cleanup := newStoreForPlacementTest(t)
	defer cleanup()

	aged, _ := store.CreateJob("m", "t", nil, 0)
	if _, err := store.AddExcludedHost(aged.ID, "h1"); err != nil {
		t.Fatalf("add excluded: %v", err)
	}
	// Force the recorded write-time into the past so it is stale/clearable.
	store.excludedAt[aged.ID]["h1"] = time.Now().Add(-time.Minute)

	fresh, _ := store.CreateJob("m", "t", nil, 0)
	if _, err := store.AddExcludedHost(fresh.ID, "h1"); err != nil {
		t.Fatalf("add excluded: %v", err)
	}
	// fresh.excludedAt[h1] is now (~0s old) → must survive a 30s minAge clear.

	healed, err := store.ClearExcludedHostForActiveJobs("h1", 30*time.Second)
	if err != nil {
		t.Fatalf("clear: %v", err)
	}
	if healed != 1 {
		t.Fatalf("healed=%d, want 1 (only the aged exclusion clears)", healed)
	}
	got, _ := store.GetJob(aged.ID)
	if got.HostExcluded("h1") {
		t.Fatalf("aged exclusion was NOT cleared")
	}
	gotFresh, _ := store.GetJob(fresh.ID)
	if !gotFresh.HostExcluded("h1") {
		t.Fatalf("fresh exclusion was cleared; must survive the minAge dampener")
	}
}

// T5 — when honouring every exclusion leaves no host, STALE exclusions are
// relaxed so the job places with reason exclusion_relaxed. (Fix 3)
func TestJobNeverPermanentlyUnplaceable(t *testing.T) {
	cfg := remotePlacementConfig()
	one := 1
	cfg.Models["m"] = ModelConfig{MemoryGB: 10, MaxConcurrent: 1, MaxInstances: &one, PressureIndex: pi(), Placements: []string{"h1", "h2"}}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())
	mgr.SetReachabilityFunc(func(string) bool { return true })

	job := &Job{ID: "j1", ModelID: "m", State: "queued", ExcludedHosts: []string{"h1", "h2"}}
	// Normal pick: both placements excluded → nowhere to go.
	if inst := mgr.PickInstanceForJob(job, true); inst != nil {
		t.Fatalf("normal pick returned %s; want nil with all placements excluded", hostOf(inst))
	}
	// Relaxed backstop: h1 is stale → relax → place on h1.
	isStale := func(host string) bool { return host == "h1" }
	inst, reason := mgr.PickInstanceRelaxedForJob(job, true, isStale)
	if inst == nil {
		t.Fatalf("relaxed pick returned nil; a stale-excluded reachable host must relax")
	}
	if inst.host != "h1" {
		t.Fatalf("relaxed pick host=%s, want h1", inst.host)
	}
	if reason != "exclusion_relaxed" {
		t.Fatalf("relaxed pick reason=%q, want exclusion_relaxed", reason)
	}
}

// T11 — a FRESH exclusion (< staleExclusionMinAge) is NOT relaxed. (Fix 3 hardening)
func TestRelaxedPassHonorsFreshExclusions(t *testing.T) {
	cfg := remotePlacementConfig()
	one := 1
	cfg.Models["m"] = ModelConfig{MemoryGB: 10, MaxConcurrent: 1, MaxInstances: &one, PressureIndex: pi(), Placements: []string{"h1", "h2"}}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())
	mgr.SetReachabilityFunc(func(string) bool { return true })

	job := &Job{ID: "j1", ModelID: "m", State: "queued", ExcludedHosts: []string{"h1", "h2"}}
	// Both exclusions fresh → relaxed pass must NOT relax either → nil.
	isStale := func(host string) bool { return false }
	if inst, _ := mgr.PickInstanceRelaxedForJob(job, true, isStale); inst != nil {
		t.Fatalf("relaxed pick placed on %s despite fresh exclusions; fresh exclusions must be honoured", hostOf(inst))
	}
}

// T6 — the acceptance test: one pass of the extracted dispatch step dispatches a
// placeable sibling (B) queued BEHIND an unplaceable job (A), reproducing (in
// ~1s) the 24-job/66-min head-of-line freeze. (Fix 4)
func TestUnplaceableJobDoesNotBlockSiblings(t *testing.T) {
	cfg := remotePlacementConfig()
	one := 1
	cfg.Models["m"] = ModelConfig{MemoryGB: 10, MaxConcurrent: 4, MaxInstances: &one, PressureIndex: pi(), Placements: []string{"h1", "h2"}}
	sched, store, mgr, cleanup := newRemoteTestScheduler(t, cfg)
	defer cleanup()
	mgr.SetReachabilityFunc(func(string) bool { return true })
	// Pre-warm h1 so a placeable job dispatches without a real load.
	markLoaded(t, mgr, "m")

	// Job A: all placements freshly excluded → unplaceable (relaxed pass honouring
	// fresh exclusions will NOT fire). Created first so it is FIFO-ahead of B.
	jobA, _ := store.CreateJob("m", "t", nil, 0)
	if _, err := store.AddExcludedHost(jobA.ID, "h1"); err != nil {
		t.Fatalf("exclude h1: %v", err)
	}
	if _, err := store.AddExcludedHost(jobA.ID, "h2"); err != nil {
		t.Fatalf("exclude h2: %v", err)
	}
	// Job B: no exclusions → placeable on h1.
	jobB, _ := store.CreateJob("m", "t", nil, 0)

	ok := sched.dispatchOneForModel("m")
	if !ok {
		t.Fatalf("dispatchOneForModel returned false; a placeable sibling (B) must dispatch past the unplaceable A")
	}
	a, _ := store.GetJob(jobA.ID)
	if a.State != "queued" {
		t.Fatalf("job A state=%q, want queued (scanned, failed, requeued — NOT blocking)", a.State)
	}
	// B must be the dispatched one (in-flight), proving the scan stepped past A.
	if !sched.isInFlight(jobB.ID) {
		t.Fatalf("job B was not dispatched; the scan must step past unplaceable A to dispatch B")
	}
}

// T7 — admission-side pressure exemption: a model with reachable remote capacity
// is NOT gated out by spark GPU pressure (qwen constructed as a NON-best model so
// the bestModel exemption doesn't make the assertion vacuous); a local-only model
// at the same pressure IS gated. Asserts the Fix-5 asymmetry. (Fix 5)
func TestRemoteDispatchChargesZeroPressure(t *testing.T) {
	cfg := remotePlacementConfig()
	one := 1
	cfg.Models["rm"] = ModelConfig{MemoryGB: 10, MaxConcurrent: 1, MaxInstances: &one, PressureIndex: ptr(0.9), Placements: []string{"h1"}}
	cfg.Models["lm"] = ModelConfig{MemoryGB: 10, MaxConcurrent: 1, MaxInstances: &one, PressureIndex: ptr(0.9), Placements: []string{"spark"}}
	sched, _, mgr, cleanup := newRemoteTestScheduler(t, cfg)
	defer cleanup()
	mgr.SetReachabilityFunc(func(string) bool { return true })
	// Make the remote placement report reachable capacity so remoteServable is true.
	markLoaded(t, mgr, "rm")

	// Charge pressure past the budget so a non-exempt model would be full.
	sched.pressureMu.Lock()
	sched.currentPressure = 0.5
	sched.pressureMu.Unlock()

	full := sched.getFullModels("") // bestModel="" → no model gets the bestModel bypass
	if full["rm"] {
		t.Fatalf("remote-capacity model rm marked full by pressure; reachable-remote-capacity must be exempt (Fix 5)")
	}
	if !full["lm"] {
		t.Fatalf("local-only model lm NOT marked full (0.5+0.9 > budget); the pressure gate must still charge local-only models")
	}
}

// T10 — a relaxed-placement failure can NEVER terminally fail a job: it routes
// to the separate bounded budget, requeues the job, and disarms after the bound.
// (Fix 3 hardening)
func TestRelaxedPassNeverTerminallyFails(t *testing.T) {
	cfg := remotePlacementConfig()
	one := 1
	cfg.Models["m"] = ModelConfig{MemoryGB: 10, MaxConcurrent: 1, MaxInstances: &one, PressureIndex: pi(), Placements: []string{"h1"}}
	sched, store, mgr, cleanup := newRemoteTestScheduler(t, cfg)
	defer cleanup()
	mgr.SetReachabilityFunc(func(string) bool { return true })
	remote := mgr.GetModelInstances("m")[0]
	remote.mu.Lock()
	remote.state = "loaded"
	remote.mu.Unlock()

	job, _ := store.CreateJob("m", "t", nil, 0)
	// Simulate that this dispatch was a relaxed placement.
	sched.inFlightMu.Lock()
	sched.inFlight[job.ID] = inFlightJob{inst: remote, pressure: 0, relaxed: true}
	sched.inFlightMu.Unlock()

	cause := syscall.ECONNREFUSED // isRemoteAbsence == true
	for i := 1; i <= relaxedExclusionMaxFailures; i++ {
		// Re-mark in-flight each iteration (routeRelaxedFailure requeues; the
		// dispatch defer would release it, but we drive the routing directly).
		sched.inFlightMu.Lock()
		sched.inFlight[job.ID] = inFlightJob{inst: remote, pressure: 0, relaxed: true}
		sched.inFlightMu.Unlock()
		applied := sched.routeRelaxedFailure(job, remote, cause)
		if !applied {
			t.Fatalf("iteration %d: routeRelaxedFailure returned false (terminal); relaxed failures must NEVER terminally fail", i)
		}
		got, _ := store.GetJob(job.ID)
		if got.State != "queued" {
			t.Fatalf("iteration %d: job state=%q, want queued (returned to waiting)", i, got.State)
		}
		// Relaxed failures must NOT count toward the normal failover budget.
		sched.failoverMu.Lock()
		fo := sched.failoverAttempts[job.ID]
		sched.failoverMu.Unlock()
		if fo != 0 {
			t.Fatalf("iteration %d: failoverAttempts=%d; relaxed failures must not count toward maxFailoverAttempts", i, fo)
		}
	}
	// After exhausting the bound the relaxed pass disarms.
	sched.relaxedMu.Lock()
	disarmed := sched.relaxedDisarmed[job.ID]
	sched.relaxedMu.Unlock()
	if disarmed.IsZero() {
		t.Fatalf("relaxed pass not disarmed after exhausting the bound")
	}
	if sched.relaxedArmed(job.ID) {
		t.Fatalf("relaxed pass still armed immediately after disarming; cooldown must hold")
	}
}

// T12 — NoRemoteSpill: a reachable but full higher-preference remote host does
// NOT cause a spill to a lower-preference remote host. The job waits instead.
func TestNoRemoteSpillSkipsLowerPreferenceRemoteWhenHigherIsFull(t *testing.T) {
	cfg := remotePlacementConfig()
	one := 1
	noSpill := true
	cfg.Models["m"] = ModelConfig{
		MemoryGB: 10, MaxConcurrent: 1, MaxInstances: &one,
		PressureIndex: pi(), Placements: []string{"h1", "h2"},
		NoRemoteSpill: &noSpill,
	}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())
	mgr.SetReachabilityFunc(func(string) bool { return true })

	// h1 is loaded and busy (max_concurrent exhausted).
	h1 := mgr.GetModelInstances("m")[0]
	if h1.host != "h1" {
		t.Fatalf("expected first instance on h1, got %s", h1.host)
	}
	markLoaded(t, mgr, "m")
	if h1.State() != "loaded" {
		t.Fatalf("h1 not loaded: %s", h1.State())
	}
	// Simulate h1 at capacity by starting one job and advancing it to in-progress.
	store, cleanup := newStoreForPlacementTest(t)
	defer cleanup()
	job, err := store.CreateJob("m", "t", nil, 0)
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	job.State = "in-progress"
	atomic.StoreInt32(&h1.activeJobs, 1)

	// A second job must not spill to h2 while h1 is reachable but full.
	job2 := &Job{ID: "j2", ModelID: "m", State: "queued"}
	if inst := mgr.PickInstanceForJob(job2, true); inst != nil {
		t.Fatalf("NoRemoteSpill=true: picked %s when h1 is reachable but full; want nil", hostOf(inst))
	}
}

// T13 — NoRemoteSpill still allows failover to a lower-preference remote host
// when the higher-preference remote host is UNREACHABLE.
func TestNoRemoteSpillAllowsRemoteFailoverWhenHigherIsAbsent(t *testing.T) {
	cfg := remotePlacementConfig()
	one := 1
	noSpill := true
	cfg.Models["m"] = ModelConfig{
		MemoryGB: 10, MaxConcurrent: 1, MaxInstances: &one,
		PressureIndex: pi(), Placements: []string{"h1", "h2"},
		NoRemoteSpill: &noSpill,
	}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())
	// h1 is reachable=false, h2 is reachable=true.
	mgr.SetReachabilityFunc(func(host string) bool { return host != "h1" })
	markLoaded(t, mgr, "m")

	job := &Job{ID: "j1", ModelID: "m", State: "queued"}
	inst, reason := mgr.PickInstanceForJobWithReason(job, true)
	if inst == nil {
		t.Fatalf("NoRemoteSpill=true: picked nil when h1 is absent; want h2")
	}
	if inst.host != "h2" {
		t.Fatalf("NoRemoteSpill=true: picked %s when h1 is absent; want h2", inst.host)
	}
	if reason != reasonSpill {
		t.Fatalf("reason=%q, want %q (absence failover is still spill)", reason, reasonSpill)
	}
}

// ---- small helpers local to this file ----

func newStoreForPlacementTest(t *testing.T) (*Store, func()) {
	t.Helper()
	dir := t.TempDir()
	store, err := NewStore(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	return store, func() { store.Close() }
}

func ptr(f float64) *float64 { return &f }
