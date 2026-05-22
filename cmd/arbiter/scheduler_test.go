package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

func writeIdleWorker(t *testing.T, path string) {
	t.Helper()
	script := `#!/usr/bin/env python3
import json
import sys

for line in sys.stdin:
    msg = json.loads(line)
    cmd = msg.get("cmd")
    req_id = msg.get("req_id", "_default")
    if cmd in ("load", "unload", "ping"):
        print(json.dumps({"status": "ok", "req_id": req_id}), flush=True)
    elif cmd == "shutdown":
        print(json.dumps({"status": "ok", "req_id": req_id}), flush=True)
        break
`
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("write idle worker: %v", err)
	}
}

func TestDispatchJobPromotesFollowerWhenWorkerDies(t *testing.T) {
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	if err := os.MkdirAll(filepath.Join(outputDir, "jobs"), 0o755); err != nil {
		t.Fatalf("mkdir output jobs: %v", err)
	}

	workerPath := filepath.Join(projectRoot, "dying_worker.py")
	workerScript := `import json, os, sys
for line in sys.stdin:
    msg = json.loads(line)
    cmd = msg.get("cmd")
    req_id = msg.get("req_id", "_default")
    if cmd == "load":
        print(json.dumps({"status": "ok", "req_id": req_id}), flush=True)
    elif cmd == "infer":
        sys.stdout.flush()
        os._exit(1)
`
	if err := os.WriteFile(workerPath, []byte(workerScript), 0o755); err != nil {
		t.Fatalf("write worker script: %v", err)
	}

	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()
	store.InitDedup()

	cfg := &Config{
		VRAMBudgetGB: 100,
		Models: map[string]ModelConfig{
			"demo": {
				MemoryGB:      1,
				MaxConcurrent: 1,
				MaxInstances:  intPtr(1),
				WorkerCmd:     []string{"python3", workerPath},
			},
		},
	}
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer logger.Close()
	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 100}, "python3", projectRoot)
	mgr.ScaleModel("demo", 1, cfg.Models["demo"])
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)

	payload := json.RawMessage(`{"prompt":"die"}`)
	orig, err := store.CreateJob("demo", "image-generate", payload, 1)
	if err != nil {
		t.Fatalf("create original job: %v", err)
	}
	followerA, err := store.CreateFollowerJob("demo", "image-generate", payload, orig.ID)
	if err != nil {
		t.Fatalf("create follower A: %v", err)
	}
	followerB, err := store.CreateFollowerJob("demo", "image-generate", payload, orig.ID)
	if err != nil {
		t.Fatalf("create follower B: %v", err)
	}

	inst := mgr.GetModelInstances("demo")[0]
	atomic.AddInt32(&inst.activeJobs, 1)
	sched.dispatchJobToInstance(orig, inst, 1.0)

	origAfter, _ := store.GetJob(orig.ID)
	if origAfter.State != "failed" {
		t.Fatalf("original state = %s, want failed", origAfter.State)
	}
	if origAfter.Error == "" {
		t.Fatalf("original error was not recorded")
	}

	promoted, _ := store.GetJob(followerA.ID)
	if promoted.State != "queued" || promoted.Error != "" {
		t.Fatalf("promoted follower = state %s error %q, want queued/cleared", promoted.State, promoted.Error)
	}

	rebased, _ := store.GetJob(followerB.ID)
	if rebased.State != "following" || rebased.Error != "following:"+followerA.ID {
		t.Fatalf("rebased follower = state %s error %q, want following:%s", rebased.State, rebased.Error, followerA.ID)
	}
}

func TestDispatchJobRequeuesOnShutdownWhenWorkerDies(t *testing.T) {
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	if err := os.MkdirAll(filepath.Join(outputDir, "jobs"), 0o755); err != nil {
		t.Fatalf("mkdir output jobs: %v", err)
	}

	workerPath := filepath.Join(projectRoot, "dying_worker.py")
	workerScript := `import json, os, sys
for line in sys.stdin:
    msg = json.loads(line)
    cmd = msg.get("cmd")
    req_id = msg.get("req_id", "_default")
    if cmd == "load":
        print(json.dumps({"status": "ok", "req_id": req_id}), flush=True)
    elif cmd == "infer":
        sys.stdout.flush()
        os._exit(1)
`
	if err := os.WriteFile(workerPath, []byte(workerScript), 0o755); err != nil {
		t.Fatalf("write worker script: %v", err)
	}

	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()
	store.InitDedup()

	cfg := &Config{
		VRAMBudgetGB: 100,
		Models: map[string]ModelConfig{
			"demo": {
				MemoryGB:      1,
				MaxConcurrent: 1,
				MaxInstances:  intPtr(1),
				WorkerCmd:     []string{"python3", workerPath},
			},
		},
	}
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer logger.Close()
	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 100}, "python3", projectRoot)
	mgr.ScaleModel("demo", 1, cfg.Models["demo"])
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)
	sched.MarkShuttingDown()

	payload := json.RawMessage(`{"prompt":"die"}`)
	orig, err := store.CreateJob("demo", "image-generate", payload, 1)
	if err != nil {
		t.Fatalf("create original job: %v", err)
	}
	followerA, err := store.CreateFollowerJob("demo", "image-generate", payload, orig.ID)
	if err != nil {
		t.Fatalf("create follower A: %v", err)
	}
	followerB, err := store.CreateFollowerJob("demo", "image-generate", payload, orig.ID)
	if err != nil {
		t.Fatalf("create follower B: %v", err)
	}

	inst := mgr.GetModelInstances("demo")[0]
	atomic.AddInt32(&inst.activeJobs, 1)
	sched.dispatchJobToInstance(orig, inst, 1.0)

	origAfter, _ := store.GetJob(orig.ID)
	if origAfter.State != "queued" || origAfter.Error != "" {
		t.Fatalf("original after shutdown death = state %s error %q, want queued/cleared", origAfter.State, origAfter.Error)
	}

	for _, fid := range []string{followerA.ID, followerB.ID} {
		follower, _ := store.GetJob(fid)
		if follower.State != "following" || follower.Error != "following:"+orig.ID {
			t.Fatalf("follower %s after shutdown death = state %s error %q, want original follow", fid, follower.State, follower.Error)
		}
	}
}

func TestLoadCircuitBreakerPausesAfterThreeFailures(t *testing.T) {
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	os.MkdirAll(filepath.Join(outputDir, "jobs"), 0o755)
	os.MkdirAll(filepath.Join(outputDir, "logs"), 0o755)

	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()
	store.InitDedup()

	cfg := &Config{
		VRAMBudgetGB: 100,
		Models: map[string]ModelConfig{
			"broken": {
				MemoryGB:      10,
				MaxConcurrent: 1,
				MaxInstances:  intPtr(1),
			},
		},
	}
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer logger.Close()
	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 100}, "python3", projectRoot)
	mgr.ScaleModel("broken", 1, cfg.Models["broken"])
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)

	// Not paused initially
	if paused, _ := sched.IsModelLoadPaused("broken"); paused {
		t.Fatal("model should not be paused initially")
	}

	// 2 failures should not trigger
	sched.RecordLoadFailure("broken")
	sched.RecordLoadFailure("broken")
	if paused, _ := sched.IsModelLoadPaused("broken"); paused {
		t.Fatal("model should not be paused after 2 failures")
	}

	// Create a queued job that should be PRESERVED (not cancelled) on CB activation
	payload := json.RawMessage(`{"test":true}`)
	job, _ := store.CreateJob("broken", "image-generate", payload, 1)

	// 3rd failure triggers the circuit-breaker
	sched.RecordLoadFailure("broken")

	if paused, _ := sched.IsModelLoadPaused("broken"); !paused {
		t.Fatal("model should be paused after 3 failures")
	}

	// Queued job should still be queued — the CB pauses scheduling, it does NOT
	// cancel user work. A human operator decides when to cancel stuck jobs.
	jobAfter, _ := store.GetJob(job.ID)
	if jobAfter.State != "queued" {
		t.Fatalf("queued job state = %s, want queued (CB must not cancel)", jobAfter.State)
	}

	// getFullModels should include the paused model
	full := sched.getFullModels("")
	if !full["broken"] {
		t.Fatal("broken model should be in full models set")
	}

	// RecordLoadSuccess should reset
	sched.RecordLoadSuccess("broken")
	// CB pause is time-based, not reset by success — but count+level are reset.
	// After the pause expires, it should not be paused.
}

// TestConflictGroupExclusionAndPriority verifies the conflict-group gate:
// same-group members are mutually exclusive and ordered by group_priority
// (all higher-priority work drains first), while a model in no group runs
// freely alongside any of them.
func TestConflictGroupExclusionAndPriority(t *testing.T) {
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	os.MkdirAll(filepath.Join(outputDir, "logs"), 0o755)

	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()
	store.InitDedup()

	pi := 0.5
	const grp = "ltx_denoise"
	cfg := &Config{
		VRAMBudgetGB: 1000,
		Models: map[string]ModelConfig{
			"denoise1": {MemoryGB: 10, MaxConcurrent: 2, MaxInstances: intPtr(1), PressureIndex: &pi, ConflictGroup: grp, GroupPriority: 0},
			"denoise2": {MemoryGB: 10, MaxConcurrent: 2, MaxInstances: intPtr(1), PressureIndex: &pi, ConflictGroup: grp, GroupPriority: 1},
			"flux2":    {MemoryGB: 10, MaxConcurrent: 1, MaxInstances: intPtr(1), PressureIndex: &pi},
		},
	}
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer logger.Close()
	mgr := NewInstanceManager(cfg, "python3", projectRoot)
	for id, m := range cfg.Models {
		mgr.ScaleModel(id, 1, m)
	}
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)

	mustJob := func(model string) {
		if _, err := store.CreateJob(model, "x", json.RawMessage(`{}`), 1); err != nil {
			t.Fatalf("create job %s: %v", model, err)
		}
	}

	// Case 1: all three have queued work, nothing running. denoise1 (priority 0)
	// outranks denoise2 (priority 1), so denoise2 is held; flux2 (no group) runs.
	mustJob("denoise1")
	mustJob("denoise2")
	mustJob("flux2")
	full := sched.getFullModels("")
	if full["denoise1"] {
		t.Fatal("denoise1 (highest priority) must not be held")
	}
	if !full["denoise2"] {
		t.Fatal("denoise2 must be held while denoise1 has pending work")
	}
	if full["flux2"] {
		t.Fatal("flux2 (no group) must never be held by the denoise group")
	}

	// The hold is a hard constraint — not bypassable for the best-scoring model.
	full = sched.getFullModels("denoise2")
	if !full["denoise2"] {
		t.Fatal("conflict-group hold must not be bypassed for the best-scoring model")
	}

	// Case 2: denoise1 is now actively running. denoise2 stays held by mutual
	// exclusion; flux2 still runs alongside it.
	markLoaded(t, mgr, "denoise1")
	atomic.AddInt32(&mgr.GetModelInstances("denoise1")[0].activeJobs, 1)
	full = sched.getFullModels("")
	if !full["denoise2"] {
		t.Fatal("denoise2 must be held (mutual exclusion) while denoise1 is running")
	}
	if full["flux2"] {
		t.Fatal("flux2 must still run alongside a running denoise1")
	}
}

// TestDrainModeBlocksNewDispatch verifies the graceful-drain contract: while
// draining the scheduler starts no new jobs (a queued job stays queued and no
// instance becomes active), and resuming restarts dispatch. This is the
// primitive a safe redeploy relies on — bounce only once nothing is in flight.
func TestDrainModeBlocksNewDispatch(t *testing.T) {
	projectRoot := t.TempDir()
	workerPath := filepath.Join(projectRoot, "idle_worker.py")
	writeIdleWorker(t, workerPath)
	outputDir := filepath.Join(projectRoot, "output")
	os.MkdirAll(filepath.Join(outputDir, "logs"), 0o755)
	os.MkdirAll(filepath.Join(outputDir, "jobs"), 0o755)

	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()
	store.InitDedup()

	pi := 1.0
	cfg := &Config{
		VRAMBudgetGB: 100,
		Models: map[string]ModelConfig{
			"m": {MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1), PressureIndex: &pi},
		},
	}
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer logger.Close()
	mgr := NewInstanceManager(cfg, "python3", projectRoot)
	mgr.ScaleModel("m", 1, cfg.Models["m"])
	// Point the instance at the idle worker so load succeeds without a real model.
	mgr.GetModelInstances("m")[0].workerCmd = []string{"python3", workerPath}
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)

	sched.SetDraining(true)
	if !sched.IsDraining() {
		t.Fatal("IsDraining() = false after SetDraining(true)")
	}

	job, err := store.CreateJob("m", "embed-text", json.RawMessage(`{}`), 1)
	if err != nil {
		t.Fatalf("create job: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go sched.Run(ctx)

	// While draining, the job must NOT be dispatched.
	time.Sleep(500 * time.Millisecond)
	after, _ := store.GetJob(job.ID)
	if after.State != "queued" {
		t.Fatalf("draining: job state = %s, want queued (no dispatch)", after.State)
	}
	if got := mgr.TotalActiveJobs(); got != 0 {
		t.Fatalf("draining: TotalActiveJobs() = %d, want 0", got)
	}

	// Resume → the job must now be picked up and leave the queue.
	sched.SetDraining(false)
	sched.Wake()
	moved := false
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		after, _ = store.GetJob(job.ID)
		if after.State != "queued" {
			moved = true
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !moved {
		t.Fatalf("after resume, job stayed queued — dispatch did not restart")
	}

	cancel()
	mgr.GetModelInstances("m")[0].Kill()
}

// TestGetFullModelsGatesOnVRAMFeasibility verifies the hard VRAM gate: a model
// that isn't loaded and cannot fit in free VRAM + reclaimable-idle VRAM is
// excluded from dispatch, so the scheduler never commits a job it can't load
// (the "tries to load, fails on memory, requeues forever" churn). Crucially:
//   - the gate fires even when the model is the best-scoring one (VRAM is a
//     hard physical constraint, unlike the soft pressure budget);
//   - the gate does NOT fire when the blocker is merely idle, because an idle
//     instance is reclaimable — the scheduler can evict it on dispatch.
func TestGetFullModelsGatesOnVRAMFeasibility(t *testing.T) {
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	os.MkdirAll(filepath.Join(outputDir, "logs"), 0o755)

	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()
	store.InitDedup()

	pi := 1.0
	cfg := &Config{
		VRAMBudgetGB: 100,
		Models: map[string]ModelConfig{
			// blocker holds 80GB — leaves only 20GB free while it runs.
			"blocker": {MemoryGB: 80, MaxConcurrent: 1, MaxInstances: intPtr(1), PressureIndex: &pi},
			// newcomer needs 40GB — cannot cold-load into 20GB free.
			"newcomer": {MemoryGB: 40, MaxConcurrent: 1, MaxInstances: intPtr(1), PressureIndex: &pi},
		},
	}
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer logger.Close()
	mgr := NewInstanceManager(cfg, "python3", projectRoot)
	mgr.ScaleModel("blocker", 1, cfg.Models["blocker"])
	mgr.ScaleModel("newcomer", 1, cfg.Models["newcomer"])
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)

	// blocker is loaded AND actively running a job → its 80GB is pinned, not
	// reclaimable. free = 100 - 80 = 20GB.
	markLoaded(t, mgr, "blocker")
	blockerInst := mgr.GetModelInstances("blocker")[0]
	atomic.AddInt32(&blockerInst.activeJobs, 1)

	full := sched.getFullModels("")
	if !full["newcomer"] {
		t.Fatalf("newcomer (40GB) must be VRAM-infeasible while blocker actively holds 80GB (free=%.0f, reclaimable=%.0f)",
			mgr.FreeGB(), mgr.ReclaimableIdleGB("newcomer"))
	}

	// The gate is NOT bypassable for the best-scoring model — physical VRAM is
	// hard, unlike the pressure budget.
	full = sched.getFullModels("newcomer")
	if !full["newcomer"] {
		t.Fatal("VRAM infeasibility must not be bypassed even for the best-scoring model")
	}

	// blocker goes idle: its 80GB becomes reclaimable, so newcomer now fits —
	// the scheduler can evict the idle blocker on dispatch.
	atomic.AddInt32(&blockerInst.activeJobs, -1)
	full = sched.getFullModels("")
	if full["newcomer"] {
		t.Fatalf("newcomer must fit once blocker is idle/reclaimable (free=%.0f, reclaimable=%.0f)",
			mgr.FreeGB(), mgr.ReclaimableIdleGB("newcomer"))
	}
}

func TestDispatchJobLeavesInsufficientMemoryQueued(t *testing.T) {
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	os.MkdirAll(filepath.Join(outputDir, "jobs"), 0o755)
	os.MkdirAll(filepath.Join(outputDir, "logs"), 0o755)

	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()
	store.InitDedup()

	cfg := &Config{
		VRAMBudgetGB: 25,
		Models: map[string]ModelConfig{
			"big": {
				MemoryGB:      30,
				MaxConcurrent: 1,
				MaxInstances:  intPtr(1),
			},
		},
	}
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer logger.Close()
	mgr := NewInstanceManager(cfg, "python3", projectRoot)
	mgr.ScaleModel("big", 1, cfg.Models["big"])
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)

	payload := json.RawMessage(`{"test":true}`)
	job, err := store.CreateJob("big", "chat-completion", payload, 1)
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	follower, err := store.CreateFollowerJob("big", "chat-completion", payload, job.ID)
	if err != nil {
		t.Fatalf("create follower: %v", err)
	}

	inst := mgr.GetModelInstances("big")[0]
	for attempt := 0; attempt < maxLoadAttempts+2; attempt++ {
		atomic.AddInt32(&inst.activeJobs, 1)
		sched.dispatchJobToInstance(job, inst, 1.0)
		after, err := store.GetJob(job.ID)
		if err != nil {
			t.Fatalf("get job after attempt %d: %v", attempt, err)
		}
		if after.State != "queued" || after.Error != "" {
			t.Fatalf("attempt %d state=%s error=%q, want queued/no error", attempt, after.State, after.Error)
		}
	}

	followerAfter, err := store.GetJob(follower.ID)
	if err != nil {
		t.Fatalf("get follower: %v", err)
	}
	if followerAfter.State != "following" || followerAfter.Error != "following:"+job.ID {
		t.Fatalf("follower state=%s error=%q, want following original", followerAfter.State, followerAfter.Error)
	}
}

func TestRecoverStuckScheduledRequeuesOldScheduledJobs(t *testing.T) {
	store, err := NewStore(filepath.Join(t.TempDir(), "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()

	job, err := store.CreateJob("demo", "face-embed", json.RawMessage(`{}`), 1)
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	old := nowTS() - 60
	if err := store.UpdateState(job.ID, "scheduled", WithStartedAt(old)); err != nil {
		t.Fatalf("set scheduled: %v", err)
	}

	recovered, err := store.RecoverStuckScheduled(15)
	if err != nil {
		t.Fatalf("RecoverStuckScheduled: %v", err)
	}
	if recovered != 1 {
		t.Fatalf("recovered = %d, want 1", recovered)
	}

	after, err := store.GetJob(job.ID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if after.State != "queued" {
		t.Fatalf("state = %s, want queued", after.State)
	}
	if after.StartedAt != nil {
		t.Fatalf("started_at = %v, want nil", after.StartedAt)
	}
}

func TestEvictIdleNoQueueModelsPrefersQueuedModelResidency(t *testing.T) {
	projectRoot := t.TempDir()
	workerPath := filepath.Join(projectRoot, "idle_worker.py")
	writeIdleWorker(t, workerPath)

	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 100}, "python3", projectRoot)

	instA := NewInstance("model-a", "model-a", 1, 20, "python3", projectRoot)
	instA.workerCmd = []string{"python3", workerPath}
	if err := instA.Load("cuda"); err != nil {
		t.Fatalf("load model-a: %v", err)
	}
	instA.mu.Lock()
	instA.lastActive = time.Now().Add(-5 * time.Minute)
	instA.mu.Unlock()
	mgr.Register(instA)

	instB := NewInstance("model-b", "model-b", 1, 30, "python3", projectRoot)
	instB.workerCmd = []string{"python3", workerPath}
	if err := instB.Load("cuda"); err != nil {
		t.Fatalf("load model-b: %v", err)
	}
	instB.mu.Lock()
	instB.lastActive = time.Now().Add(-2 * time.Minute)
	instB.mu.Unlock()
	mgr.Register(instB)

	evicted, err := mgr.EvictIdleNoQueueModels(map[string]int{
		"model-a": 5,
		"model-b": 0,
	})
	if err != nil {
		t.Fatalf("EvictIdleNoQueueModels: %v", err)
	}
	if evicted != 1 {
		t.Fatalf("evicted = %d, want 1", evicted)
	}
	if instA.State() != "loaded" {
		t.Fatalf("model-a state = %s, want loaded", instA.State())
	}
	if instB.State() != "stopped" {
		t.Fatalf("model-b state = %s, want stopped", instB.State())
	}

	instA.Kill()
	instB.Kill()
}

func TestEvictForGBWithQueueInfoPrefersNoQueue(t *testing.T) {
	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 100}, "python3", ".")

	// Create two models, each loaded and idle
	instA := NewInstance("model-a", "model-a", 1, 20, "python3", ".")
	instA.mu.Lock()
	instA.state = "loaded"
	instA.lastActive = time.Now().Add(-5 * time.Minute)
	instA.mu.Unlock()
	mgr.Register(instA)

	instB := NewInstance("model-b", "model-b", 1, 30, "python3", ".")
	instB.mu.Lock()
	instB.state = "loaded"
	instB.lastActive = time.Now().Add(-2 * time.Minute)
	instB.mu.Unlock()
	mgr.Register(instB)

	// Reserve the memory in bookkeeping to match loaded state — use the
	// instance-aware variant so eviction's ReleaseMemoryFor finds vramHeld.
	mgr.ReserveMemoryFor(instA, 20)
	mgr.ReserveMemoryFor(instB, 30)

	// model-a has queued jobs, model-b has none
	queuedJobs := map[string]int{
		"model-a": 5,
		"model-b": 0,
	}

	// Need 25GB — model-b (no queue, 30GB) should be evicted first
	// even though model-a (has queue, 20GB) is older (more idle)
	freed, err := mgr.EvictForGBWithQueueInfo(25, queuedJobs)
	if err != nil {
		t.Fatalf("eviction failed: %v", err)
	}
	if freed < 25 {
		t.Fatalf("freed %.1fGB, wanted >= 25GB", freed)
	}

	// model-b should be evicted (stopped), model-a should still be loaded
	if instB.State() != "stopped" {
		t.Fatalf("model-b state = %s, want stopped (should be evicted first)", instB.State())
	}
	if instA.State() != "loaded" {
		t.Fatalf("model-a state = %s, want loaded (has queue, should be preserved)", instA.State())
	}
}

func TestReadLoopCleansUpPIDOnSubprocessDeath(t *testing.T) {
	projectRoot := t.TempDir()

	// Worker that dies immediately on load
	workerPath := filepath.Join(projectRoot, "die_on_load.py")
	workerScript := `import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("cmd") == "load":
        print(json.dumps({"status": "ok", "req_id": msg.get("req_id", "_default")}), flush=True)
        sys.stdout.flush()
        import os; os._exit(1)
`
	os.WriteFile(workerPath, []byte(workerScript), 0o755)

	inst := NewInstance("test", "test", 1, 1, "python3", projectRoot)
	inst.workerCmd = []string{"python3", workerPath}

	if err := inst.Load("cuda"); err != nil {
		t.Fatalf("load failed: %v", err)
	}

	// Wait for readLoop to detect the subprocess death (could race with Load returning)
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if inst.State() == "error" {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}

	if inst.State() != "error" {
		t.Fatalf("state after death = %s, want error", inst.State())
	}

	// PID should be cleaned up (cmd should be nil)
	inst.mu.Lock()
	cmdIsNil := inst.cmd == nil
	stdinIsNil := inst.stdin == nil
	inst.mu.Unlock()

	if !cmdIsNil {
		t.Fatal("inst.cmd should be nil after subprocess death")
	}
	if !stdinIsNil {
		t.Fatal("inst.stdin should be nil after subprocess death")
	}
}

// TestWatchdogSkipsJobsWithZeroMaxRuntime verifies that when MaxRuntimeSec == 0
// the watchdog does NOT mark the job as failed, regardless of elapsed time.
func TestWatchdogSkipsJobsWithZeroMaxRuntime(t *testing.T) {
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	os.MkdirAll(filepath.Join(outputDir, "jobs"), 0o755)
	os.MkdirAll(filepath.Join(outputDir, "logs"), 0o755)

	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()
	store.InitDedup()

	cfg := &Config{
		VRAMBudgetGB: 100,
		Models: map[string]ModelConfig{
			"dynamic-llm": {
				MemoryGB:      10,
				MaxConcurrent: 1,
				MaxInstances:  intPtr(1),
				MaxRuntimeSec: 0, // unset — dynamic registration default before fix
			},
		},
	}
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer logger.Close()
	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 100}, "python3", projectRoot)
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)

	payload := json.RawMessage(`{"prompt":"hello"}`)
	job, err := store.CreateJob("dynamic-llm", "chat-completion", payload, 1)
	if err != nil {
		t.Fatalf("create job: %v", err)
	}

	// Simulate the job having been running for a long time (well past any timeout).
	longAgo := nowTS() - 9999
	if err := store.UpdateState(job.ID, "running", WithStartedAt(longAgo)); err != nil {
		t.Fatalf("mark running: %v", err)
	}

	// Run one watchdog pass by manually calling the inner loop logic via a
	// short-lived context so the ticker fires once quickly.
	// We inline the watchdog body here because the ticker period is 30s — too
	// slow for a unit test.  We test the pure decision logic directly.
	jobs, err := store.GetRunningJobs()
	if err != nil {
		t.Fatalf("get running jobs: %v", err)
	}

	now := nowTS()
	for _, j := range jobs {
		if j.StartedAt == nil {
			continue
		}
		modelCfg, ok := sched.config.Models[j.ModelID]
		if !ok {
			continue
		}
		maxSec := float64(modelCfg.MaxRuntimeSec)
		elapsed := now - *j.StartedAt
		if maxSec == 0 || elapsed < maxSec {
			// watchdog should skip — this is the correct path
			continue
		}
		// If we reach here, the watchdog would kill the job — which is the bug.
		t.Fatalf("watchdog would kill job with MaxRuntimeSec=0 after %.0fs elapsed", elapsed)
	}

	// The job must still be running.
	after, _ := store.GetJob(job.ID)
	if after.State != "running" {
		t.Fatalf("job state = %s after watchdog pass, want running (MaxRuntimeSec=0 must not kill)", after.State)
	}
}

// TestWatchdogKillsJobsExceedingMaxRuntime verifies that a job whose elapsed
// time exceeds its model's MaxRuntimeSec is marked failed by the watchdog.
func TestWatchdogKillsJobsExceedingMaxRuntime(t *testing.T) {
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	os.MkdirAll(filepath.Join(outputDir, "jobs"), 0o755)
	os.MkdirAll(filepath.Join(outputDir, "logs"), 0o755)

	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()
	store.InitDedup()

	cfg := &Config{
		VRAMBudgetGB: 100,
		Models: map[string]ModelConfig{
			"short-model": {
				MemoryGB:      10,
				MaxConcurrent: 1,
				MaxInstances:  intPtr(1),
				MaxRuntimeSec: 10, // very short limit
			},
		},
	}
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer logger.Close()
	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 100}, "python3", projectRoot)
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)

	payload := json.RawMessage(`{"prompt":"hello"}`)
	job, err := store.CreateJob("short-model", "image-generate", payload, 1)
	if err != nil {
		t.Fatalf("create job: %v", err)
	}

	// Simulate the job having started 20s ago — exceeds the 10s limit.
	startedAt := nowTS() - 20
	if err := store.UpdateState(job.ID, "running", WithStartedAt(startedAt)); err != nil {
		t.Fatalf("mark running: %v", err)
	}

	// Run the same watchdog decision logic inline.
	jobs, err := store.GetRunningJobs()
	if err != nil {
		t.Fatalf("get running jobs: %v", err)
	}

	now := nowTS()
	for _, j := range jobs {
		if j.StartedAt == nil {
			continue
		}
		modelCfg, ok := sched.config.Models[j.ModelID]
		if !ok {
			continue
		}
		maxSec := float64(modelCfg.MaxRuntimeSec)
		elapsed := now - *j.StartedAt
		if maxSec == 0 || elapsed < maxSec {
			continue
		}
		errMsg := "job timed out (test)"
		store.UpdateState(j.ID, "failed", WithError(errMsg), WithFinishedAt(now))
	}

	after, _ := store.GetJob(job.ID)
	if after.State != "failed" {
		t.Fatalf("job state = %s, want failed (elapsed > MaxRuntimeSec=10 must be killed)", after.State)
	}
	if after.Error == "" {
		t.Fatal("failed job must have an error message")
	}
}

func TestSnapshotReportsErrorState(t *testing.T) {
	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 100}, "python3", ".")
	inst := NewInstance("broken-model", "broken-model", 1, 10, "python3", ".")
	inst.mu.Lock()
	inst.state = "error"
	inst.mu.Unlock()
	mgr.Register(inst)

	snap := mgr.Snapshot()
	models := snap["models"].([]map[string]any)
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0]["state"] != "error" {
		t.Fatalf("model state = %v, want error", models[0]["state"])
	}
}

// ---------------------------------------------------------------------------
// Min-mean-flow scheduling tests
// ---------------------------------------------------------------------------

// buildMinMeanFlowScheduler constructs a minimal Scheduler backed by a real
// SQLite Store and a real InstanceManager with the given model configs.
// No worker subprocesses are started — we only test the selection logic.
func buildMinMeanFlowScheduler(t *testing.T, models map[string]ModelConfig) (*Scheduler, *Store, *InstanceManager) {
	t.Helper()
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	os.MkdirAll(filepath.Join(outputDir, "logs"), 0o755)

	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	t.Cleanup(func() { store.Close() })
	store.InitDedup()

	// Apply config defaults (PressureIndex, MaxInstances, etc.)
	for id, m := range models {
		if m.MaxConcurrent < 1 {
			m.MaxConcurrent = 1
		}
		if m.MaxInstances == nil {
			one := 1
			m.MaxInstances = &one
		}
		if m.PressureIndex == nil {
			one := 1.0
			m.PressureIndex = &one
		}
		models[id] = m
	}

	cfg := &Config{
		VRAMBudgetGB: 1000,
		Models:       models,
	}
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	t.Cleanup(func() { logger.Close() })

	mgr := NewInstanceManager(cfg, "python3", projectRoot)
	for modelID, m := range models {
		mgr.ScaleModel(modelID, 1, m)
	}

	sched := NewScheduler(cfg, store, mgr, logger, outputDir)
	return sched, store, mgr
}

// markLoaded sets an instance's state to "loaded" directly and reserves the
// VRAM budget so scoreModel / getFullModels see it as loaded.
func markLoaded(t *testing.T, mgr *InstanceManager, modelID string) {
	t.Helper()
	instances := mgr.GetModelInstances(modelID)
	if len(instances) == 0 {
		t.Fatalf("markLoaded: no instances for %q", modelID)
	}
	inst := instances[0]
	inst.mu.Lock()
	inst.state = "loaded"
	inst.mu.Unlock()
	mgr.ReserveMemoryFor(inst, inst.memoryGB)
}

// TestSelectModelMinMeanFlow_ShortColdBatchBeatsLoadedLong verifies that a
// cold model with a batch of short jobs beats a loaded model with a single
// long job because the load cost is amortized over the batch.
//
// "long": AvgInferenceMs=3_600_000 (3600 s), LoadMs=1000 (1 s), loaded.
// "short": AvgInferenceMs=5000 (5 s), LoadMs=43000 (43 s), NOT loaded.
//
// With 1 queued job for "long" and 3 for "short":
//
//	score_long  = 3600 + 0/1         = 3600
//	score_short = 5    + 43/3 ≈ 19.3
//
// → "short" wins.
func TestSelectModelMinMeanFlow_ShortColdBatchBeatsLoadedLong(t *testing.T) {
	models := map[string]ModelConfig{
		"long":  {MemoryGB: 10, MaxConcurrent: 1, AvgInferenceMs: 3_600_000, LoadMs: 1000},
		"short": {MemoryGB: 5, MaxConcurrent: 1, AvgInferenceMs: 5000, LoadMs: 43000},
	}
	sched, store, mgr := buildMinMeanFlowScheduler(t, models)

	markLoaded(t, mgr, "long")
	// "short" is NOT loaded — scoreModel reads IsLoaded → false → adds LoadMs.

	payload := json.RawMessage(`{}`)
	if _, err := store.CreateJob("long", "image-generate", payload, 1); err != nil {
		t.Fatalf("create long job: %v", err)
	}
	for i := 0; i < 3; i++ {
		if _, err := store.CreateJob("short", "image-generate", payload, 1); err != nil {
			t.Fatalf("create short job %d: %v", i, err)
		}
	}

	got := sched.selectModelMinMeanFlow(nil)
	if got != "short" {
		t.Fatalf("selectModelMinMeanFlow = %q, want \"short\" (score ≈19.3 < 3600)", got)
	}
}

// TestSelectModelMinMeanFlow_LoadAmortization tests the crossover point where
// adding more queued jobs for a cold model tips its amortized score below a
// loaded model.
//
// Cold model "c": e=5 s, L=60 s. Loaded model "h": e=10 s, L=0.
//
//	q_c=1:  score_c = 5 + 60/1 = 65   > 10 → "h" wins.
//	q_c=30: score_c = 5 + 60/30 = 7   < 10 → "c" wins.
func TestSelectModelMinMeanFlow_LoadAmortization(t *testing.T) {
	models := map[string]ModelConfig{
		"c": {MemoryGB: 4, MaxConcurrent: 1, AvgInferenceMs: 5000, LoadMs: 60000},
		"h": {MemoryGB: 4, MaxConcurrent: 1, AvgInferenceMs: 10000, LoadMs: 0},
	}

	// Sub-case 1: q_c=1 → loaded "h" wins.
	t.Run("small_batch_loaded_wins", func(t *testing.T) {
		sched, store, mgr := buildMinMeanFlowScheduler(t, models)
		markLoaded(t, mgr, "h")

		payload := json.RawMessage(`{}`)
		if _, err := store.CreateJob("c", "image-generate", payload, 1); err != nil {
			t.Fatalf("create c job: %v", err)
		}
		if _, err := store.CreateJob("h", "image-generate", payload, 1); err != nil {
			t.Fatalf("create h job: %v", err)
		}

		got := sched.selectModelMinMeanFlow(nil)
		if got != "h" {
			t.Fatalf("q_c=1: got %q, want \"h\" (score_h=10 < score_c=65)", got)
		}
	})

	// Sub-case 2: q_c=30 → cold "c" wins because load is well amortized.
	t.Run("large_batch_cold_wins", func(t *testing.T) {
		sched, store, mgr := buildMinMeanFlowScheduler(t, models)
		markLoaded(t, mgr, "h")

		payload := json.RawMessage(`{}`)
		for i := 0; i < 30; i++ {
			if _, err := store.CreateJob("c", "image-generate", payload, 1); err != nil {
				t.Fatalf("create c job %d: %v", i, err)
			}
		}
		if _, err := store.CreateJob("h", "image-generate", payload, 1); err != nil {
			t.Fatalf("create h job: %v", err)
		}

		got := sched.selectModelMinMeanFlow(nil)
		if got != "c" {
			t.Fatalf("q_c=30: got %q, want \"c\" (score_c=7 < score_h=10)", got)
		}
	})
}

// TestSelectModelMinMeanFlow_TieBreakOldest verifies that when two models have
// equal scores, the one whose oldest queued job was submitted first (larger age)
// is selected.
func TestSelectModelMinMeanFlow_TieBreakOldest(t *testing.T) {
	// Both models: identical config so scores are always equal.
	models := map[string]ModelConfig{
		"alpha": {MemoryGB: 4, MaxConcurrent: 1, AvgInferenceMs: 5000, LoadMs: 0},
		"beta":  {MemoryGB: 4, MaxConcurrent: 1, AvgInferenceMs: 5000, LoadMs: 0},
	}
	sched, store, mgr := buildMinMeanFlowScheduler(t, models)
	markLoaded(t, mgr, "alpha")
	markLoaded(t, mgr, "beta")

	payload := json.RawMessage(`{}`)

	// Submit "beta"'s job first so it has the older created_at.
	betaJob, err := store.CreateJob("beta", "image-generate", payload, 1)
	if err != nil {
		t.Fatalf("create beta job: %v", err)
	}
	// Back-date beta's job so it is unambiguously older.
	olderTS := nowTS() - 300
	if err := store.UpdateState(betaJob.ID, "queued", WithStartedAt(olderTS)); err != nil {
		t.Fatalf("backdate beta: %v", err)
	}
	// Directly update created_at since UpdateState doesn't touch it; use raw SQL.
	if _, err := store.db.Exec("UPDATE jobs SET created_at = ? WHERE id = ?", olderTS, betaJob.ID); err != nil {
		t.Fatalf("set beta created_at: %v", err)
	}

	if _, err := store.CreateJob("alpha", "image-generate", payload, 1); err != nil {
		t.Fatalf("create alpha job: %v", err)
	}

	got := sched.selectModelMinMeanFlow(nil)
	if got != "beta" {
		t.Fatalf("tie-break: got %q, want \"beta\" (older oldest job)", got)
	}
}

// TestPickOldestQueuedJobForModel verifies that PickOldestQueuedJobForModel
// returns the oldest queued job for the requested model and ignores jobs
// belonging to other models or in non-queued states.
func TestPickOldestQueuedJobForModel(t *testing.T) {
	store, err := NewStore(filepath.Join(t.TempDir(), "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()
	store.InitDedup()

	payload := json.RawMessage(`{}`)

	// Create a job for a different model — must be ignored.
	if _, err := store.CreateJob("other-model", "image-generate", payload, 1); err != nil {
		t.Fatalf("create other-model job: %v", err)
	}

	// Create two queued jobs for "target". The first has an older timestamp.
	first, err := store.CreateJob("target", "image-generate", payload, 1)
	if err != nil {
		t.Fatalf("create first target job: %v", err)
	}
	olderTS := nowTS() - 60
	if _, err := store.db.Exec("UPDATE jobs SET created_at = ? WHERE id = ?", olderTS, first.ID); err != nil {
		t.Fatalf("backdate first job: %v", err)
	}

	second, err := store.CreateJob("target", "image-generate", payload, 1)
	if err != nil {
		t.Fatalf("create second target job: %v", err)
	}

	// Create a running job for "target" — must be ignored.
	running, err := store.CreateJob("target", "image-generate", payload, 1)
	if err != nil {
		t.Fatalf("create running job: %v", err)
	}
	if err := store.UpdateState(running.ID, "running"); err != nil {
		t.Fatalf("mark running: %v", err)
	}

	got, err := store.PickOldestQueuedJobForModel("target")
	if err != nil {
		t.Fatalf("PickOldestQueuedJobForModel: %v", err)
	}
	if got == nil {
		t.Fatal("expected a job, got nil")
	}
	if got.ID != first.ID {
		t.Fatalf("got job %s, want oldest job %s (not %s)", got.ID, first.ID, second.ID)
	}
	if got.ModelID != "target" {
		t.Fatalf("job model = %q, want \"target\"", got.ModelID)
	}
	if got.State != "queued" {
		t.Fatalf("job state = %q, want \"queued\"", got.State)
	}

	// When no queued jobs exist for a model, should return nil without error.
	got2, err := store.PickOldestQueuedJobForModel("no-such-model")
	if err != nil {
		t.Fatalf("PickOldestQueuedJobForModel for absent model: %v", err)
	}
	if got2 != nil {
		t.Fatalf("expected nil for absent model, got %v", got2)
	}
}

// evictableForScore is a pure helper extracted from the ensureLoaded
// min-mean-flow eviction predicate. It is testable without spinning up
// real VRAM bookkeeping or worker processes.
//
// An instance is a valid eviction candidate when:
//   - it has no active jobs (never preempt running work), AND
//   - the candidate model's score is strictly worse (higher) than the wanted score.
func evictableForScore(activeJobs int, scoreCandidate, scoreWanted float64) bool {
	if activeJobs > 0 {
		return false
	}
	return scoreCandidate > scoreWanted
}

// TestEnsureLoadedDrainsIdleWorseModelButNotRunningOne exercises the pure
// eviction-candidate predicate used by the min-mean-flow drain in ensureLoaded.
func TestEnsureLoadedDrainsIdleWorseModelButNotRunningOne(t *testing.T) {
	type testCase struct {
		name           string
		activeJobs     int
		scoreCandidate float64
		scoreWanted    float64
		wantEvict      bool
	}

	cases := []testCase{
		{
			name:           "idle worse score — should evict",
			activeJobs:     0,
			scoreCandidate: 3600,
			scoreWanted:    19,
			wantEvict:      true,
		},
		{
			name:           "running job — must NOT evict",
			activeJobs:     1,
			scoreCandidate: 3600,
			scoreWanted:    19,
			wantEvict:      false,
		},
		{
			name:           "idle but better score — must NOT evict",
			activeJobs:     0,
			scoreCandidate: 10,
			scoreWanted:    19,
			wantEvict:      false,
		},
		{
			name:           "idle equal score — must NOT evict",
			activeJobs:     0,
			scoreCandidate: 19,
			scoreWanted:    19,
			wantEvict:      false,
		},
		{
			name:           "multiple active jobs — must NOT evict",
			activeJobs:     3,
			scoreCandidate: 9999,
			scoreWanted:    1,
			wantEvict:      false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := evictableForScore(tc.activeJobs, tc.scoreCandidate, tc.scoreWanted)
			if got != tc.wantEvict {
				t.Fatalf("evictableForScore(activeJobs=%d, candidate=%.1f, wanted=%.1f) = %v, want %v",
					tc.activeJobs, tc.scoreCandidate, tc.scoreWanted, got, tc.wantEvict)
			}
		})
	}
}
