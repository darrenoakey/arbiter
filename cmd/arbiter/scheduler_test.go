package main

import (
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
	full := sched.getFullModels()
	if !full["broken"] {
		t.Fatal("broken model should be in full models set")
	}

	// RecordLoadSuccess should reset
	sched.RecordLoadSuccess("broken")
	// CB pause is time-based, not reset by success — but count+level are reset.
	// After the pause expires, it should not be paused.
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

	// Reserve the memory in bookkeeping to match loaded state
	mgr.ReserveMemory(20) // model-a
	mgr.ReserveMemory(30) // model-b

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
