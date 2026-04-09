package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

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
	mgr := NewInstanceManager(100, "python3", projectRoot)
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
	mgr := NewInstanceManager(100, "python3", projectRoot)
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
	mgr := NewInstanceManager(100, "python3", projectRoot)
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

	// Create a queued job that should get cancelled on CB activation
	payload := json.RawMessage(`{"test":true}`)
	job, _ := store.CreateJob("broken", "image-generate", payload, 1)

	// 3rd failure triggers the circuit-breaker
	sched.RecordLoadFailure("broken")
	// Give the async cancellation goroutine a moment
	time.Sleep(50 * time.Millisecond)

	if paused, _ := sched.IsModelLoadPaused("broken"); !paused {
		t.Fatal("model should be paused after 3 failures")
	}

	// Queued job should be cancelled
	jobAfter, _ := store.GetJob(job.ID)
	if jobAfter.State != "cancelled" {
		t.Fatalf("queued job state = %s, want cancelled", jobAfter.State)
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

func TestEvictForGBWithQueueInfoPrefersNoQueue(t *testing.T) {
	mgr := NewInstanceManager(100, "python3", ".")

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

func TestSnapshotReportsErrorState(t *testing.T) {
	mgr := NewInstanceManager(100, "python3", ".")
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
