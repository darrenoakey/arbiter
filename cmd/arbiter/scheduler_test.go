package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
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
	sched.dispatchJobToInstance(orig, inst)

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
	sched.dispatchJobToInstance(orig, inst)

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
