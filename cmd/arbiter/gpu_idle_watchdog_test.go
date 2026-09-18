package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func newTestIdleWatchdog(t *testing.T) (*GPUIdleWatchdog, *Store, *InstanceManager, *time.Time) {
	t.Helper()
	dir := t.TempDir()
	store, err := NewStore(filepath.Join(dir, "t.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(store.Close)
	cfg := &Config{
		VRAMBudgetGB: 90,
		GPUIdle: GPUIdleConfig{
			IdleSeconds:     180,
			LowUtilPct:      5,
			HighIntervalSec: 30,
			LowIntervalSec:  5,
			CooldownSec:     120,
			Agentd3URL:      "http://127.0.0.1:9/investigate",
		},
		Models: map[string]ModelConfig{"ltx2": {MemoryGB: 40, MaxConcurrent: 1}},
	}
	mgr := NewInstanceManager(cfg, "python3", dir)
	logger := NewEventLogger(filepath.Join(dir, "logs"))
	t.Cleanup(logger.Close)
	w := NewGPUIdleWatchdog(cfg, mgr, store, logger, dir)
	now := time.Unix(1_700_000_000, 0).UTC()
	clock := &now
	w.now = func() time.Time { return *clock }
	w.gpuStatus = func() string { return "0 %, 9.06 W, 50 C" }
	w.startInvestigation = func(GPUIdleKillReport) error { return nil }
	w.killInstance = func(*Instance) {}
	w.gpuUtil = func() int { return 0 }
	return w, store, mgr, clock
}

func addRunningLocal(t *testing.T, mgr *InstanceManager, store *Store, model, instID string) (*Instance, *Job) {
	t.Helper()
	inst := NewInstance(model, instID, 1, 40, "python3", t.TempDir())
	inst.state = "loaded"
	atomic.StoreInt32(&inst.activeJobs, 1)
	mgr.Register(inst)
	job, err := store.CreateJobWithRequestedModel(model, "video-generate", json.RawMessage(`{"prompt":"hang"}`), 0, "", WithSource(&JobSource{Who: "waggler", Why: "hang williams video"}))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.UpdateState(job.ID, "running", WithStartedAt(nowTS())); err != nil {
		t.Fatal(err)
	}
	job, err = store.GetJob(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	inst.pendingMu.Lock()
	inst.pending[job.ID] = make(chan json.RawMessage, 1)
	inst.pendingMu.Unlock()
	return inst, job
}

func TestGPUIdleWatchdogHighUtilUsesLongIntervalAndDoesNotKill(t *testing.T) {
	w, store, mgr, _ := newTestIdleWatchdog(t)
	addRunningLocal(t, mgr, store, "ltx2", "ltx2#1")
	w.gpuUtil = func() int { return 80 }
	var killed []string
	w.killInstance = func(inst *Instance) { killed = append(killed, inst.InstanceID) }
	if d := w.tick(); d != 30*time.Second {
		t.Fatalf("interval = %s, want 30s", d)
	}
	if len(killed) != 0 {
		t.Fatalf("killed on busy GPU: %v", killed)
	}
}

func TestGPUIdleWatchdogLowUtilWithoutRunningJobsDoesNotKill(t *testing.T) {
	w, _, mgr, _ := newTestIdleWatchdog(t)
	inst := NewInstance("ltx2", "ltx2#idle", 1, 40, "python3", t.TempDir())
	inst.state = "loaded"
	mgr.Register(inst)
	w.gpuUtil = func() int { return 0 }
	var killed []string
	w.killInstance = func(inst *Instance) { killed = append(killed, inst.InstanceID) }
	if d := w.tick(); d != 5*time.Second {
		t.Fatalf("interval = %s, want 5s when GPU is low", d)
	}
	if len(killed) != 0 {
		t.Fatalf("killed idle loaded model with no running jobs: %v", killed)
	}
}

func TestGPUIdleWatchdogUnavailableSMIDoesNotCountAsIdle(t *testing.T) {
	w, store, mgr, _ := newTestIdleWatchdog(t)
	addRunningLocal(t, mgr, store, "ltx2", "ltx2#1")
	w.gpuUtil = func() int { return -1 }
	var killed []string
	w.killInstance = func(inst *Instance) { killed = append(killed, inst.InstanceID) }
	if d := w.tick(); d != 30*time.Second {
		t.Fatalf("interval = %s, want 30s on smi failure", d)
	}
	if len(killed) != 0 {
		t.Fatalf("killed when nvidia-smi failed: %v", killed)
	}
}

func TestGPUIdleWatchdogKillsAfterContinuousIdleAndDispatchesInvestigation(t *testing.T) {
	w, store, mgr, clock := newTestIdleWatchdog(t)
	inst, job := addRunningLocal(t, mgr, store, "ltx2", "ltx2#1")
	var killed []string
	w.killInstance = func(in *Instance) { killed = append(killed, in.InstanceID) }
	got := make(chan GPUIdleKillReport, 1)
	w.startInvestigation = func(r GPUIdleKillReport) error {
		got <- r
		return nil
	}

	if d := w.tick(); d != 5*time.Second {
		t.Fatalf("first low tick interval = %s", d)
	}
	*clock = clock.Add(179 * time.Second)
	w.tick()
	if len(killed) != 0 {
		t.Fatalf("killed before 180s: %v", killed)
	}
	*clock = clock.Add(2 * time.Second)
	w.tick()
	if len(killed) != 1 || killed[0] != inst.InstanceID {
		t.Fatalf("killed = %v, want [%s]", killed, inst.InstanceID)
	}

	select {
	case report := <-got:
		if len(report.Jobs) != 1 || report.Jobs[0].JobID != job.ID {
			t.Fatalf("report jobs = %+v, want %s", report.Jobs, job.ID)
		}
		if report.Jobs[0].Who != "waggler" || report.Jobs[0].Why != "hang williams video" {
			t.Fatalf("job source = %+v", report.Jobs[0])
		}
		if !strings.Contains(report.Prompt, job.ID) {
			t.Fatalf("prompt missing job id: %s", report.Prompt)
		}
		if !strings.Contains(report.Prompt, "/mnt/arbiter-store/output/logs/") {
			t.Fatalf("prompt missing event log path: %s", report.Prompt)
		}
		if !strings.Contains(report.Prompt, "/home/darren/local/blackbox/") {
			t.Fatalf("prompt missing blackbox path")
		}
		if report.Conversation.Origin["kind"] != "service" {
			t.Fatalf("origin = %+v", report.Conversation.Origin)
		}
		if report.BundlePath == "" {
			t.Fatal("bundle path empty")
		}
		if _, err := os.Stat(report.BundlePath); err != nil {
			t.Fatalf("bundle missing: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("investigation was not dispatched")
	}

	failed, err := store.GetJob(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if failed.State != "failed" {
		t.Fatalf("job state = %s, want failed", failed.State)
	}
	if !strings.Contains(failed.Error, "gpu-idle-watchdog") {
		t.Fatalf("error = %q", failed.Error)
	}
}

func TestGPUIdleWatchdogSkipsLoadingAndRemoteInstances(t *testing.T) {
	w, store, mgr, clock := newTestIdleWatchdog(t)
	loading, _ := addRunningLocal(t, mgr, store, "ltx2", "ltx2#loading")
	loading.state = "loading"
	remote, _ := addRunningLocal(t, mgr, store, "ltx2", "ltx2#remote")
	remote.host = "boringstack"
	var killed []string
	w.killInstance = func(inst *Instance) { killed = append(killed, inst.InstanceID) }
	w.tick()
	*clock = clock.Add(3 * time.Minute)
	w.tick()
	if len(killed) != 0 {
		t.Fatalf("killed loading/remote: %v", killed)
	}
}

func TestGPUIdleWatchdogBusySampleResetsIdleWindow(t *testing.T) {
	w, store, mgr, clock := newTestIdleWatchdog(t)
	addRunningLocal(t, mgr, store, "ltx2", "ltx2#1")
	var killed []string
	w.killInstance = func(inst *Instance) { killed = append(killed, inst.InstanceID) }
	util := 0
	w.gpuUtil = func() int { return util }
	w.tick()
	*clock = clock.Add(100 * time.Second)
	util = 40
	if d := w.tick(); d != 30*time.Second {
		t.Fatalf("busy interval = %s", d)
	}
	util = 0
	w.tick()
	*clock = clock.Add(179 * time.Second)
	w.tick()
	if len(killed) != 0 {
		t.Fatalf("killed after reset window: %v", killed)
	}
}

func TestGPUIdleWatchdogCooldownPreventsImmediateRekill(t *testing.T) {
	w, store, mgr, clock := newTestIdleWatchdog(t)
	w.cfg.GPUIdle.IdleSeconds = 3
	w.cfg.GPUIdle.CooldownSec = 60
	addRunningLocal(t, mgr, store, "ltx2", "ltx2#1")
	var killed []string
	w.killInstance = func(inst *Instance) { killed = append(killed, inst.InstanceID) }
	w.startInvestigation = func(GPUIdleKillReport) error { return nil }
	w.tick()
	*clock = clock.Add(3 * time.Second)
	w.tick()
	if len(killed) != 1 {
		t.Fatalf("first kill = %v", killed)
	}
	atomic.StoreInt32(&mgr.Get("ltx2#1").activeJobs, 1)
	mgr.Get("ltx2#1").state = "loaded"
	w.tick()
	*clock = clock.Add(3 * time.Second)
	w.tick()
	if len(killed) != 1 {
		t.Fatalf("cooldown should block second kill, got %v", killed)
	}
	*clock = clock.Add(60 * time.Second)
	w.tick()
	if len(killed) != 2 {
		t.Fatalf("after cooldown want 2 kills, got %v", killed)
	}
}

func TestPostInvestigationPostsJSON(t *testing.T) {
	var got []byte
	srv := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost || req.URL.Path != "/investigate" {
			http.NotFound(rw, req)
			return
		}
		body, _ := io.ReadAll(req.Body)
		got = body
		rw.WriteHeader(http.StatusAccepted)
	}))
	t.Cleanup(srv.Close)

	w, _, _, _ := newTestIdleWatchdog(t)
	w.cfg.GPUIdle.Agentd3URL = srv.URL + "/investigate"
	report := GPUIdleKillReport{
		GPUUtilPct: 0,
		Jobs:       []gpuIdleJobDetail{{JobID: "abc123", ModelID: "ltx2"}},
		Prompt:     "investigate abc123",
	}
	if err := w.postInvestigation(report); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(got), "abc123") {
		t.Fatalf("posted body missing job id: %s", got)
	}
}

func TestGPUIdleConfigZeroValuesUseDefaults(t *testing.T) {
	var c GPUIdleConfig
	if c.idleFor() != 180*time.Second || c.lowUtilPct() != 5 {
		t.Fatalf("defaults idle=%s util=%d", c.idleFor(), c.lowUtilPct())
	}
	if c.highInterval() != 30*time.Second || c.lowInterval() != 5*time.Second {
		t.Fatalf("intervals high=%s low=%s", c.highInterval(), c.lowInterval())
	}
	if c.agentd3URL() != gpuIdleDefaultAgentd3URL {
		t.Fatalf("url = %s", c.agentd3URL())
	}
}
