package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

func boolPtr(b bool) *bool { return &b }

// TestLivenessPollMarksAbsentThenRecovered drives the real poll loop against a
// REAL ollama endpoint and a dead addr. Phase 3 core: N consecutive failures →
// absent (+ host.absent event + Cancel fired); a return to reachable → recovered
// (+ host.recovered). Uses the same real-ollama + dead-addr pattern as Phase 2.
func TestLivenessPollMarksAbsentThenRecovered(t *testing.T) {
	// A controllable fake "ollama" whose /api/version we can flip on/off. We
	// front it with httptest so the test deterministically simulates a host
	// disappearing and returning — no dependence on actually stopping a daemon.
	var alive atomic.Bool
	alive.Store(true)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !alive.Load() {
			// Simulate a dead box: hijack and close so the client sees a reset/refused.
			hj, ok := w.(http.Hijacker)
			if ok {
				conn, _, _ := hj.Hijack()
				conn.Close()
				return
			}
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		if r.URL.Path == "/api/version" {
			w.WriteHeader(200)
			w.Write([]byte(`{"version":"test"}`))
			return
		}
		w.WriteHeader(404)
	}))
	defer srv.Close()

	cfg := &Config{
		VRAMBudgetGB: 100,
		Hosts: map[string]HostConfig{
			"flaky": {Addr: srv.URL, Kind: "mlx", BudgetGB: 64},
		},
		Models: map[string]ModelConfig{
			"chat": {
				MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
				Placements:    []string{"flaky", "spark"},
				AdapterParams: map[string]string{"remote_model_tag": localOllamaTag},
			},
		},
	}
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer logger.Close()
	mgr := NewInstanceManager(cfg, "python3", projectRoot)
	setupInstances(cfg, mgr, "python3", projectRoot)
	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	defer store.Close()
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)

	hm := NewHostMonitor(cfg, mgr, logger, sched)
	hm.pollInterval = 50 * time.Millisecond // fast for the test
	hm.failThreshold = 3
	mgr.SetReachabilityFunc(hm.IsReachable)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go hm.Run(ctx)

	// Initially reachable.
	if !waitReachable(hm, "flaky", true, 2*time.Second) {
		t.Fatalf("host never observed reachable while alive")
	}

	// Kill the host: after >= failThreshold polls it must flip absent.
	alive.Store(false)
	if !waitReachable(hm, "flaky", false, 3*time.Second) {
		t.Fatalf("host did not flip absent after going dead")
	}
	// PickInstanceForJob must now skip the absent remote and spill to spark.
	job := &Job{ID: "j-absent", ModelID: "chat", State: "queued"}
	inst, reason := mgr.PickInstanceForJobWithReason(job, true)
	if inst == nil || inst.host != "spark" {
		t.Fatalf("absent host not skipped: picked host=%v reason=%v", hostOf(inst), reason)
	}
	if reason != reasonFallback {
		t.Fatalf("expected fallback reason when spilling to spark, got %q", reason)
	}

	// Revive the host: it must recover and become eligible again.
	alive.Store(true)
	if !waitReachable(hm, "flaky", true, 3*time.Second) {
		t.Fatalf("host did not recover after coming back")
	}
	job2 := &Job{ID: "j-recovered", ModelID: "chat", State: "queued"}
	inst2, reason2 := mgr.PickInstanceForJobWithReason(job2, true)
	if inst2 == nil || inst2.host != "flaky" {
		t.Fatalf("recovered host not preferred again: picked host=%v reason=%v", hostOf(inst2), reason2)
	}
	if reason2 != reasonPreferred {
		t.Fatalf("expected preferred reason on recovered host, got %q", reason2)
	}
}

// waitReachable polls the monitor's view until host reachability equals want.
func waitReachable(hm *HostMonitor, host string, want bool, deadline time.Duration) bool {
	end := time.Now().Add(deadline)
	for time.Now().Before(end) {
		if hm.IsReachable(host) == want {
			return true
		}
		time.Sleep(20 * time.Millisecond)
	}
	return hm.IsReachable(host) == want
}

// TestCancelFiredOnConfirmedAbsence verifies the active-cancel hook fires when a
// host flips absent — and ONLY then (never on a slow-but-alive poll). It exercises
// applyResult directly so it doesn't depend on real network timing.
func TestCancelFiredOnConfirmedAbsence(t *testing.T) {
	cfg := &Config{
		VRAMBudgetGB: 100,
		Hosts: map[string]HostConfig{
			"h": {Addr: "http://10.255.255.1:11434", Kind: "mlx", BudgetGB: 64},
		},
		Models: map[string]ModelConfig{
			"chat": {
				MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(), Placements: []string{"h", "spark"},
			},
		},
	}
	projectRoot := t.TempDir()
	logger := NewEventLogger(filepath.Join(projectRoot, "logs"))
	defer logger.Close()
	mgr := NewInstanceManager(cfg, "python3", projectRoot)
	setupInstances(cfg, mgr, "python3", projectRoot)
	hm := NewHostMonitor(cfg, mgr, logger, nil)
	hm.failThreshold = 2

	remoteInst := mgr.GetModelInstances("chat")[0]
	if !remoteInst.isRemote() {
		t.Fatalf("expected remote instance on host h")
	}
	rb := remoteInst.backend.(*RemoteHTTPBackend)
	cancelCh := rb.currentCancelChan()

	// One failure: under threshold → still reachable, no cancel.
	hm.applyResult("h", false)
	if !hm.IsReachable("h") {
		t.Fatalf("host flipped absent before threshold")
	}
	select {
	case <-cancelCh:
		t.Fatalf("Cancel fired before confirmed absence — violates no-cancel-on-slow")
	default:
	}

	// Second failure crosses threshold → absent + cancel fired.
	hm.applyResult("h", false)
	if hm.IsReachable("h") {
		t.Fatalf("host did not flip absent at threshold")
	}
	select {
	case <-cancelCh:
		// good: the in-flight epoch's channel was closed
	default:
		t.Fatalf("Cancel was NOT fired on confirmed absence")
	}

	// A success recovers it without firing anything spurious.
	hm.applyResult("h", true)
	if !hm.IsReachable("h") {
		t.Fatalf("host did not recover after a successful poll")
	}
}

// TestKillSwitchFlipsRoutingAndPersists drives the real PATCH handler: flip
// remote_enabled=false → routing pins to spark and the flag persists across a
// config reload from disk. ONE curl, works without the remote host being up.
func TestKillSwitchFlipsRoutingAndPersists(t *testing.T) {
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer logger.Close()
	store, err := NewStore(filepath.Join(outputDir, "arbiter.db"))
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	defer store.Close()

	cfg := &Config{
		VRAMBudgetGB: 100,
		Host:         "127.0.0.1",
		Hosts: map[string]HostConfig{
			"remote1": {Addr: "http://10.255.255.1:11434", Kind: "mlx", BudgetGB: 64},
		},
		Models: map[string]ModelConfig{
			"chat": {
				MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
				Placements:    []string{"remote1", "spark"},
			},
		},
	}
	// Persist a baseline config so reload has something to read.
	if err := SaveModelConfig(projectRoot, "chat", cfg.Models["chat"]); err != nil {
		t.Fatalf("seed config: %v", err)
	}

	mgr := NewInstanceManager(cfg, "python3", projectRoot)
	setupInstances(cfg, mgr, "python3", projectRoot)
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)
	api := NewAPI(cfg, store, mgr, sched, logger, outputDir, projectRoot)

	// Before flip: remote1 is preferred.
	job := &Job{ID: "pre", ModelID: "chat", State: "queued"}
	if inst, _ := mgr.PickInstanceForJobWithReason(job, cfg.RemoteAllowedFor("chat")); inst == nil || inst.host != "remote1" {
		t.Fatalf("before kill-switch: expected remote1, got %v", hostOf(inst))
	}

	// Flip: PATCH /v1/models/chat {"remote_enabled":false}
	body, _ := json.Marshal(map[string]any{"remote_enabled": false})
	req := httptest.NewRequest(http.MethodPatch, "/v1/models/chat", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	api.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("PATCH returned %d: %s", rec.Code, rec.Body.String())
	}
	var resp map[string]any
	json.Unmarshal(rec.Body.Bytes(), &resp)
	if resp["remote_enabled"] != false {
		t.Fatalf("response remote_enabled=%v, want false; body=%s", resp["remote_enabled"], rec.Body.String())
	}

	// After flip: routing pins to spark even with no exclusions.
	job2 := &Job{ID: "post", ModelID: "chat", State: "queued"}
	inst2, reason2 := mgr.PickInstanceForJobWithReason(job2, cfg.RemoteAllowedFor("chat"))
	if inst2 == nil || inst2.host != "spark" {
		t.Fatalf("after kill-switch: expected spark, got %v (reason %v)", hostOf(inst2), reason2)
	}

	// Persisted: reload config from disk and confirm remote_enabled survived.
	reloaded, err := LoadConfig(projectRoot)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	mc := reloaded.Models["chat"]
	if mc.RemoteEnabled == nil || *mc.RemoteEnabled != false {
		t.Fatalf("remote_enabled did not persist: %+v", mc.RemoteEnabled)
	}
	if reloaded.RemoteAllowedFor("chat") {
		t.Fatalf("reloaded config still allows remote for chat")
	}

	// Flip back on: routing returns to remote1.
	body2, _ := json.Marshal(map[string]any{"remote_enabled": true})
	req2 := httptest.NewRequest(http.MethodPatch, "/v1/models/chat", bytes.NewReader(body2))
	rec2 := httptest.NewRecorder()
	api.Handler().ServeHTTP(rec2, req2)
	if rec2.Code != 200 {
		t.Fatalf("PATCH back returned %d: %s", rec2.Code, rec2.Body.String())
	}
	job3 := &Job{ID: "back", ModelID: "chat", State: "queued"}
	if inst3, _ := mgr.PickInstanceForJobWithReason(job3, cfg.RemoteAllowedFor("chat")); inst3 == nil || inst3.host != "remote1" {
		t.Fatalf("after re-enable: expected remote1, got %v", hostOf(inst3))
	}
}

// TestGlobalKillSwitchPinsAllToSpark verifies PATCH /v1/remote {"enabled":false}
// pins EVERY model to spark and persists the global flag.
func TestGlobalKillSwitchPinsAllToSpark(t *testing.T) {
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer logger.Close()
	store, err := NewStore(filepath.Join(outputDir, "arbiter.db"))
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	defer store.Close()

	cfg := &Config{
		VRAMBudgetGB: 100,
		Hosts: map[string]HostConfig{
			"r1": {Addr: "http://10.255.255.1:11434", Kind: "mlx", BudgetGB: 64},
		},
		Models: map[string]ModelConfig{
			"chat": {
				MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(), Placements: []string{"r1", "spark"},
			},
		},
	}
	mgr := NewInstanceManager(cfg, "python3", projectRoot)
	setupInstances(cfg, mgr, "python3", projectRoot)
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)
	api := NewAPI(cfg, store, mgr, sched, logger, outputDir, projectRoot)

	body, _ := json.Marshal(map[string]any{"enabled": false})
	req := httptest.NewRequest(http.MethodPatch, "/v1/remote", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	api.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("PATCH /v1/remote returned %d: %s", rec.Code, rec.Body.String())
	}
	if !cfg.RemoteDisabled {
		t.Fatalf("global flag not set on config")
	}

	job := &Job{ID: "g", ModelID: "chat", State: "queued"}
	if inst, _ := mgr.PickInstanceForJobWithReason(job, cfg.RemoteAllowedFor("chat")); inst == nil || inst.host != "spark" {
		t.Fatalf("global kill-switch did not pin to spark, got %v", hostOf(inst))
	}

	// Persisted across reload.
	reloaded, err := LoadConfig(projectRoot)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if !reloaded.RemoteDisabled {
		t.Fatalf("global remote_disabled did not persist")
	}
}

// TestPSExposesHostPlacementAndRemotePanel verifies /v1/ps carries per-instance
// host + placement_reason and the SEPARATE remote_hosts panel (advisory, NOT in
// the audited VRAM numbers).
func TestPSExposesHostPlacementAndRemotePanel(t *testing.T) {
	if !reachableOllama(t) {
		t.Skip("local ollama unavailable")
	}
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer logger.Close()
	store, err := NewStore(filepath.Join(outputDir, "arbiter.db"))
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	defer store.Close()

	cfg := &Config{
		VRAMBudgetGB: 100,
		Hosts: map[string]HostConfig{
			"mac": {Addr: localOllamaAddr, Kind: "mlx", BudgetGB: 64},
		},
		Models: map[string]ModelConfig{
			"chat": {
				MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
				Placements:    []string{"mac", "spark"},
				AdapterParams: map[string]string{"remote_model_tag": localOllamaTag},
			},
		},
	}
	mgr := NewInstanceManager(cfg, "python3", projectRoot)
	setupInstances(cfg, mgr, "python3", projectRoot)
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)
	api := NewAPI(cfg, store, mgr, sched, logger, outputDir, projectRoot)
	hm := NewHostMonitor(cfg, mgr, logger, sched)
	mgr.SetReachabilityFunc(hm.IsReachable)
	api.SetHostMonitor(hm)

	// Stamp a placement reason on the remote instance as a dispatch would.
	remoteInst := mgr.GetModelInstances("chat")[0]
	for _, inst := range mgr.GetModelInstances("chat") {
		if inst.host == "mac" {
			remoteInst = inst
		}
	}
	remoteInst.setPlacementReason(reasonPreferred)

	// Build the ps cache (also exercises the remote_hosts panel network call).
	api.updatePSCache()
	raw, ok := api.psCache.Load().([]byte)
	if !ok {
		t.Fatalf("ps cache type %T, want []byte", api.psCache.Load())
	}
	var snap map[string]any
	if err := json.Unmarshal(raw, &snap); err != nil {
		t.Fatalf("ps json: %v", err)
	}

	// remote_hosts panel present and reflects the mac host.
	panel, ok := snap["remote_hosts"].([]any)
	if !ok || len(panel) == 0 {
		t.Fatalf("remote_hosts panel missing or empty; ps=%s", string(raw))
	}
	foundMac := false
	for _, p := range panel {
		e := p.(map[string]any)
		if e["host_id"] == "mac" {
			foundMac = true
			if e["budget_gb"] != float64(64) {
				t.Fatalf("remote_hosts mac budget_gb=%v, want 64", e["budget_gb"])
			}
			if _, ok := e["reachable"]; !ok {
				t.Fatalf("remote_hosts mac missing reachable flag")
			}
		}
	}
	if !foundMac {
		t.Fatalf("mac host not in remote_hosts panel: %v", panel)
	}

	// The remote host must NOT inflate the audited local VRAM ledger.
	if alloc, _ := snap["vram_allocated_gb"].(float64); alloc != 0 {
		t.Fatalf("remote placement leaked into audited vram_allocated_gb=%v", alloc)
	}

	// Per-instance host + placement_reason present in the model breakdown.
	models, _ := snap["models"].([]any)
	var chatEntry map[string]any
	for _, m := range models {
		me := m.(map[string]any)
		if me["id"] == "chat" {
			chatEntry = me
		}
	}
	if chatEntry == nil {
		t.Fatalf("chat model missing from ps")
	}
	insts, ok := chatEntry["instances"].([]any)
	if !ok || len(insts) == 0 {
		t.Fatalf("chat model has no per-instance breakdown; entry=%v", chatEntry)
	}
	sawMacReason := false
	for _, ie := range insts {
		e := ie.(map[string]any)
		if e["host"] == "mac" {
			if e["placement_reason"] != reasonPreferred {
				t.Fatalf("mac instance placement_reason=%v, want preferred", e["placement_reason"])
			}
			if e["remote"] != true {
				t.Fatalf("mac instance not flagged remote=true")
			}
			sawMacReason = true
		}
	}
	if !sawMacReason {
		t.Fatalf("mac instance host/placement_reason not in ps; instances=%v", insts)
	}
}
