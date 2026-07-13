package main

import (
	"encoding/json"
	"sync/atomic"
	"testing"
	"time"
)

// setActiveJobs forces an instance's active-job counter for tests that need a
// model to look busy without dispatching real work.
func setActiveJobs(inst *Instance, n int32) {
	atomic.StoreInt32(&inst.activeJobs, n)
}

// buildPSCacheModels drives updatePSCache and returns the /v1/ps model entries
// keyed by id, so a test can assert the in_progress panel.
func buildPSCacheModels(t *testing.T, api *API) map[string]map[string]any {
	t.Helper()
	// Force the throttled aggregates to recompute this call.
	api.statsMu.Lock()
	api.statsAt = time.Time{}
	api.statsMu.Unlock()
	api.updatePSCache()

	raw, _ := api.psCache.Load().([]byte)
	var snap map[string]any
	if err := json.Unmarshal(raw, &snap); err != nil {
		t.Fatalf("unmarshal ps cache: %v", err)
	}
	out := make(map[string]map[string]any)
	models, _ := snap["models"].([]any)
	for _, m := range models {
		mm, _ := m.(map[string]any)
		id, _ := mm["id"].(string)
		out[id] = mm
	}
	return out
}

// TestInProgressPanelOnlyForActiveModels verifies the in_progress block appears
// ONLY for a model with active in-flight work, carries a correct ETA from the
// persisted rolling average, and counts done-since-load correctly — and that an
// idle (loaded but no active jobs) model gets no in_progress block at all.
func TestInProgressPanelOnlyForActiveModels(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	one := 1
	api.config.Models = map[string]ModelConfig{
		"busy": {MaxConcurrent: 2, MaxInstances: &one, AvgInferenceMs: 1000},
		"idle": {MaxConcurrent: 1, MaxInstances: &one, AvgInferenceMs: 1000},
	}

	// Two loaded models. "busy" is actively serving; "idle" is loaded but has no
	// active work.
	busy := NewInstance("busy", "busy#0", 2, 4, "python3", ".")
	api.mgr.Register(busy)
	busy.setState("loaded")
	loadEpoch := float64(busy.LoadedAt().UnixNano()) / 1e9

	idle := NewInstance("idle", "idle#0", 1, 4, "python3", ".")
	api.mgr.Register(idle)
	idle.setState("loaded")

	// "busy": 2 active jobs (running) + 23 queued => 25 outstanding. 3 already
	// completed since load, plus one completed BEFORE load that must not count.
	seed := []struct {
		state    string
		finished any
	}{
		{"completed", loadEpoch - 100}, // before load — excluded from done_since_load
		{"completed", loadEpoch + 1},
		{"completed", loadEpoch + 2},
		{"completed", loadEpoch + 3},
	}
	for i, r := range seed {
		if _, err := api.store.db.Exec(
			"INSERT INTO jobs (id, model_id, job_type, state, priority, created_at, finished_at) VALUES (?,?,?,?,?,?,?)",
			genID(), "busy", "test", r.state, float64(i), loadEpoch-200, r.finished,
		); err != nil {
			t.Fatalf("seed completed %d: %v", i, err)
		}
	}
	for i := 0; i < 23; i++ {
		if _, err := api.store.CreateJob("busy", "test", json.RawMessage(`{}`), 1); err != nil {
			t.Fatalf("seed queued %d: %v", i, err)
		}
	}
	setActiveJobs(busy, 2)

	// Persisted rolling average = 6 minutes/action (the ledger's worked example).
	for i := 0; i < 3; i++ {
		if err := api.store.RecordActionDuration("busy", 360); err != nil {
			t.Fatalf("record duration: %v", err)
		}
	}

	models := buildPSCacheModels(t, api)

	// --- busy: in_progress present and correct ---
	busyEntry := models["busy"]
	if busyEntry == nil {
		t.Fatalf("busy model missing from ps cache")
	}
	ip, ok := busyEntry["in_progress"].(map[string]any)
	if !ok {
		t.Fatalf("busy model has no in_progress panel: %v", busyEntry)
	}
	if got := ip["done_since_load"].(float64); got != 3 {
		t.Errorf("done_since_load = %v want 3", got)
	}
	if got := ip["total_since_load"].(float64); got != 28 { // 3 done + 25 outstanding
		t.Errorf("total_since_load = %v want 28", got)
	}
	if got := ip["avg_action_seconds"].(float64); !approxEq(got, 360) {
		t.Errorf("avg_action_seconds = %v want 360", got)
	}
	// 25 outstanding * 360s = 9000s = 150 minutes.
	if got := ip["eta_seconds"].(float64); !approxEq(got, 9000) {
		t.Errorf("eta_seconds = %v want 9000 (150 min)", got)
	}

	// --- idle: no in_progress panel ---
	idleEntry := models["idle"]
	if idleEntry == nil {
		t.Fatalf("idle model missing from ps cache")
	}
	if _, ok := idleEntry["in_progress"]; ok {
		t.Errorf("idle model should have no in_progress panel: %v", idleEntry)
	}
}

// TestInProgressEtaFallsBackToLifetimeAvg proves an ETA appears immediately —
// before model_stats has any sample — using the all-time execution average.
func TestInProgressEtaFallsBackToLifetimeAvg(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()

	one := 1
	api.config.Models = map[string]ModelConfig{
		"fresh": {MaxConcurrent: 1, MaxInstances: &one, AvgInferenceMs: 1000},
	}
	inst := NewInstance("fresh", "fresh#0", 1, 4, "python3", ".")
	api.mgr.Register(inst)
	inst.setState("loaded")

	// One historical completed job with a 40s execution time (started->finished),
	// and no model_stats row yet. Plus 4 queued behind the active one.
	if _, err := api.store.db.Exec(
		"INSERT INTO jobs (id, model_id, job_type, state, priority, created_at, started_at, finished_at) VALUES (?,?,?,?,?,?,?,?)",
		genID(), "fresh", "test", "completed", 0.0, 100.0, 110.0, 150.0,
	); err != nil {
		t.Fatalf("seed completed: %v", err)
	}
	for i := 0; i < 4; i++ {
		if _, err := api.store.CreateJob("fresh", "test", json.RawMessage(`{}`), 1); err != nil {
			t.Fatalf("seed queued: %v", err)
		}
	}
	setActiveJobs(inst, 1)

	models := buildPSCacheModels(t, api)
	ip, ok := models["fresh"]["in_progress"].(map[string]any)
	if !ok {
		t.Fatalf("fresh model has no in_progress panel")
	}
	// avg falls back to lifetime exec avg (40s); 5 outstanding * 40 = 200s.
	if got := ip["avg_action_seconds"].(float64); !approxEq(got, 40) {
		t.Errorf("avg_action_seconds fallback = %v want 40", got)
	}
	if got := ip["eta_seconds"].(float64); !approxEq(got, 200) {
		t.Errorf("eta_seconds = %v want 200", got)
	}
}
