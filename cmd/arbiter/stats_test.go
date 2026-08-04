package main

import (
	"math"
	"testing"
	"time"
)

// seedStatsJobs inserts a fixed mix of jobs across two models and several
// states, including completed jobs with and without started_at, so the
// grouped aggregates have non-trivial averages to reproduce.
func seedStatsJobs(t *testing.T, s *Store) {
	t.Helper()
	rows := []struct {
		model              string
		state              string
		created            float64
		started, finished  float64
		hasStarted, hasFin bool
	}{
		// model-a: 3 completed (2 with started_at), 1 queued, 1 failed
		{"model-a", "completed", 100, 110, 130, true, true},
		{"model-a", "completed", 200, 205, 245, true, true},
		{"model-a", "completed", 300, 0, 360, false, true}, // no started_at
		{"model-a", "queued", 400, 0, 0, false, false},
		{"model-a", "failed", 500, 510, 515, true, true},
		{"model-a", "scheduled", 520, 525, 0, true, false}, // active
		// model-b: 2 completed, 2 queued, 1 cancelled, 1 running
		{"model-b", "completed", 600, 610, 650, true, true},
		{"model-b", "completed", 700, 720, 760, true, true},
		{"model-b", "queued", 800, 0, 0, false, false},
		{"model-b", "queued", 810, 0, 0, false, false},
		{"model-b", "cancelled", 900, 0, 905, false, true},
		{"model-b", "running", 950, 955, 0, true, false}, // active
	}
	for i, r := range rows {
		var started, finished any
		if r.hasStarted {
			started = r.started
		}
		if r.hasFin {
			finished = r.finished
		}
		_, err := s.db.Exec(
			"INSERT INTO jobs (id, model_id, job_type, state, priority, created_at, started_at, finished_at) VALUES (?,?,?,?,?,?,?,?)",
			genID(), r.model, "test", r.state, float64(i), r.created, started, finished,
		)
		if err != nil {
			t.Fatalf("insert job %d: %v", i, err)
		}
	}
}

func approxEq(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// TestActivePendingByModelMatchesPerModel verifies the single-scan active-state
// count reproduces the old per-model (queued+scheduled+running) computation
// that the scheduler ran every sweep. Models with zero active jobs are absent
// from the map, which callers must (and do) treat as zero.
func TestActivePendingByModelMatchesPerModel(t *testing.T) {
	store, _ := newTestStore(t)
	seedStatsJobs(t, store)

	got, err := store.ActivePendingByModel()
	if err != nil {
		t.Fatalf("ActivePendingByModel: %v", err)
	}

	for _, model := range []string{"model-a", "model-b"} {
		counts, err := store.CountByState(model)
		if err != nil {
			t.Fatalf("CountByState(%s): %v", model, err)
		}
		want := counts["queued"] + counts["scheduled"] + counts["running"]
		if got[model] != want {
			t.Errorf("%s active pending=%d want %d", model, got[model], want)
		}
	}

	// A model with no active jobs must be absent (read as 0 by callers).
	if _, ok := got["nonexistent-model"]; ok {
		t.Errorf("expected absent key for model with no active jobs")
	}
}

// TestCountByStateGroupedMatchesPerModel verifies the single-scan grouped count
// query returns exactly what the per-model CountByState calls return — the
// substitution the /v1/ps cache relies on.
func TestCountByStateGroupedMatchesPerModel(t *testing.T) {
	store, _ := newTestStore(t)
	seedStatsJobs(t, store)

	perModel, global, err := store.CountByStateGrouped()
	if err != nil {
		t.Fatalf("CountByStateGrouped: %v", err)
	}

	// Global must match CountByState("").
	wantGlobal, err := store.CountByState("")
	if err != nil {
		t.Fatalf("CountByState(\"\"): %v", err)
	}
	if len(global) != len(wantGlobal) {
		t.Fatalf("global states: got %v want %v", global, wantGlobal)
	}
	for state, n := range wantGlobal {
		if global[state] != n {
			t.Errorf("global[%s]=%d want %d", state, global[state], n)
		}
	}

	// Per model must match CountByState(model).
	for _, model := range []string{"model-a", "model-b"} {
		want, err := store.CountByState(model)
		if err != nil {
			t.Fatalf("CountByState(%s): %v", model, err)
		}
		got := perModel[model]
		if len(got) != len(want) {
			t.Fatalf("%s states: got %v want %v", model, got, want)
		}
		for state, n := range want {
			if got[state] != n {
				t.Errorf("%s[%s]=%d want %d", model, state, got[state], n)
			}
		}
	}
}

// TestCompletedJobStatsGroupedMatchesPerModel verifies the single-scan grouped
// completed-stats query reproduces the per-model and global averages of the
// original CompletedJobStats exactly.
func TestCompletedJobStatsGroupedMatchesPerModel(t *testing.T) {
	store, _ := newTestStore(t)
	seedStatsJobs(t, store)

	perModel, global, err := store.CompletedJobStatsGrouped()
	if err != nil {
		t.Fatalf("CompletedJobStatsGrouped: %v", err)
	}

	wantCount, wantTotal, wantExec, err := store.CompletedJobStats("")
	if err != nil {
		t.Fatalf("CompletedJobStats(\"\"): %v", err)
	}
	if global.Count != wantCount || !approxEq(global.AvgTotal, wantTotal) || !approxEq(global.AvgExec, wantExec) {
		t.Errorf("global stats: got {%d %g %g} want {%d %g %g}",
			global.Count, global.AvgTotal, global.AvgExec, wantCount, wantTotal, wantExec)
	}

	for _, model := range []string{"model-a", "model-b"} {
		c, tot, exec, err := store.CompletedJobStats(model)
		if err != nil {
			t.Fatalf("CompletedJobStats(%s): %v", model, err)
		}
		got := perModel[model]
		if got.Count != c || !approxEq(got.AvgTotal, tot) || !approxEq(got.AvgExec, exec) {
			t.Errorf("%s stats: got {%d %g %g} want {%d %g %g}",
				model, got.Count, got.AvgTotal, got.AvgExec, c, tot, exec)
		}
	}
}

// TestStoreAllowsConcurrentReaders documents the WAL pool size that keeps
// /v1/jobs lookups free while CompletedJobStatsGrouped scans a large DB.
// MaxOpenConns(1) is the historical starvation bug: one long stats query held
// the only handle and every job lookup blocked behind it.
func TestStoreAllowsConcurrentReaders(t *testing.T) {
	store, _ := newTestStore(t)
	stats := store.db.Stats()
	if stats.MaxOpenConnections != 8 {
		t.Fatalf("MaxOpenConnections=%d, want 8 so stats scans cannot starve job lookups", stats.MaxOpenConnections)
	}
}

// TestRefreshStatsIsNonBlocking verifies updatePSCache returns immediately
// even when aggregates are stale, and that the background pass still fills
// the cache. A blocking refresh reintroduced the multi-minute /v1/ps hang.
func TestRefreshStatsIsNonBlocking(t *testing.T) {
	api, cleanup := newTestAPI(t)
	defer cleanup()
	seedStatsJobs(t, api.store)

	api.statsMu.Lock()
	api.statsAt = time.Time{}
	api.statsRefreshing = false
	api.statsMu.Unlock()

	started := time.Now()
	api.updatePSCache()
	if elapsed := time.Since(started); elapsed > 200*time.Millisecond {
		t.Fatalf("updatePSCache blocked for %s; refreshStats must be asynchronous", elapsed)
	}

	deadline := time.Now().Add(2 * time.Second)
	for {
		api.statsMu.Lock()
		ready := !api.statsAt.IsZero() && !api.statsRefreshing && api.statsGlobal.Count > 0
		count := api.statsGlobal.Count
		api.statsMu.Unlock()
		if ready {
			if count < 5 {
				t.Fatalf("statsGlobal.Count=%d, want seeded completed jobs", count)
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("async stats refresh never populated the cache")
		}
		time.Sleep(5 * time.Millisecond)
	}
}
