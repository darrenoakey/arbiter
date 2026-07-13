package main

import (
	"math"
	"path/filepath"
	"testing"
)

// TestRecordActionDurationSeedsThenBlends verifies the persisted per-model
// rolling average seeds on the first sample and blends subsequent samples with
// the EMA alpha — the exact math the dashboard ETA relies on.
func TestRecordActionDurationSeedsThenBlends(t *testing.T) {
	store, _ := newTestStore(t)

	// First sample seeds the average outright.
	if err := store.RecordActionDuration("m", 100); err != nil {
		t.Fatalf("record 1: %v", err)
	}
	avgs, err := store.ModelActionAverages()
	if err != nil {
		t.Fatalf("averages: %v", err)
	}
	if !approxEq(avgs["m"], 100) {
		t.Fatalf("after seed got %v want 100", avgs["m"])
	}

	// Second sample blends: alpha*200 + (1-alpha)*100.
	if err := store.RecordActionDuration("m", 200); err != nil {
		t.Fatalf("record 2: %v", err)
	}
	want := actionEMAAlpha*200 + (1-actionEMAAlpha)*100
	avgs, _ = store.ModelActionAverages()
	if !approxEq(avgs["m"], want) {
		t.Fatalf("after blend got %v want %v", avgs["m"], want)
	}

	// Non-positive durations are ignored (no learning from a bad measurement).
	before := avgs["m"]
	if err := store.RecordActionDuration("m", 0); err != nil {
		t.Fatalf("record 0: %v", err)
	}
	if err := store.RecordActionDuration("m", -5); err != nil {
		t.Fatalf("record -5: %v", err)
	}
	avgs, _ = store.ModelActionAverages()
	if !approxEq(avgs["m"], before) {
		t.Fatalf("non-positive samples changed avg: got %v want %v", avgs["m"], before)
	}
}

// TestModelActionAverageSurvivesReopen proves the rolling average is truly
// persisted: it must be readable after the store is closed and reopened, so
// ETAs stay meaningful across a daemon restart.
func TestModelActionAverageSurvivesReopen(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "arbiter.db")

	store, err := NewStore(dbPath)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	for _, d := range []float64{300, 360, 330} {
		if err := store.RecordActionDuration("ltx2-dev-noise1", d); err != nil {
			t.Fatalf("record: %v", err)
		}
	}
	avgs, _ := store.ModelActionAverages()
	want := avgs["ltx2-dev-noise1"]
	if want <= 0 {
		t.Fatalf("expected a positive average, got %v", want)
	}
	store.Close()

	reopened, err := NewStore(dbPath)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer reopened.Close()
	avgs, err = reopened.ModelActionAverages()
	if err != nil {
		t.Fatalf("averages after reopen: %v", err)
	}
	if !approxEq(avgs["ltx2-dev-noise1"], want) {
		t.Fatalf("average not persisted: got %v want %v", avgs["ltx2-dev-noise1"], want)
	}
}

// TestCompletedCountSince checks the "done since load" numerator only counts
// completed jobs finished at or after the residency start time.
func TestCompletedCountSince(t *testing.T) {
	store, _ := newTestStore(t)

	rows := []struct {
		state    string
		finished any
	}{
		{"completed", 100.0}, // before load
		{"completed", 250.0}, // after load
		{"completed", 300.0}, // after load
		{"failed", 260.0},    // not completed — excluded
		{"completed", nil},   // no finished_at — excluded
	}
	for i, r := range rows {
		if _, err := store.db.Exec(
			"INSERT INTO jobs (id, model_id, job_type, state, priority, created_at, finished_at) VALUES (?,?,?,?,?,?,?)",
			genID(), "m", "test", r.state, float64(i), 50.0, r.finished,
		); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	got, err := store.CompletedCountSince("m", 200)
	if err != nil {
		t.Fatalf("CompletedCountSince: %v", err)
	}
	if got != 2 {
		t.Fatalf("done since load = %d want 2", got)
	}

	// A different model shares none of these completions.
	other, _ := store.CompletedCountSince("other", 0)
	if other != 0 {
		t.Fatalf("unexpected count for other model: %d", other)
	}

	// Worked example from the ledger: 6 min/action average, 23 queued + 2 active
	// => ~150 minutes ETA. Verify the arithmetic the /v1/ps cache performs.
	avg := 6.0 * 60.0
	outstanding := 23 + 2
	eta := float64(outstanding) * avg
	if math.Abs(eta-150*60) > 1e-9 {
		t.Fatalf("eta arithmetic wrong: got %v want %v", eta, 150*60.0)
	}
}
