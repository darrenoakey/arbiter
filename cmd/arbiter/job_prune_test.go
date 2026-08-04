package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func mustCreateTerminalJob(t *testing.T, store *Store, model, jobType, state string, age time.Duration) *Job {
	t.Helper()
	job, err := store.CreateJob(model, jobType, json.RawMessage(`{"prompt":"x"}`), 1)
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	finished := nowTS() - age.Seconds()
	created := finished - 1
	if err := store.UpdateState(job.ID, state, WithFinishedAt(finished)); err != nil {
		t.Fatalf("set state %s: %v", state, err)
	}
	// Backdate created_at/finished_at so prune age checks see a real old row.
	store.mu.Lock()
	defer store.mu.Unlock()
	if _, err := store.db.Exec(
		`UPDATE jobs SET created_at = ?, finished_at = ?, state = ? WHERE id = ?`,
		created, finished, state, job.ID,
	); err != nil {
		t.Fatalf("backdate job: %v", err)
	}
	job.CreatedAt = created
	job.FinishedAt = &finished
	job.State = state
	return job
}

func TestPruneOldJobsRemovesOldTerminalKeepsRecentAndActive(t *testing.T) {
	store, outputDir := newTestStore(t)

	oldCompleted := mustCreateTerminalJob(t, store, "m", "t", "completed", 11*24*time.Hour)
	oldFailed := mustCreateTerminalJob(t, store, "m", "t", "failed", 12*24*time.Hour)
	oldCancelled := mustCreateTerminalJob(t, store, "m", "t", "cancelled", 15*24*time.Hour)
	recentCompleted := mustCreateTerminalJob(t, store, "m", "t", "completed", 2*24*time.Hour)

	active, err := store.CreateJob("m", "t", json.RawMessage(`{"prompt":"active"}`), 1)
	if err != nil {
		t.Fatalf("create active: %v", err)
	}
	// Force active row to look ancient — prune must still leave it alone.
	store.mu.Lock()
	if _, err := store.db.Exec(
		`UPDATE jobs SET created_at = ? WHERE id = ?`,
		nowTS()-20*24*time.Hour.Seconds(), active.ID,
	); err != nil {
		store.mu.Unlock()
		t.Fatalf("backdate active: %v", err)
	}
	store.mu.Unlock()

	for _, id := range []string{oldCompleted.ID, oldFailed.ID, oldCancelled.ID, recentCompleted.ID, active.ID} {
		dir := filepath.Join(outputDir, "jobs", id)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", dir, err)
		}
		if err := os.WriteFile(filepath.Join(dir, "out.txt"), []byte(id), 0o644); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}

	// Dedup entry on an old job should go with it.
	store.DedupRegister("hash-old", oldCompleted.ID)
	store.DedupRegister("hash-recent", recentCompleted.ID)

	removed, err := store.PruneOldJobs(jobRetention, outputDir)
	if err != nil {
		t.Fatalf("prune: %v", err)
	}
	if removed != 3 {
		t.Fatalf("removed = %d, want 3", removed)
	}

	for _, id := range []string{oldCompleted.ID, oldFailed.ID, oldCancelled.ID} {
		if _, err := store.GetJob(id); err == nil {
			t.Fatalf("job %s still present after prune", id)
		}
		if _, err := os.Stat(filepath.Join(outputDir, "jobs", id)); !os.IsNotExist(err) {
			t.Fatalf("output dir for %s still present: %v", id, err)
		}
	}
	for _, id := range []string{recentCompleted.ID, active.ID} {
		if _, err := store.GetJob(id); err != nil {
			t.Fatalf("kept job %s missing: %v", id, err)
		}
		if _, err := os.Stat(filepath.Join(outputDir, "jobs", id)); err != nil {
			t.Fatalf("kept output dir %s missing: %v", id, err)
		}
	}

	if got, err := store.DedupLookup("hash-old", 86400); got != "" {
		t.Fatalf("old dedup entry = %q err=%v, want gone", got, err)
	}
	if got, err := store.DedupLookup("hash-recent", 86400); err != nil || got != recentCompleted.ID {
		t.Fatalf("recent dedup entry = %q err=%v, want %s", got, err, recentCompleted.ID)
	}
}

func TestPruneOldJobsSkipsOriginalWithLiveFollower(t *testing.T) {
	store, outputDir := newTestStore(t)

	orig := mustCreateTerminalJob(t, store, "m", "t", "completed", 11*24*time.Hour)
	result := json.RawMessage(`{"file":"result.png"}`)
	store.mu.Lock()
	if _, err := store.db.Exec(
		`UPDATE jobs SET result = ? WHERE id = ?`,
		string(result), orig.ID,
	); err != nil {
		store.mu.Unlock()
		t.Fatalf("set result: %v", err)
	}
	store.mu.Unlock()

	follower, err := store.CreateFollowerJob("m", "t", json.RawMessage(`{"prompt":"x"}`), orig.ID)
	if err != nil {
		t.Fatalf("create follower: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(outputDir, "jobs", orig.ID), 0o755); err != nil {
		t.Fatalf("mkdir orig: %v", err)
	}

	removed, err := store.PruneOldJobs(jobRetention, outputDir)
	if err != nil {
		t.Fatalf("prune: %v", err)
	}
	if removed != 0 {
		t.Fatalf("removed = %d, want 0 while follower is live", removed)
	}
	if _, err := store.GetJob(orig.ID); err != nil {
		t.Fatalf("original missing while follower live: %v", err)
	}
	if _, err := store.GetJob(follower.ID); err != nil {
		t.Fatalf("follower missing: %v", err)
	}

	// Resolve the follower, then prune should reclaim the original.
	if n := store.ResolveFollowers(orig.ID, "completed", &result, "", outputDir); n != 1 {
		t.Fatalf("resolve followers = %d, want 1", n)
	}
	// Follower is now terminal but recent — only the old original is eligible.
	removed, err = store.PruneOldJobs(jobRetention, outputDir)
	if err != nil {
		t.Fatalf("second prune: %v", err)
	}
	if removed != 1 {
		t.Fatalf("removed after resolve = %d, want 1", removed)
	}
	if _, err := store.GetJob(orig.ID); err == nil {
		t.Fatal("original still present after second prune")
	}
	if _, err := store.GetJob(follower.ID); err != nil {
		t.Fatalf("recent follower pruned: %v", err)
	}
}

func TestPruneOldJobsKeepsDirWithCanonicalReference(t *testing.T) {
	store, outputDir := newTestStore(t)

	orig := mustCreateTerminalJob(t, store, "m", "t", "completed", 11*24*time.Hour)
	origDir := filepath.Join(outputDir, "jobs", orig.ID)
	if err := os.MkdirAll(origDir, 0o755); err != nil {
		t.Fatalf("mkdir orig: %v", err)
	}
	marker := filepath.Join(origDir, "shared.png")
	if err := os.WriteFile(marker, []byte("shared"), 0o644); err != nil {
		t.Fatalf("write marker: %v", err)
	}

	// Recent cache-hit job points at the old original's output.
	hit, err := store.CreateJob("m", "t", json.RawMessage(`{"prompt":"hit"}`), 1)
	if err != nil {
		t.Fatalf("create hit: %v", err)
	}
	if err := store.UpdateState(hit.ID, "completed",
		WithResult(json.RawMessage(`{"file":"`+marker+`"}`)),
		WithFinishedAt(nowTS()),
	); err != nil {
		t.Fatalf("complete hit: %v", err)
	}
	if err := store.SetCanonicalJobID(hit.ID, orig.ID); err != nil {
		t.Fatalf("set canonical: %v", err)
	}

	removed, err := store.PruneOldJobs(jobRetention, outputDir)
	if err != nil {
		t.Fatalf("prune: %v", err)
	}
	if removed != 1 {
		t.Fatalf("removed = %d, want 1 (row only)", removed)
	}
	if _, err := store.GetJob(orig.ID); err == nil {
		t.Fatal("original row still present")
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatalf("canonical output dir removed while still referenced: %v", err)
	}
	if _, err := store.GetJob(hit.ID); err != nil {
		t.Fatalf("canonical hit missing: %v", err)
	}

	// Clear the reference; dir cleanup should then succeed.
	store.mu.Lock()
	if _, err := store.db.Exec(`UPDATE jobs SET canonical_job_id = NULL WHERE id = ?`, hit.ID); err != nil {
		store.mu.Unlock()
		t.Fatalf("clear canonical: %v", err)
	}
	store.mu.Unlock()
	store.removePrunedJobDirs([]string{orig.ID}, outputDir)
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("orphaned output dir still present: %v", err)
	}
}

func TestPruneOldJobsRemovesAllOldTerminal(t *testing.T) {
	store, outputDir := newTestStore(t)

	for i := range 3 {
		j := mustCreateTerminalJob(t, store, "m", "t", "completed", time.Duration(11+i)*24*time.Hour)
		if err := os.MkdirAll(filepath.Join(outputDir, "jobs", j.ID), 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
	}
	removed, err := store.PruneOldJobs(jobRetention, outputDir)
	if err != nil {
		t.Fatalf("prune: %v", err)
	}
	if removed != 3 {
		t.Fatalf("removed = %d, want 3", removed)
	}
	var left int
	store.mu.RLock()
	err = store.db.QueryRow(`SELECT COUNT(*) FROM jobs`).Scan(&left)
	store.mu.RUnlock()
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if left != 0 {
		t.Fatalf("jobs left = %d, want 0", left)
	}
}

func TestPruneOldJobsStopsAfterOneBoundedBatch(t *testing.T) {
	store, outputDir := newTestStore(t)
	old := nowTS() - (11 * 24 * time.Hour).Seconds()

	store.mu.Lock()
	tx, err := store.db.Begin()
	if err != nil {
		store.mu.Unlock()
		t.Fatalf("begin seed tx: %v", err)
	}
	stmt, err := tx.Prepare(`
INSERT INTO jobs (id, model_id, job_type, state, priority, payload, created_at, finished_at)
VALUES (?, 'm', 't', 'completed', 1, '{}', ?, ?)`)
	if err != nil {
		_ = tx.Rollback()
		store.mu.Unlock()
		t.Fatalf("prepare seed: %v", err)
	}
	for i := range pruneBatchSize + 1 {
		if _, err := stmt.Exec(fmt.Sprintf("old-%06d", i), old-1, old); err != nil {
			_ = stmt.Close()
			_ = tx.Rollback()
			store.mu.Unlock()
			t.Fatalf("seed job %d: %v", i, err)
		}
	}
	if err := stmt.Close(); err != nil {
		_ = tx.Rollback()
		store.mu.Unlock()
		t.Fatalf("close seed statement: %v", err)
	}
	if err := tx.Commit(); err != nil {
		store.mu.Unlock()
		t.Fatalf("commit seed: %v", err)
	}
	store.mu.Unlock()

	removed, err := store.PruneOldJobs(jobRetention, outputDir)
	if err != nil {
		t.Fatalf("first prune: %v", err)
	}
	if removed != pruneBatchSize {
		t.Fatalf("first prune removed = %d, want one batch (%d)", removed, pruneBatchSize)
	}

	var left int
	store.mu.RLock()
	err = store.db.QueryRow(`SELECT COUNT(*) FROM jobs`).Scan(&left)
	store.mu.RUnlock()
	if err != nil {
		t.Fatalf("count after first prune: %v", err)
	}
	if left != 1 {
		t.Fatalf("jobs left after first prune = %d, want 1", left)
	}

	removed, err = store.PruneOldJobs(jobRetention, outputDir)
	if err != nil {
		t.Fatalf("second prune: %v", err)
	}
	if removed != 1 {
		t.Fatalf("second prune removed = %d, want 1", removed)
	}
}
