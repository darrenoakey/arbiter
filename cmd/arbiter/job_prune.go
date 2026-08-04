package main

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"
)

// jobRetention is how long terminal jobs (completed/failed/cancelled) are kept
// in the SQLite store and on-disk job output dirs. Older rows made /v1/ps stats
// scans multi-minute and the arbiter.db grew past 37GB.
const jobRetention = 10 * 24 * time.Hour

// pruneBatchSize bounds each DELETE so a multi-million-row backlog cannot hold
// the writer lock for minutes and starve the scheduler.
const pruneBatchSize = 5000

// pruneStartupDelay keeps the first prune pass off the critical path so a
// multi-GB DB cannot delay ListenAndServe past the deploy health window.
const pruneStartupDelay = 2 * time.Minute

const pruneTerminalStates = `state IN ('completed','failed','cancelled')`

// pruneCutoff returns the unix-seconds cutoff for retention.
func pruneCutoff(retention time.Duration) float64 {
	if retention <= 0 {
		retention = jobRetention
	}
	return nowTS() - retention.Seconds()
}

// PruneOldJobs deletes at most one bounded batch of terminal jobs older than
// retention, then removes their on-disk output dirs when no remaining row still
// references them via canonical_job_id. Active jobs (queued/scheduled/running/
// following) are never touched. A terminal original still referenced by a live
// follower is skipped
// so ReconcileFollowingJobs can resolve it first. Each invocation performs
// exactly one batch: an old multi-batch loop kept rescanning the 40GB database
// for hours and starved ordinary job reads. Returns the number of rows deleted.
//
// Query plan uses idx_jobs_state (state=?) then filters by age. No expression
// index is created at startup — building one over a 40GB jobs table blocks
// ListenAndServe for many minutes and fails the deploy health window.
func (s *Store) PruneOldJobs(retention time.Duration, outputDir string) (int, error) {
	cutoff := pruneCutoff(retention)
	return s.pruneOldJobsBatch(cutoff, outputDir)
}

func (s *Store) pruneOldJobsBatch(cutoff float64, outputDir string) (int, error) {
	ids, err := s.selectPruneCandidates(cutoff)
	if err != nil {
		return 0, err
	}
	if len(ids) == 0 {
		return 0, nil
	}
	if err := s.deleteJobsByID(ids); err != nil {
		return 0, err
	}
	s.removePrunedJobDirs(ids, outputDir)
	return len(ids), nil
}

func (s *Store) selectPruneCandidates(cutoff float64) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	// Prefer finished_at when present; fall back to created_at for terminal
	// rows that never recorded a finish time. Skip originals still named by a
	// live follower (error = following:<id>) so ReconcileFollowingJobs can
	// resolve them first.
	rows, err := s.db.Query(`
SELECT id FROM jobs
WHERE `+pruneTerminalStates+`
  AND COALESCE(finished_at, created_at) < ?
  AND NOT EXISTS (
    SELECT 1 FROM jobs AS followers
    WHERE followers.state = 'following'
      AND followers.error = 'following:' || jobs.id
  )
ORDER BY COALESCE(finished_at, created_at) ASC
LIMIT ?`, cutoff, pruneBatchSize)
	if err != nil {
		return nil, fmt.Errorf("select prune candidates: %w", err)
	}
	defer func() { _ = rows.Close() }()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

func (s *Store) deleteJobsByID(ids []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin prune tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	jobStmt, err := tx.Prepare(`DELETE FROM jobs WHERE id = ? AND ` + pruneTerminalStates)
	if err != nil {
		return fmt.Errorf("prepare job delete: %w", err)
	}
	defer func() { _ = jobStmt.Close() }()
	dedupStmt, err := tx.Prepare(`DELETE FROM dedup_cache WHERE job_id = ?`)
	if err != nil {
		return fmt.Errorf("prepare dedup delete: %w", err)
	}
	defer func() { _ = dedupStmt.Close() }()
	for _, id := range ids {
		if _, err := jobStmt.Exec(id); err != nil {
			return fmt.Errorf("delete job %s: %w", id, err)
		}
		if _, err := dedupStmt.Exec(id); err != nil {
			return fmt.Errorf("delete dedup for %s: %w", id, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit prune tx: %w", err)
	}
	return nil
}

func (s *Store) removePrunedJobDirs(ids []string, outputDir string) {
	if outputDir == "" {
		return
	}
	for _, id := range ids {
		n, err := s.CountCanonicalReferences(id)
		if err != nil {
			slog.Warn("job prune: count canonical refs failed", "job_id", id, "error", err)
			continue
		}
		if n > 0 {
			// A newer cache-hit job still points at this original's output paths.
			continue
		}
		dir := filepath.Join(outputDir, "jobs", id)
		if err := os.RemoveAll(dir); err != nil && !os.IsNotExist(err) {
			slog.Warn("job prune: remove output dir failed", "dir", dir, "error", err)
		}
	}
}

// RunJobPruner deletes terminal jobs older than jobRetention on an interval.
// The first pass is delayed so a huge backlog cannot block ListenAndServe past
// the deploy health window; subsequent passes run every `every`.
func RunJobPruner(ctxDone <-chan struct{}, store *Store, outputDir string, every time.Duration) {
	if every <= 0 {
		every = time.Hour
	}
	runOnce := func() {
		n, err := store.PruneOldJobs(jobRetention, outputDir)
		if err != nil {
			slog.Warn("job prune failed", "error", err)
			return
		}
		if n > 0 {
			slog.Info("job prune", "removed", n, "retention", jobRetention.String())
		}
	}

	// Delayed first pass — do not touch the 40GB jobs table before health is up.
	timer := time.NewTimer(pruneStartupDelay)
	defer timer.Stop()
	select {
	case <-ctxDone:
		return
	case <-timer.C:
		runOnce()
	}

	ticker := time.NewTicker(every)
	defer ticker.Stop()
	for {
		select {
		case <-ctxDone:
			return
		case <-ticker.C:
			runOnce()
		}
	}
}
