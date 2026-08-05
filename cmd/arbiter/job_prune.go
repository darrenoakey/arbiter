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

// pruneBatchSize bounds each DELETE so retention work releases the Store writer
// lock quickly enough for job polling and scheduler state updates.
const pruneBatchSize = 100

// pruneStartupDelay keeps the first prune pass off the critical path so a
// multi-GB DB cannot delay ListenAndServe past the deploy health window.
const pruneStartupDelay = 2 * time.Minute

const pruneTerminalStates = `state IN ('completed','failed','cancelled')`

const completedPruneCandidatesSQL = `
SELECT id FROM jobs INDEXED BY idx_jobs_completed_stats
WHERE state = 'completed'
  AND finished_at IS NOT NULL
  AND finished_at < ?
ORDER BY finished_at ASC
LIMIT ?`

var remainingPruneCandidateQueries = []string{
	`SELECT id FROM jobs INDEXED BY idx_jobs_state WHERE state = 'failed' AND finished_at IS NOT NULL AND finished_at < ? ORDER BY finished_at ASC LIMIT ?`,
	`SELECT id FROM jobs INDEXED BY idx_jobs_state WHERE state = 'cancelled' AND finished_at IS NOT NULL AND finished_at < ? ORDER BY finished_at ASC LIMIT ?`,
	`SELECT id FROM jobs INDEXED BY idx_jobs_created_at WHERE state = 'completed' AND finished_at IS NULL AND created_at < ? ORDER BY created_at ASC LIMIT ?`,
	`SELECT id FROM jobs INDEXED BY idx_jobs_created_at WHERE state = 'failed' AND finished_at IS NULL AND created_at < ? ORDER BY created_at ASC LIMIT ?`,
	`SELECT id FROM jobs INDEXED BY idx_jobs_created_at WHERE state = 'cancelled' AND finished_at IS NULL AND created_at < ? ORDER BY created_at ASC LIMIT ?`,
}

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
// Completed jobs use the existing idx_jobs_completed_stats retention index.
// Less-common terminal shapes are considered only after that backlog is empty.
// No index is created at startup: building one over a 40GB jobs table blocks
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

	liveOriginals, err := s.liveFollowerOriginalsLocked()
	if err != nil {
		return nil, err
	}
	selected := make([]string, 0, pruneBatchSize)
	queries := append([]string{completedPruneCandidatesSQL}, remainingPruneCandidateQueries...)
	for _, query := range queries {
		remaining := pruneBatchSize - len(selected)
		ids, err := s.selectPruneCandidateQueryLocked(query, cutoff, remaining+len(liveOriginals), remaining, liveOriginals)
		if err != nil {
			return nil, err
		}
		selected = append(selected, ids...)
		if len(selected) == pruneBatchSize {
			break
		}
	}
	return selected, nil
}

func (s *Store) liveFollowerOriginalsLocked() (map[string]struct{}, error) {
	rows, err := s.db.Query(`SELECT error FROM jobs WHERE state = 'following' AND error LIKE 'following:%'`)
	if err != nil {
		return nil, fmt.Errorf("select live prune followers: %w", err)
	}
	defer func() { _ = rows.Close() }()
	originals := make(map[string]struct{})
	for rows.Next() {
		var reference string
		if err := rows.Scan(&reference); err != nil {
			return nil, err
		}
		if id := followerOriginalJobID(reference); id != "" {
			originals[id] = struct{}{}
		}
	}
	return originals, rows.Err()
}

func (s *Store) selectPruneCandidateQueryLocked(query string, cutoff float64, queryLimit int, resultLimit int, liveOriginals map[string]struct{}) ([]string, error) {
	rows, err := s.db.Query(query, cutoff, queryLimit)
	if err != nil {
		return nil, fmt.Errorf("select prune candidates: %w", err)
	}
	defer func() { _ = rows.Close() }()
	ids := make([]string, 0, resultLimit)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		if _, blocked := liveOriginals[id]; blocked {
			continue
		}
		ids = append(ids, id)
		if len(ids) == resultLimit {
			break
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return ids, nil
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
		every = time.Minute
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
