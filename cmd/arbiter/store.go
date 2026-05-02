package main

import (
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

type Job struct {
	ID         string           `json:"id"`
	ModelID    string           `json:"model_id"`
	JobType    string           `json:"job_type"`
	State      string           `json:"state"`
	Priority   float64          `json:"priority"`
	Payload    json.RawMessage  `json:"payload"`
	Result     *json.RawMessage `json:"result,omitempty"`
	Error      string           `json:"error,omitempty"`
	CreatedAt  float64          `json:"created_at"`
	StartedAt  *float64         `json:"started_at,omitempty"`
	FinishedAt *float64         `json:"finished_at,omitempty"`
	// CanonicalJobID is set on dedup-cache-hit jobs to point at the original
	// job whose result this one inherits. Replaces the previous
	// os.Symlink(origDir, newDir) hack which left dangling pointers on CIFS
	// when the orig dir became unreachable. Empty for normal jobs.
	CanonicalJobID string `json:"canonical_job_id,omitempty"`
}

const schema = `
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'queued',
    priority REAL NOT NULL DEFAULT 0,
    payload TEXT NOT NULL DEFAULT '{}',
    result TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    canonical_job_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority) WHERE state = 'queued';
CREATE INDEX IF NOT EXISTS idx_jobs_model ON jobs(model_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_completed_model ON jobs(state, model_id) WHERE state = 'completed';
CREATE INDEX IF NOT EXISTS idx_jobs_completed_stats ON jobs(state, finished_at, started_at, created_at) WHERE state = 'completed' AND finished_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_jobs_canonical ON jobs(canonical_job_id) WHERE canonical_job_id IS NOT NULL;
`

// migrateAddCanonicalJobID adds the canonical_job_id column to existing
// databases that pre-date the field. Idempotent — SQLite returns "duplicate
// column name" if it's already there, which we ignore.
func migrateAddCanonicalJobID(db *sql.DB) {
	_, err := db.Exec(`ALTER TABLE jobs ADD COLUMN canonical_job_id TEXT`)
	if err == nil {
		return
	}
	if !strings.Contains(err.Error(), "duplicate column") {
		// Anything else is unexpected — log but don't abort startup.
		// scanJob falls back gracefully if the column is missing.
		_ = err
	}
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_jobs_canonical ON jobs(canonical_job_id) WHERE canonical_job_id IS NOT NULL`)
}

type Store struct {
	db *sql.DB
	mu sync.RWMutex
}

func NewStore(dbPath string) (*Store, error) {
	db, err := sql.Open("sqlite", dbPath+"?_journal_mode=WAL&_synchronous=NORMAL&_busy_timeout=5000")
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	db.SetMaxOpenConns(1) // SQLite doesn't support concurrent writers
	// Migrate BEFORE the schema block runs — the schema's CREATE INDEX on
	// canonical_job_id will fail if the column doesn't exist on a pre-
	// existing jobs table that was created without it.
	migrateAddCanonicalJobID(db)
	if _, err := db.Exec(schema); err != nil {
		return nil, fmt.Errorf("init schema: %w", err)
	}
	return &Store{db: db}, nil
}

func genID() string {
	b := make([]byte, 6)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func nowTS() float64 {
	return float64(time.Now().UnixNano()) / 1e9
}

func (s *Store) CreateJob(modelID, jobType string, payload json.RawMessage, priority float64) (*Job, error) {
	id := genID()
	now := nowTS()
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.db.Exec(
		"INSERT INTO jobs (id, model_id, job_type, state, priority, payload, created_at) VALUES (?,?,?,'queued',?,?,?)",
		id, modelID, jobType, priority, string(payload), now,
	)
	if err != nil {
		return nil, err
	}
	return &Job{
		ID: id, ModelID: modelID, JobType: jobType,
		State: "queued", Priority: priority, Payload: payload, CreatedAt: now,
	}, nil
}

func (s *Store) GetJob(id string) (*Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.scanJob(s.db.QueryRow("SELECT * FROM jobs WHERE id = ?", id))
}

func (s *Store) scanJob(row *sql.Row) (*Job, error) {
	var j Job
	var payload, result, errStr, canonical sql.NullString
	var startedAt, finishedAt sql.NullFloat64
	err := row.Scan(&j.ID, &j.ModelID, &j.JobType, &j.State, &j.Priority,
		&payload, &result, &errStr, &j.CreatedAt, &startedAt, &finishedAt, &canonical)
	if err != nil {
		return nil, err
	}
	if payload.Valid {
		j.Payload = json.RawMessage(payload.String)
	}
	if result.Valid {
		rm := json.RawMessage(result.String)
		j.Result = &rm
	}
	if errStr.Valid {
		j.Error = errStr.String
	}
	if startedAt.Valid {
		j.StartedAt = &startedAt.Float64
	}
	if finishedAt.Valid {
		j.FinishedAt = &finishedAt.Float64
	}
	if canonical.Valid {
		j.CanonicalJobID = canonical.String
	}
	return &j, nil
}

// SetCanonicalJobID marks a job as a dedup-cache-hit pointer to origID.
// Replaces the previous dedup symlink hack — the lookup is now in the DB
// and survives unmount / FS issues that broke filesystem symlinks.
func (s *Store) SetCanonicalJobID(jobID, origID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.db.Exec(
		`UPDATE jobs SET canonical_job_id = ? WHERE id = ?`,
		origID, jobID,
	)
	return err
}

// CountCanonicalReferences returns how many jobs point at origID via
// canonical_job_id. Used by output-cleanup paths so the orig dir can't
// be removed while followers still reference its result paths.
func (s *Store) CountCanonicalReferences(origID string) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var n int
	err := s.db.QueryRow(
		`SELECT COUNT(*) FROM jobs WHERE canonical_job_id = ?`,
		origID,
	).Scan(&n)
	return n, err
}

func (s *Store) ListJobs(state, modelID string, limit int) ([]*Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	query := "SELECT * FROM jobs WHERE 1=1"
	var args []any
	if state != "" {
		query += " AND state = ?"
		args = append(args, state)
	}
	if modelID != "" {
		query += " AND model_id = ?"
		args = append(args, modelID)
	}
	query += " ORDER BY created_at DESC LIMIT ?"
	args = append(args, limit)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var jobs []*Job
	for rows.Next() {
		j, err := scanJobFromRows(rows)
		if err != nil {
			return nil, err
		}
		jobs = append(jobs, j)
	}
	return jobs, nil
}

// scanJobFromRows is the shared row-scan helper. Centralised so adding
// columns to the schema only requires changing one place — the previous
// open-coded scans in 4 places drift apart immediately when the column
// list changes.
func scanJobFromRows(rows *sql.Rows) (*Job, error) {
	var j Job
	var payload, result, errStr, canonical sql.NullString
	var startedAt, finishedAt sql.NullFloat64
	if err := rows.Scan(&j.ID, &j.ModelID, &j.JobType, &j.State, &j.Priority,
		&payload, &result, &errStr, &j.CreatedAt, &startedAt, &finishedAt, &canonical); err != nil {
		return nil, err
	}
	if payload.Valid {
		j.Payload = json.RawMessage(payload.String)
	}
	if result.Valid {
		rm := json.RawMessage(result.String)
		j.Result = &rm
	}
	if errStr.Valid {
		j.Error = errStr.String
	}
	if startedAt.Valid {
		j.StartedAt = &startedAt.Float64
	}
	if finishedAt.Valid {
		j.FinishedAt = &finishedAt.Float64
	}
	if canonical.Valid {
		j.CanonicalJobID = canonical.String
	}
	return &j, nil
}

func (s *Store) PickNextJob(excludeModels map[string]bool) (*Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	query := "SELECT * FROM jobs WHERE state = 'queued'"
	var args []any
	if len(excludeModels) > 0 {
		for m := range excludeModels {
			query += " AND model_id != ?"
			args = append(args, m)
		}
	}
	// FCFS: oldest queued job wins regardless of model. The per-model
	// `priority` column is left in place for observability but is NOT used
	// for ordering. SJF (shortest-job-first) caused fast models to jump
	// ahead of slow batch work that had been submitted earlier, which in
	// practice made batch pipelines like ltx2 get interrupted by unrelated
	// image jobs that were queued later.
	query += " ORDER BY created_at ASC LIMIT 1"

	row := s.db.QueryRow(query, args...)
	j, err := s.scanJob(row)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	return j, err
}

func (s *Store) UpdateState(jobID, state string, opts ...func(*stateUpdate)) error {
	u := &stateUpdate{}
	for _, o := range opts {
		o(u)
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	sets := "state = ?"
	args := []any{state}
	if u.startedAt != nil {
		sets += ", started_at = ?"
		args = append(args, *u.startedAt)
	} else if u.clearStartedAt {
		sets += ", started_at = NULL"
	}
	if u.finishedAt != nil {
		sets += ", finished_at = ?"
		args = append(args, *u.finishedAt)
	}
	if u.result != nil {
		sets += ", result = ?"
		args = append(args, string(*u.result))
	}
	if u.error != "" {
		sets += ", error = ?"
		args = append(args, u.error)
	}
	args = append(args, jobID)
	_, err := s.db.Exec("UPDATE jobs SET "+sets+" WHERE id = ?", args...)
	return err
}

type stateUpdate struct {
	startedAt  *float64
	finishedAt *float64
	result     *json.RawMessage
	error      string
	clearStartedAt bool
}

func WithStartedAt(t float64) func(*stateUpdate)  { return func(u *stateUpdate) { u.startedAt = &t } }
func WithFinishedAt(t float64) func(*stateUpdate) { return func(u *stateUpdate) { u.finishedAt = &t } }
func WithClearStartedAt() func(*stateUpdate)      { return func(u *stateUpdate) { u.clearStartedAt = true } }
func WithResult(r json.RawMessage) func(*stateUpdate) {
	return func(u *stateUpdate) { u.result = &r }
}
func WithError(e string) func(*stateUpdate) { return func(u *stateUpdate) { u.error = e } }

func (s *Store) UpdatePriority(modelID string, priority float64) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	res, err := s.db.Exec(
		"UPDATE jobs SET priority = ? WHERE model_id = ? AND state = 'queued'",
		priority, modelID,
	)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	return int(n), nil
}

func (s *Store) CountByState(modelID string) (map[string]int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	query := "SELECT state, COUNT(*) FROM jobs"
	var args []any
	if modelID != "" {
		query += " WHERE model_id = ?"
		args = append(args, modelID)
	}
	query += " GROUP BY state"

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	counts := make(map[string]int)
	for rows.Next() {
		var state string
		var count int
		rows.Scan(&state, &count)
		counts[state] = count
	}
	return counts, nil
}

// OldestQueuedAgeByModel returns the wait time (seconds since created_at) of
// the oldest currently-queued job per model. Models with no queued jobs are
// absent from the map. Used by the scheduler to age the pressure budget so
// long-waiting jobs aren't starved by continuous low-pressure traffic.
func (s *Store) OldestQueuedAgeByModel() (map[string]float64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	rows, err := s.db.Query(
		"SELECT model_id, MIN(created_at) FROM jobs WHERE state = 'queued' GROUP BY model_id",
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	now := nowTS()
	out := make(map[string]float64)
	for rows.Next() {
		var modelID string
		var oldestTs float64
		if err := rows.Scan(&modelID, &oldestTs); err != nil {
			return nil, err
		}
		age := now - oldestTs
		if age < 0 {
			age = 0
		}
		out[modelID] = age
	}
	return out, rows.Err()
}

func (s *Store) CountActive(modelID string) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var count int
	err := s.db.QueryRow(
		"SELECT COUNT(*) FROM jobs WHERE model_id = ? AND state IN ('scheduled','running')",
		modelID,
	).Scan(&count)
	return count, err
}

func (s *Store) CancelJob(jobID string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := nowTS()
	res, err := s.db.Exec(
		"UPDATE jobs SET state = 'cancelled', finished_at = ?, error = CASE WHEN state = 'following' THEN 'cancelled while following original job' ELSE error END WHERE id = ? AND state IN ('queued','scheduled','following')",
		now, jobID,
	)
	if err != nil {
		return false, err
	}
	n, _ := res.RowsAffected()
	return n > 0, nil
}

func (s *Store) RecoverFromCrash() (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	res, err := s.db.Exec(
		"UPDATE jobs SET state = 'queued', started_at = NULL WHERE state IN ('scheduled','running')",
	)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	return int(n), nil
}

func (s *Store) RecoverStuckScheduled(olderThanSec float64) (int, error) {
	cutoff := nowTS() - olderThanSec
	s.mu.Lock()
	defer s.mu.Unlock()
	res, err := s.db.Exec(
		"UPDATE jobs SET state = 'queued', started_at = NULL WHERE state = 'scheduled' AND started_at IS NOT NULL AND started_at < ?",
		cutoff,
	)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	return int(n), nil
}

func (s *Store) CancelQueuedForModel(modelID string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := nowTS()
	res, err := s.db.Exec(
		"UPDATE jobs SET state = 'cancelled', finished_at = ? WHERE model_id = ? AND state = 'queued'",
		now, modelID,
	)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	return int(n), nil
}

func (s *Store) CancelFollowingForModel(modelID, errMsg string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := nowTS()
	res, err := s.db.Exec(
		"UPDATE jobs SET state = 'cancelled', error = ?, finished_at = ? WHERE model_id = ? AND state = 'following'",
		errMsg, now, modelID,
	)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	return int(n), nil
}

func (s *Store) FailActiveForModel(modelID, errMsg string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := nowTS()
	res, err := s.db.Exec(
		"UPDATE jobs SET state = 'failed', error = ?, finished_at = ? WHERE model_id = ? AND state IN ('scheduled','running')",
		errMsg, now, modelID,
	)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	return int(n), nil
}

func (s *Store) Close() {
	s.db.Close()
}

// GetJobs fetches multiple jobs by ID in a single query.
// Returns a map of jobID -> Job. Missing IDs are omitted.
func (s *Store) GetJobs(ids []string) (map[string]*Job, error) {
	if len(ids) == 0 {
		return map[string]*Job{}, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	placeholders := ""
	args := make([]any, len(ids))
	for i, id := range ids {
		if i > 0 {
			placeholders += ","
		}
		placeholders += "?"
		args[i] = id
	}

	query := "SELECT * FROM jobs WHERE id IN (" + placeholders + ")"
	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	result := make(map[string]*Job, len(ids))
	for rows.Next() {
		j, err := scanJobFromRows(rows)
		if err != nil {
			return nil, err
		}
		result[j.ID] = j
	}
	return result, nil
}

func (s *Store) CompletedJobStats(modelID string) (int, float64, float64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	query := `
SELECT
	COUNT(*),
	COALESCE(AVG(finished_at - created_at), 0),
	COALESCE(AVG(CASE WHEN started_at IS NOT NULL THEN finished_at - started_at END), 0)
FROM jobs
WHERE state = 'completed' AND finished_at IS NOT NULL
`
	var args []any
	if modelID != "" {
		query += " AND model_id = ?"
		args = append(args, modelID)
	}

	var count int
	var avgTotal float64
	var avgExec float64
	err := s.db.QueryRow(query, args...).Scan(&count, &avgTotal, &avgExec)
	return count, avgTotal, avgExec, err
}

// GetRunningJobs returns all jobs currently in the "running" state with their model_id and started_at.
func (s *Store) GetRunningJobs() ([]*Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query(
		"SELECT id, model_id, job_type, state, priority, payload, result, error, created_at, started_at, finished_at, canonical_job_id FROM jobs WHERE state = 'running'",
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var jobs []*Job
	for rows.Next() {
		j, err := scanJobFromRows(rows)
		if err != nil {
			return nil, err
		}
		jobs = append(jobs, j)
	}
	return jobs, nil
}

// GetActiveJobs returns all jobs in a non-terminal state (queued, scheduled, running, following).
func (s *Store) GetActiveJobs() ([]*Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query(
		"SELECT id, model_id, job_type, state, priority, payload, result, error, created_at, started_at, finished_at, canonical_job_id FROM jobs WHERE state IN ('queued','scheduled','running','following')",
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var jobs []*Job
	for rows.Next() {
		j, err := scanJobFromRows(rows)
		if err != nil {
			return nil, err
		}
		jobs = append(jobs, j)
	}
	return jobs, nil
}
