package main

import (
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
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
	// ExcludedHosts is the set of host ids that must NOT be picked for this job.
	// It is how transparent mid-job failover is made durable: when a remote
	// executor disappears while running a job, the dead host is appended here,
	// the job goes running→queued, and PickInstance walks the placement chain
	// skipping every excluded host — so the next attempt lands on a different
	// box. Persisted as a JSON array in the excluded_hosts column so it survives
	// a crash/restart mid-failover. Empty/nil for the common case.
	ExcludedHosts []string `json:"excluded_hosts,omitempty"`
	// RequestedModel is the caller's original model string (e.g. "local-chat"
	// or "qwen3.6-35b"). It is used at serve time to echo the requested string
	// in response.model and in job results, even when the canonical target or a
	// cached/deduped result carries a different model field.
	RequestedModel string `json:"requested_model,omitempty"`
}

// HostExcluded reports whether the given host id is in the job's excluded set.
func (j *Job) HostExcluded(hostID string) bool {
	for _, h := range j.ExcludedHosts {
		if h == hostID {
			return true
		}
	}
	return false
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
    canonical_job_id TEXT,
    excluded_hosts TEXT,
    requested_model TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority) WHERE state = 'queued';
CREATE INDEX IF NOT EXISTS idx_jobs_model ON jobs(model_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_completed_model ON jobs(state, model_id) WHERE state = 'completed';
CREATE INDEX IF NOT EXISTS idx_jobs_completed_stats ON jobs(state, finished_at, started_at, created_at) WHERE state = 'completed' AND finished_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_jobs_canonical ON jobs(canonical_job_id) WHERE canonical_job_id IS NOT NULL;

-- Persisted per-model rolling average of seconds-per-completed-action. This is
-- the ONLY source of the ETA the dashboard shows: it survives daemon restarts
-- (so ETAs stay meaningful across a bounce) and is fed exclusively by real
-- completed-job execution timings via RecordActionDuration. avg_action_seconds
-- is an exponential moving average (first sample seeds it, alpha thereafter).
CREATE TABLE IF NOT EXISTS model_stats (
    model_id TEXT PRIMARY KEY,
    avg_action_seconds REAL NOT NULL,
    samples INTEGER NOT NULL,
    updated_at REAL NOT NULL
);
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
	if _, err := db.Exec(`CREATE INDEX IF NOT EXISTS idx_jobs_canonical ON jobs(canonical_job_id) WHERE canonical_job_id IS NOT NULL`); err != nil {
		return
	}
}

// migrateAddExcludedHosts adds the excluded_hosts column to databases that
// pre-date transparent failover. Idempotent — a "duplicate column name" error
// (column already present) is ignored.
func migrateAddExcludedHosts(db *sql.DB) {
	_, err := db.Exec(`ALTER TABLE jobs ADD COLUMN excluded_hosts TEXT`)
	if err == nil || strings.Contains(err.Error(), "duplicate column") {
		return
	}
	// Anything else is unexpected — scanJob falls back gracefully if the column
	// is missing, so don't abort startup.
	_ = err
}

// migrateAddRequestedModel adds the requested_model column to databases that
// pre-date the LLM alias layer. Idempotent.
func migrateAddRequestedModel(db *sql.DB) error {
	_, err := db.Exec(`ALTER TABLE jobs ADD COLUMN requested_model TEXT`)
	if err == nil || strings.Contains(err.Error(), "duplicate column") ||
		strings.Contains(err.Error(), "no such table") {
		return nil
	}
	return fmt.Errorf("add requested_model column: %w", err)
}

type Store struct {
	db *sql.DB
	mu sync.RWMutex
	// excludedAt tracks the write-time of each (job, host) exclusion so
	// ClearExcludedHostForActiveJobs can dampen with a min-age: an exclusion
	// written seconds ago by an active host flap is NOT forgiven even when the
	// host reports RECOVERED, preventing clear→fail→exclude→recover→clear
	// churn at the liveness cadence. Absent entries (pre-existing rows, or any
	// exclusion present across a restart) are treated as old and ARE clearable.
	// Bounded by the number of distinct jobs that ever failover; cleared on restart.
	excludedAt map[string]map[string]time.Time
}

func NewStore(dbPath string) (*Store, error) {
	db, err := sql.Open("sqlite", dbPath+"?_journal_mode=WAL&_synchronous=NORMAL&_busy_timeout=5000")
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	// WAL mode allows concurrent readers. Writers still serialize via
	// Store.mu; a single connection starves /v1/jobs lookups whenever the
	// completed-stats scan holds the only handle for minutes on a large DB.
	db.SetMaxOpenConns(8)
	db.SetMaxIdleConns(8)
	// Migrate BEFORE the schema block runs — the schema's CREATE INDEX on
	// canonical_job_id will fail if the column doesn't exist on a pre-
	// existing jobs table that was created without it.
	migrateAddCanonicalJobID(db)
	migrateAddExcludedHosts(db)
	if err := migrateAddRequestedModel(db); err != nil {
		_ = db.Close()
		return nil, err
	}
	if _, err := db.Exec(schema); err != nil {
		return nil, fmt.Errorf("init schema: %w", err)
	}
	return &Store{db: db, excludedAt: map[string]map[string]time.Time{}}, nil
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
	return s.CreateJobWithRequestedModel(modelID, jobType, payload, priority, "")
}

// CreateJobWithRequestedModel persists both canonical routing identity and the
// caller's original model string for response-time provenance.
func (s *Store) CreateJobWithRequestedModel(modelID, jobType string, payload json.RawMessage, priority float64, requestedModel string) (*Job, error) {
	id := genID()
	now := nowTS()
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.db.Exec(
		"INSERT INTO jobs (id, model_id, job_type, state, priority, payload, created_at, requested_model) VALUES (?,?,?,'queued',?,?,?,?)",
		id, modelID, jobType, priority, string(payload), now, nullableRequestedModel(requestedModel),
	)
	if err != nil {
		return nil, err
	}
	return &Job{
		ID: id, ModelID: modelID, JobType: jobType,
		State: "queued", Priority: priority, Payload: payload, CreatedAt: now,
		RequestedModel: requestedModel,
	}, nil
}

func nullableRequestedModel(requestedModel string) sql.NullString {
	return sql.NullString{String: requestedModel, Valid: requestedModel != ""}
}

func (s *Store) GetJob(id string) (*Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.scanJob(s.db.QueryRow("SELECT * FROM jobs WHERE id = ?", id))
}

func (s *Store) scanJob(row *sql.Row) (*Job, error) {
	var j Job
	var payload, result, errStr, canonical, excluded, requested sql.NullString
	var startedAt, finishedAt sql.NullFloat64
	err := row.Scan(&j.ID, &j.ModelID, &j.JobType, &j.State, &j.Priority,
		&payload, &result, &errStr, &j.CreatedAt, &startedAt, &finishedAt, &canonical, &excluded, &requested)
	if err != nil {
		return nil, err
	}
	fillJobNullable(&j, payload, result, errStr, canonical, excluded, requested, startedAt, finishedAt)
	return &j, nil
}

// fillJobNullable is the shared decode of the nullable job columns so scanJob
// (QueryRow) and scanJobFromRows (Rows) stay in lockstep when columns change.
func fillJobNullable(j *Job, payload, result, errStr, canonical, excluded, requested sql.NullString, startedAt, finishedAt sql.NullFloat64) {
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
	if requested.Valid && requested.String != "" {
		j.RequestedModel = requested.String
	}
	if excluded.Valid && excluded.String != "" {
		// Stored as a JSON array; ignore malformed values (treat as none).
		_ = json.Unmarshal([]byte(excluded.String), &j.ExcludedHosts)
	}
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

// AddExcludedHost durably appends a host id to a job's excluded set and returns
// the updated list. This is the persistence half of transparent failover: a
// remote executor that died mid-job is recorded here so PickInstance never
// re-routes the requeued job back to the dead box, and so the exclusion
// survives a crash mid-failover. Idempotent — re-adding an existing host is a
// no-op. The read+write is done under the write lock so concurrent failovers on
// the same job can't clobber each other.
func (s *Store) AddExcludedHost(jobID, hostID string) ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var existing sql.NullString
	if err := s.db.QueryRow("SELECT excluded_hosts FROM jobs WHERE id = ?", jobID).Scan(&existing); err != nil {
		return nil, err
	}
	var hosts []string
	if existing.Valid && existing.String != "" {
		if err := json.Unmarshal([]byte(existing.String), &hosts); err != nil {
			return nil, fmt.Errorf("decode excluded hosts for job %s: %w", jobID, err)
		}
	}
	for _, h := range hosts {
		if h == hostID {
			return hosts, nil // already excluded — no write
		}
	}
	hosts = append(hosts, hostID)
	if s.excludedAt[jobID] == nil {
		s.excludedAt[jobID] = make(map[string]time.Time)
	}
	if _, ok := s.excludedAt[jobID][hostID]; !ok {
		s.excludedAt[jobID][hostID] = time.Now() // first-write time; drives the min-age dampener
	}
	encoded, _ := json.Marshal(hosts)
	if _, err := s.db.Exec("UPDATE jobs SET excluded_hosts = ? WHERE id = ?", string(encoded), jobID); err != nil {
		return nil, err
	}
	return hosts, nil
}

// ClearExcludedHostForActiveJobs removes hostID from the excluded set of every
// non-terminal job, EXCEPT exclusions younger than minAge: a just-written
// exclusion from an actively flapping host must age before it is forgiven, or
// clear→fail→exclude→recover→clear churns at the liveness cadence. Exclusions
// with no recorded write-time (pre-existing rows, or any exclusion present
// across a restart) are treated as old and ARE cleared. Returns the count
// healed. Called from the host monitor on RECOVERED and on the first
// successful probe after (re)start.
func (s *Store) ClearExcludedHostForActiveJobs(hostID string, minAge time.Duration) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rows, err := s.db.Query(`SELECT id, excluded_hosts FROM jobs WHERE state IN ('queued','scheduled','running','following') AND excluded_hosts IS NOT NULL AND excluded_hosts != ''`)
	if err != nil {
		return 0, err
	}
	type pending struct {
		id    string
		hosts []string
	}
	var toUpdate []pending
	healed := 0
	now := time.Now()
	for rows.Next() {
		var id, raw string
		if err := rows.Scan(&id, &raw); err != nil {
			continue
		}
		var hosts []string
		if err := json.Unmarshal([]byte(raw), &hosts); err != nil {
			continue
		}
		found := false
		for _, h := range hosts {
			if h == hostID {
				found = true
				break
			}
		}
		if !found {
			continue
		}
		// Dampener: honour a fresh exclusion from an active flap.
		if jobTimes, ok := s.excludedAt[id]; ok {
			if wt, ok := jobTimes[hostID]; ok && now.Sub(wt) < minAge {
				continue
			}
		}
		filtered := make([]string, 0, len(hosts))
		for _, h := range hosts {
			if h != hostID {
				filtered = append(filtered, h)
			}
		}
		toUpdate = append(toUpdate, pending{id, filtered})
		healed++
	}
	if err := rows.Err(); err != nil {
		_ = rows.Close()
		return healed, err
	}
	_ = rows.Close()
	for _, p := range toUpdate {
		var enc string
		if len(p.hosts) > 0 {
			b, _ := json.Marshal(p.hosts)
			enc = string(b)
		}
		if _, err := s.db.Exec("UPDATE jobs SET excluded_hosts = ? WHERE id = ?", enc, p.id); err != nil {
			slog.Warn("clear excluded host: update failed", "job", p.id, "error", err)
			continue
		}
		if jt, ok := s.excludedAt[p.id]; ok {
			delete(jt, hostID)
			if len(jt) == 0 {
				delete(s.excludedAt, p.id)
			}
		}
	}
	return healed, nil
}

// ExclusionIsStale reports whether the (jobID, hostID) exclusion is old enough
// to forgive (>= minAge), OR has no recorded write-time (pre-existing row, or
// present across a restart). Used by the relaxed-placement backstop to ignore
// only exclusions recording a PAST absence while honouring fresh ones from an
// active flap.
func (s *Store) ExclusionIsStale(jobID, hostID string, minAge time.Duration) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if jt, ok := s.excludedAt[jobID]; ok {
		if wt, ok := jt[hostID]; ok {
			return time.Since(wt) >= minAge
		}
	}
	return true // no recorded write-time → treat as old (forgive)
}

// TouchExclusion refreshes the write-time of an existing (jobID, hostID)
// exclusion to now, so a host that just failed a relaxed placement is honoured
// as freshly excluded for the next minAge window (not immediately re-relaxed).
// No-op if the exclusion is not recorded.
func (s *Store) TouchExclusion(jobID, hostID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if jt, ok := s.excludedAt[jobID]; ok {
		if _, ok := jt[hostID]; ok {
			jt[hostID] = time.Now()
		}
	}
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
	defer func() { _ = rows.Close() }()

	var jobs []*Job
	for rows.Next() {
		j, err := scanJobFromRows(rows)
		if err != nil {
			return nil, err
		}
		jobs = append(jobs, j)
	}
	return jobs, rows.Err()
}

// scanJobFromRows is the shared row-scan helper. Centralised so adding
// columns to the schema only requires changing one place — the previous
// open-coded scans in 4 places drift apart immediately when the column
// list changes.
func scanJobFromRows(rows *sql.Rows) (*Job, error) {
	var j Job
	var payload, result, errStr, canonical, excluded, requested sql.NullString
	var startedAt, finishedAt sql.NullFloat64
	if err := rows.Scan(&j.ID, &j.ModelID, &j.JobType, &j.State, &j.Priority,
		&payload, &result, &errStr, &j.CreatedAt, &startedAt, &finishedAt, &canonical, &excluded, &requested); err != nil {
		return nil, err
	}
	fillJobNullable(&j, payload, result, errStr, canonical, excluded, requested, startedAt, finishedAt)
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
	// Terminal jobs are IMMUTABLE: once a row is completed/failed/cancelled, no
	// UpdateState may change it — not to an active state (resurrection) and not
	// to another terminal state (a late/duplicate result overwriting the first).
	// This is the idempotency guard transparent failover relies on: a late
	// upstream response from a dead host cannot overwrite the result the
	// failover target already wrote, so exactly one result is recorded per job.
	// Every legitimate terminal write in the codebase is a first transition from
	// an active state, so blocking terminal→* loses nothing.
	_, err := s.db.Exec(
		"UPDATE jobs SET "+sets+" WHERE id = ? AND state NOT IN ('completed','failed','cancelled')",
		args...,
	)
	return err
}

type stateUpdate struct {
	startedAt      *float64
	finishedAt     *float64
	result         *json.RawMessage
	error          string
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
	defer func() { _ = rows.Close() }()

	counts := make(map[string]int)
	for rows.Next() {
		var state string
		var count int
		if err := rows.Scan(&state, &count); err != nil {
			return nil, err
		}
		counts[state] = count
	}
	return counts, rows.Err()
}

// ActivePendingByModel returns, per model, the number of jobs in an active
// state (queued + scheduled + running) in a SINGLE indexed scan over just the
// active rows. The scheduler calls this every sweep; the old approach
// (CountByState once per model) scanned the entire jobs table — all completed,
// cancelled and failed history included — on every sweep, which pinned a core
// as job history grew (gemma alone has 86k completed rows). Active rows are
// few, so this is O(active jobs), not O(all history). Models with no active
// jobs are simply absent from the map (callers read missing keys as 0).
func (s *Store) ActivePendingByModel() (map[string]int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query(
		"SELECT model_id, COUNT(*) FROM jobs WHERE state IN ('queued','scheduled','running') GROUP BY model_id",
	)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	out := make(map[string]int)
	for rows.Next() {
		var modelID string
		var n int
		if err := rows.Scan(&modelID, &n); err != nil {
			return nil, err
		}
		out[modelID] = n
	}
	return out, rows.Err()
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
	defer func() { _ = rows.Close() }()
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

// PickOldestQueuedJobForModel returns the oldest queued job for the given model,
// ignoring all other models and non-queued states. Returns (nil, nil) when no
// matching row exists.
func (s *Store) PickOldestQueuedJobForModel(modelID string) (*Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	row := s.db.QueryRow(
		"SELECT * FROM jobs WHERE state = 'queued' AND model_id = ? ORDER BY created_at ASC LIMIT 1",
		modelID,
	)
	j, err := s.scanJob(row)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	return j, err
}

// PickQueuedJobsForModel returns up to `limit` oldest queued jobs for the model
// in FIFO (created_at ASC) order, so the dispatcher can scan PAST a job no host
// can currently accept instead of head-of-line blocking every sibling behind it
// (Fix 4: the 24-job/66-min freeze).
func (s *Store) PickQueuedJobsForModel(modelID string, limit int) ([]*Job, error) {
	if limit < 1 {
		limit = 1
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	rows, err := s.db.Query(
		"SELECT * FROM jobs WHERE state = 'queued' AND model_id = ? ORDER BY created_at ASC LIMIT ?",
		modelID, limit,
	)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	var jobs []*Job
	for rows.Next() {
		var j Job
		var payload, result, errStr, canonical, excluded, requested sql.NullString
		var startedAt, finishedAt sql.NullFloat64
		if err := rows.Scan(&j.ID, &j.ModelID, &j.JobType, &j.State, &j.Priority,
			&payload, &result, &errStr, &j.CreatedAt, &startedAt, &finishedAt, &canonical, &excluded, &requested); err != nil {
			return nil, err
		}
		fillJobNullable(&j, payload, result, errStr, canonical, excluded, requested, startedAt, finishedAt)
		jobs = append(jobs, &j)
	}
	return jobs, rows.Err()
}

// CountRequestedModelSince returns how many jobs used the given requested_model
// string (typically an alias) at or after the given timestamp. Used by the
// guarded DELETE /v1/llm/aliases/{alias} endpoint.
func (s *Store) CountRequestedModelSince(requestedModel string, since float64) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var n int
	err := s.db.QueryRow(
		"SELECT COUNT(*) FROM jobs WHERE requested_model = ? AND created_at >= ?",
		requestedModel, since,
	).Scan(&n)
	return n, err
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
	if _, err := s.db.Exec(
		"UPDATE jobs SET state = 'failed', error = COALESCE(NULLIF(error, ''), 'recovered non-terminal job with finished_at') WHERE state IN ('queued','scheduled','running','following') AND finished_at IS NOT NULL",
	); err != nil {
		return 0, err
	}
	res, err := s.db.Exec(
		"UPDATE jobs SET state = 'queued', started_at = NULL WHERE state IN ('scheduled','running') AND finished_at IS NULL",
	)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	return int(n), nil
}

// ListStuckScheduled returns the IDs of jobs stuck in 'scheduled' longer than
// olderThanSec. The caller decides which to requeue — in particular it skips
// jobs whose dispatch goroutine is still alive (in the scheduler's in-flight
// registry), because a legitimately slow load (denoise models take minutes)
// keeps a job 'scheduled' without being orphaned. Requeuing those caused
// double-dispatch and leaked reservations.
func (s *Store) ListStuckScheduled(olderThanSec float64) ([]string, error) {
	cutoff := nowTS() - olderThanSec
	s.mu.Lock()
	defer s.mu.Unlock()
	rows, err := s.db.Query(
		"SELECT id FROM jobs WHERE state = 'scheduled' AND started_at IS NOT NULL AND started_at < ?",
		cutoff,
	)
	if err != nil {
		return nil, err
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
	_ = s.db.Close()
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
	defer func() { _ = rows.Close() }()

	result := make(map[string]*Job, len(ids))
	for rows.Next() {
		j, err := scanJobFromRows(rows)
		if err != nil {
			return nil, err
		}
		result[j.ID] = j
	}
	return result, rows.Err()
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

// JobStats holds completed-job aggregates for a single model or the whole DB.
type JobStats struct {
	Count    int
	AvgTotal float64
	AvgExec  float64
}

// CountByStateGrouped returns job counts per (model, state) plus a global
// per-state total, computed in ONE table scan. The /v1/ps cache used to call
// CountByState once globally and once per model (an N+1 full scan every
// second); this collapses that into a single GROUP BY.
func (s *Store) CountByStateGrouped() (perModel map[string]map[string]int, global map[string]int, err error) {
	// This all-history query can take minutes on the production database. Do
	// not hold the operational Store mutex: sql.DB is concurrency-safe and WAL
	// readers can coexist, while a queued writer on sync.RWMutex would otherwise
	// block every new GetJob reader until this scan completes.
	rows, err := s.db.Query("SELECT model_id, state, COUNT(*) FROM jobs GROUP BY model_id, state")
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = rows.Close() }()

	perModel = make(map[string]map[string]int)
	global = make(map[string]int)
	for rows.Next() {
		var modelID, state string
		var count int
		if err := rows.Scan(&modelID, &state, &count); err != nil {
			return nil, nil, err
		}
		if perModel[modelID] == nil {
			perModel[modelID] = make(map[string]int)
		}
		perModel[modelID][state] = count
		global[state] += count
	}
	return perModel, global, rows.Err()
}

// CompletedJobStatsGrouped returns completed-job statistics per model plus a
// global aggregate, in ONE scan of the completed-jobs index. Replaces the
// per-model CompletedJobStats loop the /v1/ps cache used to run (a full scan
// per model, every second). Averages are computed from per-model SUM/COUNT so
// the global figure matches a single AVG over all completed jobs exactly.
func (s *Store) CompletedJobStatsGrouped() (perModel map[string]JobStats, global JobStats, err error) {
	// Keep the all-history scan outside the operational Store mutex for the same
	// reason as CountByStateGrouped: SQLite WAL supplies read isolation without
	// starving primary-key job polling behind Go's writer-preferring RWMutex.
	rows, err := s.db.Query(`
SELECT
	model_id,
	COUNT(*),
	COALESCE(SUM(finished_at - created_at), 0),
	SUM(CASE WHEN started_at IS NOT NULL THEN finished_at - started_at END),
	COUNT(CASE WHEN started_at IS NOT NULL THEN 1 END)
FROM jobs
WHERE state = 'completed' AND finished_at IS NOT NULL
GROUP BY model_id`)
	if err != nil {
		return nil, JobStats{}, err
	}
	defer func() { _ = rows.Close() }()

	perModel = make(map[string]JobStats)
	var gCount, gExecCount int
	var gSumTotal, gSumExec float64
	for rows.Next() {
		var modelID string
		var count, execCount int
		var sumTotal float64
		var sumExec sql.NullFloat64
		if err := rows.Scan(&modelID, &count, &sumTotal, &sumExec, &execCount); err != nil {
			return nil, JobStats{}, err
		}
		js := JobStats{Count: count}
		if count > 0 {
			js.AvgTotal = sumTotal / float64(count)
		}
		if execCount > 0 && sumExec.Valid {
			js.AvgExec = sumExec.Float64 / float64(execCount)
		}
		perModel[modelID] = js
		gCount += count
		gExecCount += execCount
		gSumTotal += sumTotal
		if sumExec.Valid {
			gSumExec += sumExec.Float64
		}
	}
	if gCount > 0 {
		global.Count = gCount
		global.AvgTotal = gSumTotal / float64(gCount)
	}
	if gExecCount > 0 {
		global.AvgExec = gSumExec / float64(gExecCount)
	}
	return perModel, global, rows.Err()
}

// actionEMAAlpha weights the newest completed-action duration when folding it
// into a model's rolling average. 0.2 keeps the average stable against outliers
// while still tracking genuine drift in a model's per-action cost over time.
const actionEMAAlpha = 0.2

// RecordActionDuration folds one completed action's real execution time into the
// persisted per-model exponential moving average (model_stats). The first sample
// for a model seeds the average outright; subsequent samples blend with
// actionEMAAlpha. Non-positive durations are ignored (nothing to learn from a
// zero/negative measurement). This is the sole writer of model_stats and is
// called only from the scheduler's successful-completion paths, so the average
// always reflects real timing data and never a mock/estimate.
func (s *Store) RecordActionDuration(modelID string, seconds float64) error {
	if modelID == "" || seconds <= 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	var avg float64
	var samples int
	err := s.db.QueryRow(
		"SELECT avg_action_seconds, samples FROM model_stats WHERE model_id = ?", modelID,
	).Scan(&avg, &samples)
	switch {
	case err == sql.ErrNoRows:
		avg = seconds
		samples = 1
	case err != nil:
		return err
	default:
		avg = actionEMAAlpha*seconds + (1-actionEMAAlpha)*avg
		samples++
	}

	_, err = s.db.Exec(
		`INSERT INTO model_stats (model_id, avg_action_seconds, samples, updated_at)
		 VALUES (?,?,?,?)
		 ON CONFLICT(model_id) DO UPDATE SET
		     avg_action_seconds = excluded.avg_action_seconds,
		     samples            = excluded.samples,
		     updated_at         = excluded.updated_at`,
		modelID, avg, samples, nowTS(),
	)
	return err
}

// ModelActionAverages returns the persisted rolling average seconds-per-action
// for every model that has recorded at least one completed action. Used by the
// /v1/ps cache to compute each active model's ETA.
func (s *Store) ModelActionAverages() (map[string]float64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query("SELECT model_id, avg_action_seconds FROM model_stats")
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	out := make(map[string]float64)
	for rows.Next() {
		var modelID string
		var avg float64
		if err := rows.Scan(&modelID, &avg); err != nil {
			return nil, err
		}
		out[modelID] = avg
	}
	return out, rows.Err()
}

// CompletedCountSince returns how many jobs for modelID reached the completed
// state with finished_at at or after the given timestamp (float epoch seconds).
// The dashboard uses this with a model's current load time to show "done since
// the model was loaded" — a counter that intentionally resets each residency.
func (s *Store) CompletedCountSince(modelID string, since float64) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var n int
	err := s.db.QueryRow(
		"SELECT COUNT(*) FROM jobs WHERE state = 'completed' AND model_id = ? AND finished_at IS NOT NULL AND finished_at >= ?",
		modelID, since,
	).Scan(&n)
	if err != nil {
		return 0, err
	}
	return n, nil
}

// GetRunningJobs returns all jobs currently in the "running" state with their model_id and started_at.
func (s *Store) GetRunningJobs() ([]*Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query(
		"SELECT id, model_id, job_type, state, priority, payload, result, error, created_at, started_at, finished_at, canonical_job_id, excluded_hosts, requested_model FROM jobs WHERE state = 'running'",
	)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var jobs []*Job
	for rows.Next() {
		j, err := scanJobFromRows(rows)
		if err != nil {
			return nil, err
		}
		jobs = append(jobs, j)
	}
	return jobs, rows.Err()
}

// GetActiveJobs returns all jobs in a non-terminal state (queued, scheduled, running, following).
func (s *Store) GetActiveJobs() ([]*Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query(
		"SELECT id, model_id, job_type, state, priority, payload, result, error, created_at, started_at, finished_at, canonical_job_id, excluded_hosts, requested_model FROM jobs WHERE state IN ('queued','scheduled','running','following')",
	)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var jobs []*Job
	for rows.Next() {
		j, err := scanJobFromRows(rows)
		if err != nil {
			return nil, err
		}
		jobs = append(jobs, j)
	}
	return jobs, rows.Err()
}
