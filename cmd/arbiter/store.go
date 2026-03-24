package main

import (
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
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
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority) WHERE state = 'queued';
CREATE INDEX IF NOT EXISTS idx_jobs_model ON jobs(model_id);
`

type Store struct {
	db *sql.DB
	mu sync.Mutex
}

func NewStore(dbPath string) (*Store, error) {
	db, err := sql.Open("sqlite", dbPath+"?_journal_mode=WAL&_synchronous=NORMAL&_busy_timeout=5000")
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	db.SetMaxOpenConns(1) // SQLite doesn't support concurrent writers
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
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.scanJob(s.db.QueryRow("SELECT * FROM jobs WHERE id = ?", id))
}

func (s *Store) scanJob(row *sql.Row) (*Job, error) {
	var j Job
	var payload, result, errStr sql.NullString
	var startedAt, finishedAt sql.NullFloat64
	err := row.Scan(&j.ID, &j.ModelID, &j.JobType, &j.State, &j.Priority,
		&payload, &result, &errStr, &j.CreatedAt, &startedAt, &finishedAt)
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
	return &j, nil
}

func (s *Store) ListJobs(state, modelID string, limit int) ([]*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

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
		var j Job
		var payload, result, errStr sql.NullString
		var startedAt, finishedAt sql.NullFloat64
		if err := rows.Scan(&j.ID, &j.ModelID, &j.JobType, &j.State, &j.Priority,
			&payload, &result, &errStr, &j.CreatedAt, &startedAt, &finishedAt); err != nil {
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
		jobs = append(jobs, &j)
	}
	return jobs, nil
}

func (s *Store) PickNextJob(excludeModels map[string]bool) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	query := "SELECT * FROM jobs WHERE state = 'queued'"
	var args []any
	if len(excludeModels) > 0 {
		for m := range excludeModels {
			query += " AND model_id != ?"
			args = append(args, m)
		}
	}
	query += " ORDER BY priority ASC, created_at ASC LIMIT 1"

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
}

func WithStartedAt(t float64) func(*stateUpdate)  { return func(u *stateUpdate) { u.startedAt = &t } }
func WithFinishedAt(t float64) func(*stateUpdate) { return func(u *stateUpdate) { u.finishedAt = &t } }
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
	s.mu.Lock()
	defer s.mu.Unlock()

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

func (s *Store) CountActive(modelID string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
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
		"UPDATE jobs SET state = 'cancelled', finished_at = ? WHERE id = ? AND state IN ('queued','scheduled')",
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

func (s *Store) Close() {
	s.db.Close()
}

// GetJobs fetches multiple jobs by ID in a single query.
// Returns a map of jobID -> Job. Missing IDs are omitted.
func (s *Store) GetJobs(ids []string) (map[string]*Job, error) {
	if len(ids) == 0 {
		return map[string]*Job{}, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

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
		var j Job
		var payload, res, errStr sql.NullString
		var startedAt, finishedAt sql.NullFloat64
		if err := rows.Scan(&j.ID, &j.ModelID, &j.JobType, &j.State, &j.Priority,
			&payload, &res, &errStr, &j.CreatedAt, &startedAt, &finishedAt); err != nil {
			return nil, err
		}
		if payload.Valid {
			j.Payload = json.RawMessage(payload.String)
		}
		if res.Valid {
			rm := json.RawMessage(res.String)
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
		result[j.ID] = &j
	}
	return result, nil
}
