package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

// DedupCache provides job deduplication based on input hashing.
// Jobs with identical inputs within the TTL window are deduplicated:
// - Completed original → new job instantly completes with cached result
// - In-flight original → new "follower" job waits for original
// - Failed/cancelled original → treated as cache miss

// computeJobHash produces a SHA256 hash of the job type + canonical params.
// For *_file params, hashes file contents instead of the path.
func computeJobHash(jobType string, params json.RawMessage) string {
	h := sha256.New()
	h.Write([]byte(jobType))
	h.Write([]byte{0}) // separator

	// Parse params to handle file content hashing
	var paramMap map[string]any
	if err := json.Unmarshal(params, &paramMap); err != nil {
		// Can't parse — just hash raw JSON
		h.Write(params)
		return hex.EncodeToString(h.Sum(nil))
	}

	// Replace file paths with content hashes
	canonicalized := canonicalizeParams(paramMap)

	// Canonical JSON (sorted keys)
	canonical, _ := json.Marshal(canonicalized)
	h.Write(canonical)

	return hex.EncodeToString(h.Sum(nil))
}

// canonicalizeParams replaces *_file values with file content hashes
// and sorts all keys for consistent hashing.
func canonicalizeParams(params map[string]any) map[string]any {
	result := make(map[string]any)

	// Get sorted keys
	keys := make([]string, 0, len(params))
	for k := range params {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		v := params[k]

		// Hash file contents for *_file params
		if strings.HasSuffix(k, "_file") {
			if path, ok := v.(string); ok && path != "" {
				contentHash := hashFileContents(path)
				if contentHash != "" {
					result[k] = "filehash:" + contentHash
					continue
				}
			}
		}

		// Recursively handle nested maps
		if nested, ok := v.(map[string]any); ok {
			result[k] = canonicalizeParams(nested)
			continue
		}

		// Recursively handle arrays
		if arr, ok := v.([]any); ok {
			newArr := make([]any, len(arr))
			for i, item := range arr {
				if m, ok := item.(map[string]any); ok {
					newArr[i] = canonicalizeParams(m)
				} else {
					newArr[i] = item
				}
			}
			result[k] = newArr
			continue
		}

		result[k] = v
	}
	return result
}

func hashFileContents(path string) string {
	// Handle ref: prefix
	if strings.HasPrefix(path, "ref:") {
		// ref files are in output/refs/ — resolve later
		// For now, just hash the ref ID (it's a content-addressed name)
		return "ref:" + path
	}

	f, err := os.Open(path)
	if err != nil {
		return "" // file not accessible, use path as-is
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return ""
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Store methods for dedup cache

const dedupSchema = `
CREATE TABLE IF NOT EXISTS dedup_cache (
    hash TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dedup_created ON dedup_cache(created_at);
`

// InitDedup creates the dedup table if needed.
func (s *Store) InitDedup() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.db.Exec(dedupSchema)
}

// DedupLookup checks for an existing job with the same input hash.
// Returns the job_id if found within TTL, or empty string.
func (s *Store) DedupLookup(hash string, ttlSeconds float64) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	cutoff := nowTS() - ttlSeconds
	var jobID string
	err := s.db.QueryRow(
		"SELECT job_id FROM dedup_cache WHERE hash = ? AND created_at > ?",
		hash, cutoff,
	).Scan(&jobID)
	if err != nil {
		return "", err
	}
	return jobID, nil
}

// DedupRegister stores a hash → job_id mapping.
func (s *Store) DedupRegister(hash, jobID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.db.Exec(
		"INSERT OR REPLACE INTO dedup_cache (hash, job_id, created_at) VALUES (?, ?, ?)",
		hash, jobID, nowTS(),
	)
}

// DedupCleanup removes entries older than ttlSeconds.
func (s *Store) DedupCleanup(ttlSeconds float64) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	cutoff := nowTS() - ttlSeconds
	res, _ := s.db.Exec("DELETE FROM dedup_cache WHERE created_at < ?", cutoff)
	n, _ := res.RowsAffected()
	return int(n)
}

// GetFollowers returns all job IDs that are following a given original job.
func (s *Store) GetFollowers(originalJobID string) ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rows, err := s.db.Query(
		"SELECT id FROM jobs WHERE state = 'following' AND error = ?",
		"following:"+originalJobID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		rows.Scan(&id)
		ids = append(ids, id)
	}
	return ids, nil
}

// CreateFollowerJob creates a job in "following" state linked to the original.
func (s *Store) CreateFollowerJob(modelID, jobType string, payload json.RawMessage, originalJobID string) (*Job, error) {
	id := genID()
	now := nowTS()
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.db.Exec(
		"INSERT INTO jobs (id, model_id, job_type, state, priority, payload, created_at, error) VALUES (?,?,?,'following',0,?,?,?)",
		id, modelID, jobType, string(payload), now, "following:"+originalJobID,
	)
	if err != nil {
		return nil, err
	}
	return &Job{
		ID: id, ModelID: modelID, JobType: jobType,
		State: "following", Payload: payload, CreatedAt: now,
		Error: "following:" + originalJobID,
	}, nil
}

// ResolveFollowers handles completion of an original job:
// - If original completed: all followers complete with same result
// - If original failed/cancelled: followers become fresh queued jobs
func (s *Store) ResolveFollowers(originalJobID string, originalState string, result *json.RawMessage, errMsg string, outputDir string) int {
	followers, _ := s.GetFollowers(originalJobID)
	if len(followers) == 0 {
		return 0
	}

	now := nowTS()
	for _, fid := range followers {
		if originalState == "completed" {
			// Symlink output directory
			origDir := fmt.Sprintf("%s/jobs/%s", outputDir, originalJobID)
			followerDir := fmt.Sprintf("%s/jobs/%s", outputDir, fid)
			os.Symlink(origDir, followerDir)

			s.UpdateState(fid, "completed", WithResult(*result), WithFinishedAt(now))
		} else {
			// Original failed/cancelled — promote follower to a real queued job
			s.mu.Lock()
			s.db.Exec("UPDATE jobs SET state = 'queued', error = NULL, priority = 0 WHERE id = ?", fid)
			s.mu.Unlock()
		}
	}
	return len(followers)
}

// DedupRecoveredJobs scans all queued jobs after crash recovery,
// populates the dedup cache, and cancels duplicate jobs on the queue.
func (s *Store) DedupRecoveredJobs() int {
	s.mu.Lock()
	rows, err := s.db.Query(
		"SELECT id, job_type, payload FROM jobs WHERE state = 'queued' ORDER BY created_at ASC",
	)
	s.mu.Unlock()
	if err != nil {
		return 0
	}
	defer rows.Close()

	type queuedJob struct {
		id      string
		jobType string
		payload json.RawMessage
	}
	var jobs []queuedJob
	for rows.Next() {
		var j queuedJob
		var payload string
		rows.Scan(&j.id, &j.jobType, &payload)
		j.payload = json.RawMessage(payload)
		jobs = append(jobs, j)
	}

	seen := make(map[string]string) // hash -> first job_id
	removed := 0
	now := nowTS()

	for _, j := range jobs {
		hash := computeJobHash(j.jobType, j.payload)
		if firstID, exists := seen[hash]; exists {
			// Duplicate — cancel it
			s.mu.Lock()
			s.db.Exec(
				"UPDATE jobs SET state = 'cancelled', finished_at = ?, error = ? WHERE id = ?",
				now, "dedup: duplicate of "+firstID, j.id,
			)
			s.mu.Unlock()
			removed++
		} else {
			seen[hash] = j.id
			s.DedupRegister(hash, j.id)
		}
	}
	return removed
}
