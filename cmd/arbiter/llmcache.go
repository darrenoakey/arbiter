package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// LLMCache is a content-addressed on-disk cache for LLM chat completions.
//
// Purpose: tests (and any caller) repeatedly issue byte-for-byte identical chat
// requests. An identical request must NOT touch a model — it returns the stored
// worker result instantly and the entry's mtime is bumped so a nightly sweeper
// keeps it alive. Entries not hit within the TTL (32h) are deleted.
//
// Key = sha256 of the canonicalized request (deterministic, sorted-key JSON of
// every OUTPUT-affecting field: model, messages incl. image content, max_tokens,
// temperature, top_p, response_format, and any other field the caller sent).
// Non-semantic fields (currently "stream") are stripped before hashing so a
// streamed request and a non-streamed one with otherwise identical content share
// one entry — the streaming path replays the cached full completion as SSE.
//
// Each entry is a single JSON file named "<hash>.json" holding the full worker
// result shape ({"format","response","text",...}) so both the sync
// /v1/chat/completions handler and the async chat-completion job path can
// reconstruct their output from one cached value. Writes are atomic (temp file +
// rename) so concurrent identical calls never observe a partial file.
type LLMCache struct {
	dir string
	ttl time.Duration

	// mu serializes timestamp bumps and directory creation; the actual file I/O
	// is safe to run concurrently because writes go via temp+rename.
	mu sync.Mutex
}

// nonSemanticFields are request fields that do NOT affect model output and must
// be excluded from the cache key. "stream" is transport-only: arbiter replays a
// cached full completion as SSE, so a streamed and non-streamed request with
// identical content are the same cache entry.
var nonSemanticFields = map[string]bool{
	"stream": true,
}

// NewLLMCache builds a cache rooted at dir with the given TTL. dir is created if
// missing. A nil cache (dir == "") disables caching entirely (all methods no-op).
func NewLLMCache(dir string, ttl time.Duration) *LLMCache {
	if dir == "" {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		slog.Error("llmcache: cannot create cache dir; caching disabled", "dir", dir, "error", err)
		return nil
	}
	return &LLMCache{dir: dir, ttl: ttl}
}

// canonicalJSON re-encodes v with recursively sorted object keys so that two
// semantically identical requests (differing only in key order or whitespace)
// produce identical bytes. Go's encoding/json already sorts map[string]any keys
// on marshal, so decoding into any and re-marshaling is sufficient and stable.
func canonicalJSON(v any) ([]byte, error) {
	// Round-trip through any so all nested objects become map[string]any, which
	// json.Marshal emits with sorted keys.
	normalized := sortValue(v)
	return json.Marshal(normalized)
}

// sortValue is a defensive no-op transform that leaves values as-is; json.Marshal
// sorts map keys itself. It exists so the canonicalization intent is explicit and
// so future non-map ordering rules (e.g. within arrays) have one home.
func sortValue(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			out[k] = sortValue(t[k])
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, e := range t {
			out[i] = sortValue(e)
		}
		return out
	default:
		return v
	}
}

// Key computes the content-addressed cache key (sha256 hex) for a raw chat
// request body. It decodes the body, strips non-semantic fields, canonicalizes
// the remainder deterministically, and hashes. The model name is part of the
// body ("model" field) so it is folded into the key automatically. A "llm:"
// prefix on the model field is normalized to the bare name before hashing so
// canonical and prefixed forms share a key. Callers resolving aliases must
// rewrite the body to the canonical bare name before calling Key; this
// function only normalizes the prefix.
func (c *LLMCache) Key(body []byte) (string, error) {
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return "", fmt.Errorf("llmcache: request body not JSON: %w", err)
	}
	for f := range nonSemanticFields {
		delete(req, f)
	}
	if model, ok := req["model"].(string); ok {
		req["model"] = strings.TrimPrefix(model, "llm:")
	}
	canon, err := canonicalJSON(req)
	if err != nil {
		return "", fmt.Errorf("llmcache: canonicalize: %w", err)
	}
	sum := sha256.Sum256(canon)
	return hex.EncodeToString(sum[:]), nil
}

func (c *LLMCache) path(key string) string {
	return filepath.Join(c.dir, key+".json")
}

// Get returns the cached worker result for key, or (nil, false) on a miss. On a
// hit it explicitly bumps the file's mtime (os.Chtimes) to now — atime is
// unreliable on modern filesystems, so the sweeper keys on mtime and every hit
// must refresh it. A corrupt/unreadable entry is treated as a miss.
func (c *LLMCache) Get(key string) (json.RawMessage, bool) {
	if c == nil {
		return nil, false
	}
	p := c.path(key)
	data, err := os.ReadFile(p)
	if err != nil {
		return nil, false
	}
	if !json.Valid(data) {
		// Defensive: a truncated/garbage file (should be impossible with atomic
		// writes) is a miss, and we drop it so it gets rewritten cleanly.
		slog.Warn("llmcache: dropping invalid cache entry", "key", key)
		if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
			slog.Warn("llmcache: remove invalid entry failed", "key", key, "error", err)
		}
		return nil, false
	}
	now := time.Now()
	if err := os.Chtimes(p, now, now); err != nil {
		slog.Warn("llmcache: chtimes on hit failed", "key", key, "error", err)
	}
	return json.RawMessage(data), true
}

// Put atomically writes result under key. It writes a uniquely-named temp file in
// the cache dir and renames it into place, so parallel identical calls never
// corrupt an entry (last writer wins, and every reader sees a whole file). Empty
// results are refused by the caller (Put itself does not police content, but is
// only ever invoked for successful non-empty completions).
func (c *LLMCache) Put(key string, result json.RawMessage) error {
	if c == nil {
		return nil
	}
	if len(result) == 0 || !json.Valid(result) {
		return fmt.Errorf("llmcache: refusing to store empty/invalid result for %s", key)
	}
	c.mu.Lock()
	_ = os.MkdirAll(c.dir, 0o755)
	c.mu.Unlock()

	tmp, err := os.CreateTemp(c.dir, key+".tmp-*")
	if err != nil {
		return fmt.Errorf("llmcache: create temp: %w", err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(result); err != nil {
		if closeErr := tmp.Close(); closeErr != nil {
			slog.Warn("llmcache: close failed temp", "path", tmpName, "error", closeErr)
		}
		if removeErr := os.Remove(tmpName); removeErr != nil && !os.IsNotExist(removeErr) {
			slog.Warn("llmcache: remove failed temp", "path", tmpName, "error", removeErr)
		}
		return fmt.Errorf("llmcache: write temp: %w", err)
	}
	if err := tmp.Close(); err != nil {
		if removeErr := os.Remove(tmpName); removeErr != nil && !os.IsNotExist(removeErr) {
			slog.Warn("llmcache: remove unclosed temp", "path", tmpName, "error", removeErr)
		}
		return fmt.Errorf("llmcache: close temp: %w", err)
	}
	if err := os.Rename(tmpName, c.path(key)); err != nil {
		if removeErr := os.Remove(tmpName); removeErr != nil && !os.IsNotExist(removeErr) {
			slog.Warn("llmcache: remove unrenamed temp", "path", tmpName, "error", removeErr)
		}
		return fmt.Errorf("llmcache: rename: %w", err)
	}
	return nil
}

// Sweep deletes every cache file whose mtime is older than the TTL and returns
// the number deleted. Files hit within the TTL (whose mtime Get bumped) survive.
func (c *LLMCache) Sweep() (int, error) {
	if c == nil {
		return 0, nil
	}
	entries, err := os.ReadDir(c.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	cutoff := time.Now().Add(-c.ttl)
	deleted := 0
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if filepath.Ext(name) != ".json" {
			continue // skip stray temp files etc.
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if info.ModTime().Before(cutoff) {
			if err := os.Remove(filepath.Join(c.dir, name)); err == nil {
				deleted++
			}
		}
	}
	return deleted, nil
}

// RunSweeper sweeps once immediately (so a restart doesn't accumulate stale
// entries) and then daily. It blocks until ctx is done; run it in a goroutine.
func (c *LLMCache) RunSweeper(stop <-chan struct{}) {
	if c == nil {
		return
	}
	sweep := func() {
		n, err := c.Sweep()
		if err != nil {
			slog.Warn("llmcache: sweep error", "error", err)
			return
		}
		if n > 0 {
			slog.Info("llmcache: swept stale entries", "deleted", n, "ttl_hours", c.ttl.Hours())
		} else {
			slog.Info("llmcache: sweep found nothing to delete", "ttl_hours", c.ttl.Hours())
		}
	}
	sweep() // startup sweep
	ticker := time.NewTicker(24 * time.Hour)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			sweep()
		}
	}
}
