package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// sampleResult builds a minimal worker-shaped chat result with the given content.
func sampleResult(content string) json.RawMessage {
	resp := map[string]any{
		"id":     "chatcmpl-x",
		"object": "chat.completion",
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": content},
			"finish_reason": "stop",
		}},
	}
	respBytes, _ := json.Marshal(resp)
	out := map[string]any{
		"format":   "json",
		"response": json.RawMessage(respBytes),
		"text":     content,
	}
	b, _ := json.Marshal(out)
	return b
}

func TestLLMCacheKeyDeterministicAndCanonical(t *testing.T) {
	c := NewLLMCache(t.TempDir(), 32*time.Hour)

	// Same content, different key order + whitespace → same key.
	a := []byte(`{"model":"qwen","messages":[{"role":"user","content":"hi"}],"temperature":0.2,"max_tokens":50}`)
	b := []byte(`{  "max_tokens":50,"temperature":0.2,  "messages":[{"content":"hi","role":"user"}], "model":"qwen" }`)
	ka, err := c.Key(a)
	if err != nil {
		t.Fatalf("Key(a): %v", err)
	}
	kb, err := c.Key(b)
	if err != nil {
		t.Fatalf("Key(b): %v", err)
	}
	if ka != kb {
		t.Fatalf("canonicalization failed: %s != %s", ka, kb)
	}

	// Different model → different key.
	kc, _ := c.Key([]byte(`{"model":"gemma","messages":[{"role":"user","content":"hi"}],"temperature":0.2,"max_tokens":50}`))
	if kc == ka {
		t.Fatal("different model must produce different key")
	}

	// Different temperature → different key.
	kd, _ := c.Key([]byte(`{"model":"qwen","messages":[{"role":"user","content":"hi"}],"temperature":0.9,"max_tokens":50}`))
	if kd == ka {
		t.Fatal("different temperature must produce different key")
	}
}

func TestLLMCacheStreamFlagStrippedFromKey(t *testing.T) {
	c := NewLLMCache(t.TempDir(), 32*time.Hour)
	plain := []byte(`{"model":"qwen","messages":[{"role":"user","content":"hi"}]}`)
	streamed := []byte(`{"model":"qwen","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	kp, _ := c.Key(plain)
	ks, _ := c.Key(streamed)
	if kp != ks {
		t.Fatalf("stream flag must not affect key: %s != %s", kp, ks)
	}
}

func TestLLMCacheKeyRejectsNonJSON(t *testing.T) {
	c := NewLLMCache(t.TempDir(), 32*time.Hour)
	if _, err := c.Key([]byte("not json")); err == nil {
		t.Fatal("expected error for non-JSON body")
	}
}

func TestLLMCacheHitAndMiss(t *testing.T) {
	c := NewLLMCache(t.TempDir(), 32*time.Hour)
	key := "deadbeef"

	if _, ok := c.Get(key); ok {
		t.Fatal("expected miss on empty cache")
	}
	res := sampleResult("hello")
	if err := c.Put(key, res); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, ok := c.Get(key)
	if !ok {
		t.Fatal("expected hit after Put")
	}
	if string(got) != string(res) {
		t.Fatalf("hit returned wrong bytes:\n got %s\nwant %s", got, res)
	}
}

func TestLLMCacheMtimeBumpedOnHit(t *testing.T) {
	dir := t.TempDir()
	c := NewLLMCache(dir, 32*time.Hour)
	key := "abc123"
	if err := c.Put(key, sampleResult("x")); err != nil {
		t.Fatalf("Put: %v", err)
	}
	p := filepath.Join(dir, key+".json")

	// Backdate the file well into the past.
	old := time.Now().Add(-10 * time.Hour)
	if err := os.Chtimes(p, old, old); err != nil {
		t.Fatalf("chtimes: %v", err)
	}
	beforeInfo, _ := os.Stat(p)
	if time.Since(beforeInfo.ModTime()) < 9*time.Hour {
		t.Fatal("precondition: file should be backdated")
	}

	if _, ok := c.Get(key); !ok {
		t.Fatal("expected hit")
	}
	afterInfo, _ := os.Stat(p)
	if time.Since(afterInfo.ModTime()) > time.Minute {
		t.Fatalf("mtime not bumped on hit: mtime is %v old", time.Since(afterInfo.ModTime()))
	}
}

func TestLLMCacheRefusesEmptyOrInvalid(t *testing.T) {
	c := NewLLMCache(t.TempDir(), 32*time.Hour)
	if err := c.Put("k", json.RawMessage(``)); err == nil {
		t.Fatal("expected refusal of empty result")
	}
	if err := c.Put("k", json.RawMessage(`{bad json`)); err == nil {
		t.Fatal("expected refusal of invalid JSON")
	}
	if _, ok := c.Get("k"); ok {
		t.Fatal("nothing should have been stored")
	}
}

func TestLLMCacheSweeperDeletesOldKeepsFresh(t *testing.T) {
	dir := t.TempDir()
	c := NewLLMCache(dir, 32*time.Hour)

	// Fresh entry (now).
	if err := c.Put("fresh", sampleResult("f")); err != nil {
		t.Fatalf("Put fresh: %v", err)
	}
	// Old entry (backdated 40h > 32h TTL).
	if err := c.Put("stale", sampleResult("s")); err != nil {
		t.Fatalf("Put stale: %v", err)
	}
	old := time.Now().Add(-40 * time.Hour)
	if err := os.Chtimes(filepath.Join(dir, "stale.json"), old, old); err != nil {
		t.Fatalf("chtimes: %v", err)
	}
	// Boundary entry just inside TTL (31h) must survive.
	if err := c.Put("boundary", sampleResult("b")); err != nil {
		t.Fatalf("Put boundary: %v", err)
	}
	inside := time.Now().Add(-31 * time.Hour)
	if err := os.Chtimes(filepath.Join(dir, "boundary.json"), inside, inside); err != nil {
		t.Fatalf("chtimes boundary: %v", err)
	}

	n, err := c.Sweep()
	if err != nil {
		t.Fatalf("Sweep: %v", err)
	}
	if n != 1 {
		t.Fatalf("expected 1 deleted, got %d", n)
	}
	if _, ok := c.Get("fresh"); !ok {
		t.Fatal("fresh entry should survive")
	}
	if _, ok := c.Get("boundary"); !ok {
		t.Fatal("boundary (31h) entry should survive")
	}
	if _, err := os.Stat(filepath.Join(dir, "stale.json")); !os.IsNotExist(err) {
		t.Fatal("stale entry should have been deleted")
	}
}

func TestLLMCacheAtomicWriteUnderConcurrency(t *testing.T) {
	dir := t.TempDir()
	c := NewLLMCache(dir, 32*time.Hour)
	key := "concurrent"

	var wg sync.WaitGroup
	const n = 50
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// Every writer stores a fully-valid result. Interleaved temp+rename
			// must never leave a partial file — a concurrent reader must always
			// get whole, valid JSON.
			if err := c.Put(key, sampleResult(fmt.Sprintf("answer-%d", i))); err != nil {
				t.Errorf("Put: %v", err)
			}
			if got, ok := c.Get(key); ok {
				if !json.Valid(got) {
					t.Errorf("reader saw non-atomic (invalid) JSON: %s", got)
				}
			}
		}(i)
	}
	wg.Wait()

	got, ok := c.Get(key)
	if !ok {
		t.Fatal("expected final entry present")
	}
	if !json.Valid(got) {
		t.Fatalf("final entry corrupt: %s", got)
	}
	// No stray temp files left behind.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if filepath.Ext(e.Name()) != ".json" {
			t.Fatalf("stray non-json file left in cache dir: %s", e.Name())
		}
	}
}

func TestLLMCacheDisabledNoOp(t *testing.T) {
	var c *LLMCache // nil == disabled
	if _, ok := c.Get("x"); ok {
		t.Fatal("nil cache Get must miss")
	}
	if err := c.Put("x", sampleResult("y")); err != nil {
		t.Fatalf("nil cache Put must be no-op, got %v", err)
	}
	if n, err := c.Sweep(); err != nil || n != 0 {
		t.Fatalf("nil cache Sweep must be no-op, got %d %v", n, err)
	}
}

func TestChatResultHasContent(t *testing.T) {
	if !chatResultHasContent(sampleResult("real answer")) {
		t.Fatal("non-empty content must be cacheable")
	}
	if chatResultHasContent(sampleResult("")) {
		t.Fatal("empty content must NOT be cacheable")
	}
	// Malformed result → not cacheable.
	if chatResultHasContent(json.RawMessage(`{"format":"json"}`)) {
		t.Fatal("result with no choices must NOT be cacheable")
	}
}

func TestExtractCachedResponse(t *testing.T) {
	res := sampleResult("hi there")
	inner := extractCachedResponse(res)
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(inner, &parsed); err != nil {
		t.Fatalf("extracted response not valid OpenAI JSON: %v", err)
	}
	if len(parsed.Choices) == 0 || parsed.Choices[0].Message.Content != "hi there" {
		t.Fatalf("extracted response missing content: %s", inner)
	}
}

func TestNewLLMCacheEmptyDirDisables(t *testing.T) {
	if c := NewLLMCache("", 32*time.Hour); c != nil {
		t.Fatal("empty dir must yield a nil (disabled) cache")
	}
}

func TestLLMCacheConfigDefaults(t *testing.T) {
	cfg := &Config{}
	if !cfg.LLMCacheEnabledOrDefault() {
		t.Fatal("cache must be ON by default")
	}
	if cfg.LLMCacheTTL() != 32*time.Hour {
		t.Fatalf("default TTL must be 32h, got %v", cfg.LLMCacheTTL())
	}
	cfg.LLMCacheDisabled = true
	if cfg.LLMCacheEnabledOrDefault() {
		t.Fatal("disabled flag must turn cache off")
	}
	cfg2 := &Config{LLMCacheTTLHours: 10}
	if cfg2.LLMCacheTTL() != 10*time.Hour {
		t.Fatalf("custom TTL not honored, got %v", cfg2.LLMCacheTTL())
	}
}
