package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRelocateJobOutput_NoShareIsNoop(t *testing.T) {
	tmp := t.TempDir()
	local := filepath.Join(tmp, "local", "jobs", "abc")
	if err := os.MkdirAll(local, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(local, "result.txt"), []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{ShareMount: ""}
	got := relocateJobOutput(cfg, "abc", local)
	if got != local {
		t.Fatalf("expected no-op (return localDir) when ShareMount empty, got %q", got)
	}
	if _, err := os.Stat(filepath.Join(local, "result.txt")); err != nil {
		t.Fatalf("local file should still exist: %v", err)
	}
}

func TestRelocateJobOutput_MovesToShare(t *testing.T) {
	tmp := t.TempDir()
	share := filepath.Join(tmp, "share")
	local := filepath.Join(tmp, "local", "jobs", "abc")
	if err := os.MkdirAll(local, 0o755); err != nil {
		t.Fatal(err)
	}
	payload := []byte("frame-data")
	if err := os.WriteFile(filepath.Join(local, "result.mp4"), payload, 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{ShareMount: share}
	got := relocateJobOutput(cfg, "abc", local)
	want := filepath.Join(share, "output", "jobs", "abc")
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	moved, err := os.ReadFile(filepath.Join(want, "result.mp4"))
	if err != nil {
		t.Fatalf("file should exist at share: %v", err)
	}
	if string(moved) != string(payload) {
		t.Fatalf("payload mismatch")
	}
	if _, err := os.Stat(local); !os.IsNotExist(err) {
		t.Fatalf("local dir should be removed, stat err=%v", err)
	}
}

func TestRelocateJobOutput_MissingLocalDirIsNoop(t *testing.T) {
	tmp := t.TempDir()
	cfg := &Config{ShareMount: filepath.Join(tmp, "share")}
	missing := filepath.Join(tmp, "nope", "jobs", "abc")
	got := relocateJobOutput(cfg, "abc", missing)
	if got != missing {
		t.Fatalf("expected localDir return for missing dir, got %q", got)
	}
}

func TestResolveJobDir_PrefersShareWhenPresent(t *testing.T) {
	tmp := t.TempDir()
	share := filepath.Join(tmp, "share")
	local := filepath.Join(tmp, "local")
	shareJob := filepath.Join(share, "output", "jobs", "abc")
	if err := os.MkdirAll(shareJob, 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{ShareMount: share}
	got := resolveJobDir(cfg, local, "abc")
	if got != shareJob {
		t.Fatalf("expected share path %q, got %q", shareJob, got)
	}
}

func TestResolveJobDir_FallsBackToLocal(t *testing.T) {
	tmp := t.TempDir()
	cfg := &Config{ShareMount: filepath.Join(tmp, "share")} // share dir doesn't exist for this job
	got := resolveJobDir(cfg, filepath.Join(tmp, "local"), "abc")
	want := filepath.Join(tmp, "local", "jobs", "abc")
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestRewriteResultPaths_RewritesNestedStrings(t *testing.T) {
	in := json.RawMessage(`{"format":"mp4","file":"result.mp4","paths":["/old/x","/old/y"],"meta":{"src":"/old/in.png","other":"untouched"}}`)
	out := rewriteResultPaths(in, "/old", "/new")
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	paths := got["paths"].([]any)
	if paths[0].(string) != "/new/x" || paths[1].(string) != "/new/y" {
		t.Fatalf("paths not rewritten: %v", paths)
	}
	meta := got["meta"].(map[string]any)
	if meta["src"].(string) != "/new/in.png" {
		t.Fatalf("meta.src not rewritten: %v", meta["src"])
	}
	if meta["other"].(string) != "untouched" {
		t.Fatalf("unrelated string changed: %v", meta["other"])
	}
}

func TestRewriteResultPaths_NoOpOnEmptyPrefix(t *testing.T) {
	in := json.RawMessage(`{"x":"/old/y"}`)
	out := rewriteResultPaths(in, "", "/new")
	if !strings.Contains(string(out), "/old/y") {
		t.Fatalf("expected unchanged, got %s", out)
	}
}
