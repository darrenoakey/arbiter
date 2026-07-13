package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

// localOllama is the real ollama endpoint the Phase-2 remote tests dispatch
// against. Phase 0 set up llama3.2:3b on this Mac's local ollama; it is the
// small, fast, reliable model the spec mandates (NOT gemma — that would contend
// with other work and is slow). Host-absence is simulated with an unreachable
// addr (deadOllamaAddr), so one real endpoint + one dead addr covers routing +
// failover without depending on the flaky this-MBP backup daemon.
const (
	localOllamaAddr = "http://127.0.0.1:11434"
	localOllamaTag  = "llama3.2:3b"
)

// reachableOllama reports whether the local ollama serves the test model. Tests
// skip (not fail) when it isn't up, so the suite stays runnable on a box without
// ollama; on this dev Mac it runs for real.
func reachableOllama(t *testing.T) bool {
	t.Helper()
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(localOllamaAddr + "/api/tags")
	if err != nil {
		t.Logf("local ollama not reachable (%v) — skipping real-remote test", err)
		return false
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			t.Logf("close local ollama response: %v", err)
		}
	}()
	if resp.StatusCode != 200 {
		t.Logf("local ollama /api/tags returned %d — skipping", resp.StatusCode)
		return false
	}
	var tags struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		t.Logf("decode local ollama tags: %v", err)
		return false
	}
	for _, m := range tags.Models {
		if m.Name == localOllamaTag {
			return true
		}
	}
	t.Logf("local ollama present but %s not pulled — skipping", localOllamaTag)
	return false
}

// deadOllamaAddr returns an addr that will refuse/never-route — a TCP listener
// bound and immediately closed leaves a port nothing listens on, so a dial gets
// connection-refused (a CONFIRMED-absence INFRA error). OS-assigned port avoids
// collisions.
func deadOllamaAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve dead port: %v", err)
	}
	addr := ln.Addr().String()
	if err := ln.Close(); err != nil { // nothing listens now → dial refused
		t.Fatalf("close dead-port listener: %v", err)
	}
	return "http://" + addr
}

func newRemoteTestScheduler(t *testing.T, cfg *Config) (*Scheduler, *Store, *InstanceManager, func()) {
	t.Helper()
	projectRoot := t.TempDir()
	outputDir := filepath.Join(projectRoot, "output")
	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	store.InitDedup()
	logger := NewEventLogger(filepath.Join(outputDir, "logs"))
	mgr := NewInstanceManager(cfg, "python3", projectRoot)
	setupInstances(cfg, mgr, "python3", projectRoot)
	sched := NewScheduler(cfg, store, mgr, logger, outputDir)
	cleanup := func() {
		store.Close()
		logger.Close()
	}
	return sched, store, mgr, cleanup
}

func chatPayload(prompt string) json.RawMessage {
	body, _ := json.Marshal(map[string]any{
		"messages":   []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens": 32,
	})
	return body
}

// waitForState polls the store until the job reaches one of the wanted states or
// the deadline elapses.
func waitForState(t *testing.T, store *Store, jobID string, deadline time.Duration, want ...string) *Job {
	t.Helper()
	end := time.Now().Add(deadline)
	for time.Now().Before(end) {
		j, _ := store.GetJob(jobID)
		if j != nil {
			for _, w := range want {
				if j.State == w {
					return j
				}
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	j, _ := store.GetJob(jobID)
	if j != nil {
		t.Fatalf("job %s did not reach %v within %s (state=%s error=%q)", jobID, want, deadline, j.State, j.Error)
	}
	t.Fatalf("job %s vanished waiting for %v", jobID, want)
	return nil
}

func pi() *float64 { p := 1.0; return &p }

// (a) A chat job routed to a REMOTE instance returns a real completion.
func TestRemoteDispatchReturnsRealCompletion(t *testing.T) {
	if !reachableOllama(t) {
		t.Skip("local ollama unavailable")
	}
	cfg := &Config{
		VRAMBudgetGB: 100,
		Hosts: map[string]HostConfig{
			"mac": {Addr: localOllamaAddr, Kind: "mlx", BudgetGB: 64},
		},
		Models: map[string]ModelConfig{
			"chat": {
				MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
				Placements:    []string{"mac"},
				AdapterParams: map[string]string{"remote_model_tag": localOllamaTag},
			},
		},
	}
	sched, store, mgr, cleanup := newRemoteTestScheduler(t, cfg)
	defer cleanup()

	// The single instance must be remote.
	inst := mgr.GetModelInstances("chat")[0]
	if !inst.isRemote() {
		t.Fatalf("expected remote instance, host=%s", inst.host)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go sched.Run(ctx)

	job, err := store.CreateJob("chat", "chat-completion", chatPayload("Reply with exactly the word PONG."), 1)
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	sched.Wake()

	done := waitForState(t, store, job.ID, 90*time.Second, "completed", "failed")
	if done.State != "completed" {
		t.Fatalf("job state=%s error=%q, want completed", done.State, done.Error)
	}
	var result map[string]any
	if err := json.Unmarshal(*done.Result, &result); err != nil {
		t.Fatalf("decode completed result: %v", err)
	}
	text, _ := result["text"].(string)
	if text == "" {
		t.Fatalf("completed job had empty text; result=%s", string(*done.Result))
	}
	if _, ok := result["response"]; !ok {
		t.Fatalf("result missing full OpenAI response field; result=%s", string(*done.Result))
	}
	t.Logf("remote completion text=%q", text)
}

// (b) CONFIRMED host-absence on the preferred host → the job transparently fails
// over to a reachable endpoint, NEVER fails, exactly one result.
func TestRemoteFailoverOnAbsenceCompletesElsewhere(t *testing.T) {
	if !reachableOllama(t) {
		t.Skip("local ollama unavailable")
	}
	dead := deadOllamaAddr(t)
	cfg := &Config{
		VRAMBudgetGB: 100,
		Hosts: map[string]HostConfig{
			"dead": {Addr: dead, Kind: "mlx", BudgetGB: 64},
			"live": {Addr: localOllamaAddr, Kind: "mlx", BudgetGB: 64},
		},
		Models: map[string]ModelConfig{
			"chat": {
				MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
				// Preferred host is unreachable; failover must spill to "live".
				Placements:    []string{"dead", "live"},
				AdapterParams: map[string]string{"remote_model_tag": localOllamaTag},
			},
		},
	}
	sched, store, _, cleanup := newRemoteTestScheduler(t, cfg)
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go sched.Run(ctx)

	job, err := store.CreateJob("chat", "chat-completion", chatPayload("Reply with the word OK."), 1)
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	sched.Wake()

	done := waitForState(t, store, job.ID, 90*time.Second, "completed", "failed")
	if done.State != "completed" {
		t.Fatalf("job state=%s error=%q, want completed (failover should never fail it)", done.State, done.Error)
	}
	// The dead host must be in the durable excluded set.
	if !done.HostExcluded("dead") {
		t.Fatalf("expected 'dead' in excluded_hosts, got %v", done.ExcludedHosts)
	}
	// Exactly one result: it's completed with a result and a single finished_at.
	if done.Result == nil {
		t.Fatalf("completed job has no result")
	}
	if done.FinishedAt == nil {
		t.Fatalf("completed job has no finished_at")
	}
	var result map[string]any
	if err := json.Unmarshal(*done.Result, &result); err != nil {
		t.Fatalf("decode completed result: %v", err)
	}
	if text, _ := result["text"].(string); text == "" {
		t.Fatalf("failover completion had empty text; result=%s", string(*done.Result))
	}
	t.Logf("failover: excluded=%v final_text=%v", done.ExcludedHosts, result["text"])
}

// (c) PickInstanceForJob honors excluded_hosts and the placement order.
func TestPickInstanceForJobHonorsExcludedHosts(t *testing.T) {
	cfg := &Config{
		VRAMBudgetGB: 100,
		Hosts: map[string]HostConfig{
			"h1": {Addr: "http://10.255.255.1:11434", Kind: "mlx", BudgetGB: 64},
			"h2": {Addr: "http://10.255.255.2:11434", Kind: "mlx", BudgetGB: 64},
		},
		Models: map[string]ModelConfig{
			"chat": {
				MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
				Placements:    []string{"h1", "h2", "spark"},
			},
		},
	}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())

	// No exclusions: must pick the most-preferred host h1.
	job := &Job{ID: "j1", ModelID: "chat", State: "queued"}
	got := mgr.PickInstanceForJob(job, true)
	if got == nil || got.host != "h1" {
		t.Fatalf("no-exclusion pick host=%v, want h1", hostOf(got))
	}

	// Exclude h1: must spill to h2.
	job.ExcludedHosts = []string{"h1"}
	got = mgr.PickInstanceForJob(job, true)
	if got == nil || got.host != "h2" {
		t.Fatalf("exclude-h1 pick host=%v, want h2", hostOf(got))
	}

	// Exclude h1+h2: must fall back to spark (local, always reachable).
	job.ExcludedHosts = []string{"h1", "h2"}
	got = mgr.PickInstanceForJob(job, true)
	if got == nil || got.host != "spark" {
		t.Fatalf("exclude-both pick host=%v, want spark", hostOf(got))
	}

	// Kill-switch (remoteEnabled=false): remotes skipped, pin to spark even with
	// no exclusions.
	job.ExcludedHosts = nil
	got = mgr.PickInstanceForJob(job, false)
	if got == nil || got.host != "spark" {
		t.Fatalf("kill-switch pick host=%v, want spark", hostOf(got))
	}
}

func TestPickInstanceForJobSpillsFromBusyLocalToRemote(t *testing.T) {
	cfg := &Config{
		VRAMBudgetGB: 100,
		Hosts: map[string]HostConfig{
			"boringstack": {Addr: "http://10.255.255.1:11434", Kind: "mlx", BudgetGB: 96},
			"darrens-mbp": {Addr: "http://10.255.255.2:11434", Kind: "mlx", BudgetGB: 40},
		},
		Models: map[string]ModelConfig{
			"ltx2-dev-denoise1": {
				MemoryGB: 80, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
			},
			"llm:gemma4-26b": {
				MemoryGB: 10, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
				Placements:    []string{"spark", "boringstack", "darrens-mbp"},
			},
		},
	}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())
	mgr.SetReachabilityFunc(func(string) bool { return true })

	markLoaded(t, mgr, "ltx2-dev-denoise1")
	blocker := mgr.GetModelInstances("ltx2-dev-denoise1")[0]
	atomic.AddInt32(&blocker.activeJobs, 1)

	job := &Job{ID: "j1", ModelID: "llm:gemma4-26b", State: "queued"}
	got, reason := mgr.PickInstanceForJobWithReason(job, true)
	if got == nil || got.host != "boringstack" {
		t.Fatalf("spark-busy pick host=%v, want boringstack", hostOf(got))
	}
	if reason != reasonSpill {
		t.Fatalf("spark-busy placement reason=%q, want %q", reason, reasonSpill)
	}

	atomic.AddInt32(&got.activeJobs, 1)
	got.setState("loaded")
	got2, reason2 := mgr.PickInstanceForJobWithReason(&Job{ID: "j2", ModelID: "llm:gemma4-26b", State: "queued"}, true)
	if got2 == nil || got2.host != "darrens-mbp" {
		t.Fatalf("boringstack-full pick host=%v, want darrens-mbp", hostOf(got2))
	}
	if reason2 != reasonSpill {
		t.Fatalf("boringstack-full placement reason=%q, want %q", reason2, reasonSpill)
	}

	atomic.AddInt32(&got2.activeJobs, 1)
	got2.setState("loaded")
	got3, reason3 := mgr.PickInstanceForJobWithReason(&Job{ID: "j3", ModelID: "llm:gemma4-26b", State: "queued"}, true)
	if got3 == nil || got3.host != "spark" {
		t.Fatalf("all-remotes-full pick host=%v, want spark fallback", hostOf(got3))
	}
	if reason3 != reasonPreferred {
		t.Fatalf("all-remotes-full placement reason=%q, want %q", reason3, reasonPreferred)
	}
}

func hostOf(inst *Instance) string {
	if inst == nil {
		return "<nil>"
	}
	return inst.host
}

// (d) Idempotency: a late/duplicate response from a dead host must NOT double-
// write a result or resurrect a job that already reached a terminal state. The
// guard is in store.UpdateState (terminal-stays-terminal) — a late writer can't
// move completed→running, and a late result write to an already-completed job
// is rejected so the first result wins.
func TestLateResponseDoesNotResurrectTerminalJob(t *testing.T) {
	projectRoot := t.TempDir()
	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()
	store.InitDedup()

	job, err := store.CreateJob("chat", "chat-completion", chatPayload("hi"), 1)
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	// Simulate the job completing on the failover target (the real, first result).
	winner := json.RawMessage(`{"format":"json","text":"WINNER"}`)
	if err := store.UpdateState(job.ID, "completed", WithResult(winner), WithFinishedAt(nowTS())); err != nil {
		t.Fatalf("complete: %v", err)
	}

	// A LATE response from the originally-dispatched (dead) host tries to write a
	// second result and/or move the job back to running. Both must be rejected.
	loser := json.RawMessage(`{"format":"json","text":"LATE_LOSER"}`)
	if err := store.UpdateState(job.ID, "running"); err != nil { // terminal→active: rejected
		t.Fatalf("attempt terminal-to-running update: %v", err)
	}
	if err := store.UpdateState(job.ID, "completed", WithResult(loser)); err != nil { // overwrite attempt
		t.Fatalf("attempt terminal overwrite: %v", err)
	}
	if err := store.UpdateState(job.ID, "queued", WithClearStartedAt()); err != nil { // failover-requeue race
		t.Fatalf("attempt terminal requeue: %v", err)
	}

	final, _ := store.GetJob(job.ID)
	if final.State != "completed" {
		t.Fatalf("terminal job moved to %s — terminal-stays-terminal violated", final.State)
	}
	var result map[string]any
	if err := json.Unmarshal(*final.Result, &result); err != nil {
		t.Fatalf("decode final result: %v", err)
	}
	if result["text"] != "WINNER" {
		t.Fatalf("result overwritten by late response: got %v, want WINNER", result["text"])
	}
}

// (d-extension) AddExcludedHost is durable and idempotent — the persistence half
// of failover. A re-add is a no-op; the set survives a re-read from disk.
func TestAddExcludedHostDurableAndIdempotent(t *testing.T) {
	projectRoot := t.TempDir()
	store, err := NewStore(filepath.Join(projectRoot, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	defer store.Close()
	store.InitDedup()

	job, _ := store.CreateJob("chat", "chat-completion", chatPayload("hi"), 1)

	hosts, err := store.AddExcludedHost(job.ID, "boringstack")
	if err != nil {
		t.Fatalf("add: %v", err)
	}
	if len(hosts) != 1 || hosts[0] != "boringstack" {
		t.Fatalf("add returned %v, want [boringstack]", hosts)
	}
	// Idempotent re-add.
	hosts, _ = store.AddExcludedHost(job.ID, "boringstack")
	if len(hosts) != 1 {
		t.Fatalf("re-add grew set to %v", hosts)
	}
	// Second distinct host appends.
	hosts, _ = store.AddExcludedHost(job.ID, "mbp")
	if len(hosts) != 2 {
		t.Fatalf("second add gave %v, want 2 entries", hosts)
	}
	// Durable: re-read from store reflects both.
	reread, _ := store.GetJob(job.ID)
	if !reread.HostExcluded("boringstack") || !reread.HostExcluded("mbp") {
		t.Fatalf("excluded set not persisted: %v", reread.ExcludedHosts)
	}
}

// errorClassification documents the INFRA-vs-JOB split that drives failover.
func TestRemoteAbsenceClassification(t *testing.T) {
	cases := []struct {
		name   string
		err    error
		absent bool
	}{
		{"wrapped-absent", errRemoteAbsent{err: fmt.Errorf("boom")}, true},
		{"job-4xx", fmt.Errorf("remote mac returned 400: bad model"), false},
		{"nil", nil, false},
	}
	for _, c := range cases {
		if got := isRemoteAbsence(c.err); got != c.absent {
			t.Errorf("%s: isRemoteAbsence=%v want %v", c.name, got, c.absent)
		}
	}
}
