package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func newTestStore(t *testing.T) (*Store, string) {
	t.Helper()
	dir := t.TempDir()
	store, err := NewStore(filepath.Join(dir, "arbiter.db"))
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	t.Cleanup(func() { store.Close() })
	store.InitDedup()
	outputDir := filepath.Join(dir, "output")
	if err := os.MkdirAll(filepath.Join(outputDir, "jobs"), 0o755); err != nil {
		t.Fatalf("mkdir output: %v", err)
	}
	return store, outputDir
}

func TestReconcileFollowingJobs(t *testing.T) {
	store, outputDir := newTestStore(t)

	completedPayload := json.RawMessage(`{"prompt":"completed"}`)
	completedOrig, err := store.CreateJob("z-image-turbo", "image-generate", completedPayload, 1)
	if err != nil {
		t.Fatalf("create completed original: %v", err)
	}
	completedResult := json.RawMessage(`{"file":"result.png"}`)
	if err := store.UpdateState(completedOrig.ID, "completed", WithResult(completedResult), WithFinishedAt(nowTS())); err != nil {
		t.Fatalf("complete original: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(outputDir, "jobs", completedOrig.ID), 0o755); err != nil {
		t.Fatalf("mkdir completed original dir: %v", err)
	}
	completedFollowerA, _ := store.CreateFollowerJob("z-image-turbo", "image-generate", completedPayload, completedOrig.ID)
	completedFollowerB, _ := store.CreateFollowerJob("z-image-turbo", "image-generate", completedPayload, completedOrig.ID)

	cancelledPayload := json.RawMessage(`{"prompt":"cancelled"}`)
	cancelledOrig, err := store.CreateJob("tts-clone", "tts-clone", cancelledPayload, 1)
	if err != nil {
		t.Fatalf("create cancelled original: %v", err)
	}
	if err := store.UpdateState(cancelledOrig.ID, "cancelled", WithFinishedAt(nowTS())); err != nil {
		t.Fatalf("cancel original: %v", err)
	}
	cancelledHash := computeJobHash(cancelledOrig.JobType, cancelledOrig.ModelID, cancelledOrig.Payload)
	store.DedupRegister(cancelledHash, cancelledOrig.ID)
	cancelledFollowerA, _ := store.CreateFollowerJob("tts-clone", "tts-clone", cancelledPayload, cancelledOrig.ID)
	cancelledFollowerB, _ := store.CreateFollowerJob("tts-clone", "tts-clone", cancelledPayload, cancelledOrig.ID)

	missingPayload := json.RawMessage(`{"prompt":"missing"}`)
	missingFollower, _ := store.CreateFollowerJob("tts-clone", "tts-clone", missingPayload, "deadbeefcafe")

	queuedPayload := json.RawMessage(`{"prompt":"queued"}`)
	queuedOrig, _ := store.CreateJob("tts-clone", "tts-clone", queuedPayload, 1)
	queuedFollower, _ := store.CreateFollowerJob("tts-clone", "tts-clone", queuedPayload, queuedOrig.ID)

	resolved := store.ReconcileFollowingJobs(outputDir)
	if resolved != 5 {
		t.Fatalf("resolved followers = %d, want 5", resolved)
	}

	for _, followerID := range []string{completedFollowerA.ID, completedFollowerB.ID} {
		follower, err := store.GetJob(followerID)
		if err != nil {
			t.Fatalf("get completed follower %s: %v", followerID, err)
		}
		if follower.State != "completed" {
			t.Fatalf("completed follower state = %s, want completed", follower.State)
		}
		if follower.Error != "" {
			t.Fatalf("completed follower error = %q, want cleared", follower.Error)
		}
		if follower.Result == nil || string(*follower.Result) != string(completedResult) {
			t.Fatalf("completed follower result = %v, want %s", follower.Result, completedResult)
		}
		info, err := os.Lstat(filepath.Join(outputDir, "jobs", followerID))
		if err != nil {
			t.Fatalf("stat follower symlink %s: %v", followerID, err)
		}
		if info.Mode()&os.ModeSymlink == 0 {
			t.Fatalf("follower dir for %s is not a symlink", followerID)
		}
	}

	promoted, err := store.GetJob(cancelledFollowerA.ID)
	if err != nil {
		t.Fatalf("get promoted follower: %v", err)
	}
	if promoted.State != "queued" || promoted.Error != "" {
		t.Fatalf("promoted follower = state %s error %q, want queued/cleared", promoted.State, promoted.Error)
	}
	rebased, err := store.GetJob(cancelledFollowerB.ID)
	if err != nil {
		t.Fatalf("get rebased follower: %v", err)
	}
	if rebased.State != "following" || rebased.Error != "following:"+cancelledFollowerA.ID {
		t.Fatalf("rebased follower = state %s error %q, want following:%s", rebased.State, rebased.Error, cancelledFollowerA.ID)
	}
	if dedupJobID, err := store.DedupLookup(cancelledHash, 86400); err != nil || dedupJobID != cancelledFollowerA.ID {
		t.Fatalf("dedup lookup = %q, %v, want %s", dedupJobID, err, cancelledFollowerA.ID)
	}

	missingPromoted, err := store.GetJob(missingFollower.ID)
	if err != nil {
		t.Fatalf("get missing follower: %v", err)
	}
	if missingPromoted.State != "queued" || missingPromoted.Error != "" {
		t.Fatalf("missing follower = state %s error %q, want queued/cleared", missingPromoted.State, missingPromoted.Error)
	}

	stillFollowing, err := store.GetJob(queuedFollower.ID)
	if err != nil {
		t.Fatalf("get queued follower: %v", err)
	}
	if stillFollowing.State != "following" || stillFollowing.Error != "following:"+queuedOrig.ID {
		t.Fatalf("queued follower = state %s error %q, want following original", stillFollowing.State, stillFollowing.Error)
	}
}

// TestDedupRecoveredJobsCancelsPlainDuplicates verifies the base behaviour:
// two content-identical queued jobs WITHOUT force → the later one is cancelled.
func TestDedupRecoveredJobsCancelsPlainDuplicates(t *testing.T) {
	store, _ := newTestStore(t)
	payload := json.RawMessage(`{"prompt":"same"}`)

	first, _ := store.CreateJob("z-image-turbo", "image-generate", payload, 1)
	second, _ := store.CreateJob("z-image-turbo", "image-generate", payload, 1)

	removed := store.DedupRecoveredJobs()
	if removed != 1 {
		t.Fatalf("removed = %d, want 1", removed)
	}

	firstAfter, _ := store.GetJob(first.ID)
	if firstAfter.State != "queued" {
		t.Fatalf("first job state = %s, want queued", firstAfter.State)
	}
	secondAfter, _ := store.GetJob(second.ID)
	if secondAfter.State != "cancelled" || secondAfter.Error != "dedup: duplicate of "+first.ID {
		t.Fatalf("second job = state %s error %q, want cancelled/dedup of %s",
			secondAfter.State, secondAfter.Error, first.ID)
	}
}

func TestComputeJobHashIncludesCanonicalModel(t *testing.T) {
	payload := json.RawMessage(`{"messages":[{"role":"user","content":"same"}]}`)
	first := computeJobHash("chat-completion", "llm:qwen", payload)
	second := computeJobHash("chat-completion", "llm:gemma", payload)
	if first == second {
		t.Fatal("different concrete models shared one dedup identity")
	}
}

func TestComputeJobHashUsesCanonicalizedChatPayload(t *testing.T) {
	aliasPayload, err := canonicalizeChatParams(
		json.RawMessage(`{"model":"local-chat","messages":[{"role":"user","content":"same"}]}`),
		"llm:qwen",
	)
	if err != nil {
		t.Fatal(err)
	}
	concretePayload := json.RawMessage(`{"model":"qwen","messages":[{"role":"user","content":"same"}]}`)
	if computeJobHash("chat-completion", "llm:qwen", aliasPayload) !=
		computeJobHash("chat-completion", "llm:qwen", concretePayload) {
		t.Fatal("alias was hashed before canonicalization")
	}
}

// TestDedupRecoveredJobsExemptsForceJobs mirrors the 2026-07-04 incident:
// two content-identical queued jobs, BOTH force:true, the later one created
// after the earlier. Recovery-time dedup must NOT cancel the later force job
// (it deliberately opted out of dedup on the submit path), and must not treat
// force jobs as canonical originals either.
func TestDedupRecoveredJobsExemptsForceJobs(t *testing.T) {
	store, _ := newTestStore(t)
	forcePayload := json.RawMessage(`{"prompt":"denoise chunk","force":true}`)

	// Earlier job (like 73a18713ca69 at 10:20) and later retry (like
	// 1f6130b5c0ee at 12:50) — content-identical, both force:true.
	earlier, _ := store.CreateJob("ltx2-dev-denoise", "video-generate", forcePayload, 1)
	later, _ := store.CreateJob("ltx2-dev-denoise", "video-generate", forcePayload, 1)

	// A plain (non-force) duplicate of the SAME content must still be cancelled,
	// confirming the exemption is scoped to force jobs and the sweep still works.
	plainPayload := json.RawMessage(`{"prompt":"denoise chunk"}`)
	plainFirst, _ := store.CreateJob("ltx2-dev-denoise", "video-generate", plainPayload, 1)
	plainDup, _ := store.CreateJob("ltx2-dev-denoise", "video-generate", plainPayload, 1)

	store.DedupRecoveredJobs()

	earlierAfter, _ := store.GetJob(earlier.ID)
	if earlierAfter.State != "queued" {
		t.Fatalf("earlier force job state = %s, want queued", earlierAfter.State)
	}
	laterAfter, _ := store.GetJob(later.ID)
	if laterAfter.State != "queued" {
		t.Fatalf("later force job state = %s (error %q), want queued — force jobs must survive recovery dedup",
			laterAfter.State, laterAfter.Error)
	}

	plainFirstAfter, _ := store.GetJob(plainFirst.ID)
	if plainFirstAfter.State != "queued" {
		t.Fatalf("plain first job state = %s, want queued", plainFirstAfter.State)
	}
	plainDupAfter, _ := store.GetJob(plainDup.ID)
	if plainDupAfter.State != "cancelled" || plainDupAfter.Error != "dedup: duplicate of "+plainFirst.ID {
		t.Fatalf("plain duplicate = state %s error %q, want cancelled/dedup of %s",
			plainDupAfter.State, plainDupAfter.Error, plainFirst.ID)
	}
}

func TestRecoverFromCrashRequeuesRunningAndScheduled(t *testing.T) {
	store, _ := newTestStore(t)
	payload := json.RawMessage(`{"prompt":"recover"}`)

	scheduled, _ := store.CreateJob("tts-clone", "tts-clone", payload, 1)
	if err := store.UpdateState(scheduled.ID, "scheduled"); err != nil {
		t.Fatalf("set scheduled: %v", err)
	}

	running, _ := store.CreateJob("tts-clone", "tts-clone", payload, 1)
	if err := store.UpdateState(running.ID, "running", WithStartedAt(nowTS())); err != nil {
		t.Fatalf("set running: %v", err)
	}

	original, _ := store.CreateJob("tts-clone", "tts-clone", payload, 1)
	follower, _ := store.CreateFollowerJob("tts-clone", "tts-clone", payload, original.ID)

	recovered, err := store.RecoverFromCrash()
	if err != nil {
		t.Fatalf("recover from crash: %v", err)
	}
	if recovered != 2 {
		t.Fatalf("recovered = %d, want 2", recovered)
	}

	scheduledAfter, _ := store.GetJob(scheduled.ID)
	if scheduledAfter.State != "queued" || scheduledAfter.StartedAt != nil {
		t.Fatalf("scheduled after recovery = state %s started_at %v, want queued/nil", scheduledAfter.State, scheduledAfter.StartedAt)
	}

	runningAfter, _ := store.GetJob(running.ID)
	if runningAfter.State != "queued" || runningAfter.StartedAt != nil {
		t.Fatalf("running after recovery = state %s started_at %v, want queued/nil", runningAfter.State, runningAfter.StartedAt)
	}

	followerAfter, _ := store.GetJob(follower.ID)
	if followerAfter.State != "following" || followerAfter.Error != "following:"+original.ID {
		t.Fatalf("follower after recovery = state %s error %q, want untouched following", followerAfter.State, followerAfter.Error)
	}
}

func TestUpdateStateDoesNotResurrectTerminalJob(t *testing.T) {
	store, _ := newTestStore(t)
	payload := json.RawMessage(`{"prompt":"cancel race"}`)
	job, _ := store.CreateJob("llm:test", "chat-completion", payload, 1)

	if err := store.UpdateState(job.ID, "cancelled", WithFinishedAt(nowTS())); err != nil {
		t.Fatalf("cancel job: %v", err)
	}
	if err := store.UpdateState(job.ID, "scheduled", WithStartedAt(nowTS())); err != nil {
		t.Fatalf("late schedule update: %v", err)
	}

	after, _ := store.GetJob(job.ID)
	if after.State != "cancelled" {
		t.Fatalf("state resurrected to %s, want cancelled", after.State)
	}
	if after.FinishedAt == nil {
		t.Fatalf("finished_at was cleared")
	}
}

func TestRecoverFromCrashDoesNotRequeueFinishedActiveState(t *testing.T) {
	store, _ := newTestStore(t)
	payload := json.RawMessage(`{"prompt":"finished scheduled"}`)
	job, _ := store.CreateJob("llm:test", "chat-completion", payload, 1)

	finished := nowTS()
	if err := store.UpdateState(job.ID, "scheduled", WithStartedAt(finished-10), WithFinishedAt(finished)); err != nil {
		t.Fatalf("set inconsistent scheduled job: %v", err)
	}

	recovered, err := store.RecoverFromCrash()
	if err != nil {
		t.Fatalf("recover from crash: %v", err)
	}
	if recovered != 0 {
		t.Fatalf("recovered = %d, want 0", recovered)
	}
	after, _ := store.GetJob(job.ID)
	if after.State != "failed" {
		t.Fatalf("state = %s, want failed", after.State)
	}
	if after.FinishedAt == nil {
		t.Fatalf("finished_at missing after recovery cleanup")
	}
}
