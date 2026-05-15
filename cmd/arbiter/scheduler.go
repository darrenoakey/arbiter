package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Scheduler struct {
	config          *Config
	store           *Store
	mgr             *InstanceManager
	logger          *EventLogger
	outputDir       string
	inboxDir        string // if set, input files are deleted after job completion/failure
	wake            chan struct{}
	shuttingDown    atomic.Bool
	cooldownMu      sync.Mutex
	cooldownUntil   map[string]time.Time // model -> skip until this time (load failures)
	pressureMu      sync.Mutex
	currentPressure float64 // sum of in-flight job pressure indices
	// Inference failure circuit-breaker
	failureCountMu       sync.Mutex
	failureCount         map[string]int       // model -> consecutive inference failures
	failurePaused        map[string]time.Time // model -> paused until (inference circuit-breaker)
	failureCooldownLevel map[string]int       // model -> escalation level (0=30s,1=1m,2=5m,3+=15m)
	// Load failure circuit-breaker
	loadFailureCountMu       sync.Mutex
	loadFailureCount         map[string]int       // model -> consecutive load failures
	loadFailurePaused        map[string]time.Time // model -> paused until
	loadFailureCooldownLevel map[string]int       // model -> escalation level
	// Per-job load-failure attempts — bounds the retry loop when a fresh worker
	// dies during load because the previous inference poisoned the CUDA state.
	// Without this, a single bad input could loop forever: subprocess dies
	// during load → requeue → subprocess dies during load → ... and every job
	// behind it is starved. Map is reset on successful load or terminal state.
	loadAttemptsMu sync.Mutex
	loadAttempts   map[string]int
	// Stream handoff: when a chat-completion-stream job is dispatched the
	// scheduler hands the picked instance to the API handler via instCh,
	// then blocks on doneCh until the handler finishes proxying SSE. The
	// slot remains reserved (via the same activeJobs accounting as any
	// other job) for the entire stream duration.
	streamMu       sync.Mutex
	streamHandoffs map[string]*streamHandoff
	// Rate-limit for the "scheduler ticked but dispatched nothing while queue
	// is non-empty" diagnostic log. Without rate-limiting we'd emit it every
	// 100ms tick and drown the log; without the log itself we have no way to
	// see why a queue with thousands of jobs isn't draining.
	lastIdleLogMu sync.Mutex
	lastIdleLog   time.Time
}

type streamHandoff struct {
	instCh chan *Instance
	doneCh chan error
}

const maxLoadAttempts = 3

type insufficientMemoryError struct {
	instanceID string
	neededGB   float64
	freeGB     float64
}

func (err insufficientMemoryError) Error() string {
	return fmt.Sprintf("can't load %s: need %.1fGB, only %.1fGB free", err.instanceID, err.neededGB, err.freeGB)
}

func NewScheduler(cfg *Config, store *Store, mgr *InstanceManager, logger *EventLogger, outputDir string) *Scheduler {
	inboxDir := ""
	if cfg.ShareMount != "" {
		inboxDir = filepath.Join(cfg.ShareMount, "inbox")
	}
	return &Scheduler{
		config:                   cfg,
		store:                    store,
		mgr:                      mgr,
		logger:                   logger,
		outputDir:                outputDir,
		inboxDir:                 inboxDir,
		wake:                     make(chan struct{}, 1),
		cooldownUntil:            make(map[string]time.Time),
		failureCount:             make(map[string]int),
		failurePaused:            make(map[string]time.Time),
		failureCooldownLevel:     make(map[string]int),
		loadFailureCount:         make(map[string]int),
		loadFailurePaused:        make(map[string]time.Time),
		loadFailureCooldownLevel: make(map[string]int),
		loadAttempts:             make(map[string]int),
		streamHandoffs:           make(map[string]*streamHandoff),
	}
}

// RegisterStreamHandoff is called by the streaming-chat API handler before it
// waits for dispatch. The returned handoff carries the instance from the
// scheduler to the handler (instCh) and the completion signal back (doneCh).
func (s *Scheduler) RegisterStreamHandoff(jobID string) *streamHandoff {
	h := &streamHandoff{
		instCh: make(chan *Instance, 1),
		doneCh: make(chan error, 1),
	}
	s.streamMu.Lock()
	s.streamHandoffs[jobID] = h
	s.streamMu.Unlock()
	return h
}

// UnregisterStreamHandoff removes a handoff entry. Idempotent.
func (s *Scheduler) UnregisterStreamHandoff(jobID string) {
	s.streamMu.Lock()
	delete(s.streamHandoffs, jobID)
	s.streamMu.Unlock()
}

// waitForStreamHandoff polls briefly for the API handler to register a
// handoff. Returns nil if the handler never registered (handler abandoned
// before the scheduler picked the job, or crashed).
func (s *Scheduler) waitForStreamHandoff(jobID string, timeout time.Duration) *streamHandoff {
	deadline := time.Now().Add(timeout)
	for {
		s.streamMu.Lock()
		h := s.streamHandoffs[jobID]
		s.streamMu.Unlock()
		if h != nil {
			return h
		}
		if time.Now().After(deadline) {
			return nil
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// Wake signals the scheduler to check for new work.
func (s *Scheduler) Wake() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

func (s *Scheduler) MarkShuttingDown() {
	s.shuttingDown.Store(true)
}

func (s *Scheduler) shouldRequeueForShutdown(err error, resp *WorkerResponse) bool {
	if !s.shuttingDown.Load() {
		return false
	}
	if err != nil {
		return true
	}
	if resp == nil {
		return false
	}
	return resp.Status == "error" && strings.Contains(resp.Error, "subprocess died")
}

func (s *Scheduler) computePriority(modelID string) float64 {
	cfg, ok := s.config.Models[modelID]
	if !ok {
		return 1e9
	}
	p := cfg.AvgInferenceMs
	if !s.mgr.IsLoaded(modelID) {
		p += cfg.LoadMs
	}
	return p
}

func (s *Scheduler) rescoreModel(modelID string) {
	p := s.computePriority(modelID)
	s.store.UpdatePriority(modelID, p)
}

func (s *Scheduler) rescoreAll() {
	for modelID := range s.config.Models {
		s.rescoreModel(modelID)
	}
}

func (s *Scheduler) pendingJobsByModel() map[string]int {
	pending := make(map[string]int, len(s.config.Models))
	for modelID := range s.config.Models {
		counts, err := s.store.CountByState(modelID)
		if err != nil {
			continue
		}
		pending[modelID] = counts["queued"] + counts["scheduled"] + counts["running"]
	}
	return pending
}

// failureCooldownDurations defines escalating pause durations for the inference
// circuit-breaker. The index maps to failureCooldownLevel.
var failureCooldownDurations = []time.Duration{
	30 * time.Second,
	1 * time.Minute,
	5 * time.Minute,
	15 * time.Minute,
}

// RecordSuccess resets the consecutive failure counter for a model. If the model
// is currently paused the pause is left in place — it will expire on its own.
// The failure count is cleared so the next round of failures starts fresh.
func (s *Scheduler) RecordSuccess(modelID string) {
	s.failureCountMu.Lock()
	defer s.failureCountMu.Unlock()
	s.failureCount[modelID] = 0
}

// RecordFailure increments the consecutive failure counter for a model. When the
// count reaches the threshold the model is paused for an escalating duration.
func (s *Scheduler) RecordFailure(modelID string) {
	const threshold = 10
	s.failureCountMu.Lock()
	defer s.failureCountMu.Unlock()
	s.failureCount[modelID]++
	if s.failureCount[modelID] < threshold {
		return
	}
	// Threshold reached — enter cooldown.
	level := s.failureCooldownLevel[modelID]
	if level >= len(failureCooldownDurations) {
		level = len(failureCooldownDurations) - 1
	}
	dur := failureCooldownDurations[level]
	until := time.Now().Add(dur)
	s.failurePaused[modelID] = until
	// Escalate for next time, capped at the last level.
	if s.failureCooldownLevel[modelID] < len(failureCooldownDurations)-1 {
		s.failureCooldownLevel[modelID]++
	}
	// Reset the counter so that after the pause lifts the next N failures
	// are counted fresh.
	s.failureCount[modelID] = 0
	slog.Warn("circuit-breaker: model paused after consecutive failures",
		"model", modelID,
		"threshold", threshold,
		"cooldown", dur,
		"resume_at", until.Format(time.RFC3339),
	)
}

// IsModelPaused reports whether the model's inference circuit-breaker is active.
// If the pause has expired it is cleared and (false, zero) is returned.
func (s *Scheduler) IsModelPaused(modelID string) (bool, time.Time) {
	s.failureCountMu.Lock()
	defer s.failureCountMu.Unlock()
	until, ok := s.failurePaused[modelID]
	if !ok {
		return false, time.Time{}
	}
	if time.Now().After(until) {
		delete(s.failurePaused, modelID)
		return false, time.Time{}
	}
	return true, until
}

// RecordLoadFailure increments the consecutive load failure counter for a model.
// Threshold: 3 failures → escalating pause. Queued jobs are preserved — only a
// human operator may cancel them. Pauses escalate 30s → 1m → 5m → 15m so a
// persistent failure stops spinning the scheduler but doesn't destroy user work.
func (s *Scheduler) RecordLoadFailure(modelID string) {
	const threshold = 3
	s.loadFailureCountMu.Lock()
	defer s.loadFailureCountMu.Unlock()
	s.loadFailureCount[modelID]++
	if s.loadFailureCount[modelID] < threshold {
		return
	}
	level := s.loadFailureCooldownLevel[modelID]
	if level >= len(failureCooldownDurations) {
		level = len(failureCooldownDurations) - 1
	}
	dur := failureCooldownDurations[level]
	until := time.Now().Add(dur)
	s.loadFailurePaused[modelID] = until
	if s.loadFailureCooldownLevel[modelID] < len(failureCooldownDurations)-1 {
		s.loadFailureCooldownLevel[modelID]++
	}
	s.loadFailureCount[modelID] = 0
	slog.Warn("load circuit-breaker: model paused after consecutive load failures — queue preserved",
		"model", modelID, "threshold", threshold, "cooldown", dur,
		"resume_at", until.Format(time.RFC3339))
	s.logger.Log("model.load_circuit_breaker", map[string]any{
		"model_id": modelID,
		"cooldown": dur.String(),
	})
	// NOTE: Deliberately do NOT cancel queued or following jobs here.
	// A load failure may be transient (bad deployment, missing file, dependency
	// update in progress, GPU OOM from another model). Cancelling queued work
	// destroys user data — only a human operator should decide to cancel.
	// When the cooldown expires, the next scheduling attempt tries to load again;
	// if the problem persists, the CB re-activates with a longer cooldown.
}

// RecordLoadSuccess resets the load failure counter and escalation level.
func (s *Scheduler) RecordLoadSuccess(modelID string) {
	s.loadFailureCountMu.Lock()
	defer s.loadFailureCountMu.Unlock()
	s.loadFailureCount[modelID] = 0
	s.loadFailureCooldownLevel[modelID] = 0
}

// IsModelLoadPaused reports whether the model's load circuit-breaker is active.
func (s *Scheduler) IsModelLoadPaused(modelID string) (bool, time.Time) {
	s.loadFailureCountMu.Lock()
	defer s.loadFailureCountMu.Unlock()
	until, ok := s.loadFailurePaused[modelID]
	if !ok {
		return false, time.Time{}
	}
	if time.Now().After(until) {
		delete(s.loadFailurePaused, modelID)
		return false, time.Time{}
	}
	return true, until
}

// pressureAgeRate is how fast a model's effective pressure budget grows per
// second its oldest queued job has been waiting. With rate=0.01 / s a job
// waiting 100s adds 1.0 to the budget — enough to fit a PI=1.0 "exclusive"
// model alongside any amount of low-pressure traffic.
//
// This prevents starvation: without aging, a heavy-pressure model
// (ltx2-encode at PI=1.0) is permanently blocked by any low-pressure
// traffic (gemma at PI=0.01) because cp + 1.0 > 1.0 always. With aging the
// blocked job's budget grows linearly with wait time until it fits.
const pressureAgeRate = 0.01

// getFullModels returns model IDs that are at total capacity or would exceed the pressure budget.
func (s *Scheduler) getFullModels() map[string]bool {
	full := make(map[string]bool)
	now := time.Now()
	s.cooldownMu.Lock()
	for modelID, until := range s.cooldownUntil {
		if now.Before(until) {
			full[modelID] = true
		} else {
			delete(s.cooldownUntil, modelID)
		}
	}
	s.cooldownMu.Unlock()

	s.pressureMu.Lock()
	cp := s.currentPressure
	s.pressureMu.Unlock()

	// Per-model oldest queued age — used to age the pressure budget so a
	// long-waiting job earns more allowance.
	ages, _ := s.store.OldestQueuedAgeByModel()

	for modelID, cfg := range s.config.Models {
		if full[modelID] {
			continue
		}
		// Check load circuit-breaker
		if paused, _ := s.IsModelLoadPaused(modelID); paused {
			full[modelID] = true
			continue
		}
		// Check inference circuit-breaker
		if paused, _ := s.IsModelPaused(modelID); paused {
			full[modelID] = true
			continue
		}
		// Capacity check uses the instance manager (atomic counters), NOT
		// store.CountActive — they diverge during dispatch races, and the
		// dispatcher gates on PickInstance which uses HasCapacity. If we use
		// store state here, the scheduler picks a job for a model that has
		// no instance capacity, then PickInstance returns nil, then the job
		// bounces back to "queued" — and meanwhile other models with ready
		// work and loaded instances are starved.
		if !s.mgr.ModelHasAnyCapacity(modelID) {
			slog.Info("scheduler.full: no instance with capacity",
				"model", modelID,
				"max_concurrent", cfg.MaxConcurrent, "max_instances", *cfg.MaxInstances)
			full[modelID] = true
			continue
		}
		// Age the budget by the oldest queued job's wait time. A model with
		// queued work that's been sitting for a while gets more allowance,
		// guaranteeing eventual dispatch even under sustained low-pressure
		// load. This is the multiplier-style aging the user asked for: a job's
		// effective weight grows linearly with wait time, no thresholds.
		budget := 1.0 + ages[modelID]*pressureAgeRate
		if cp+*cfg.PressureIndex > budget+1e-9 {
			slog.Info("scheduler.full: model would exceed pressure budget",
				"model", modelID, "current_pressure", cp,
				"model_pressure_index", *cfg.PressureIndex,
				"sum", cp+*cfg.PressureIndex,
				"aged_budget", budget, "wait_seconds", ages[modelID])
			full[modelID] = true
		}
	}
	return full
}

// ensureLoaded makes sure an instance is loaded within the VRAM budget.
// Strategy: try reserve, evict idle if needed, retry.
func (s *Scheduler) ensureLoaded(inst *Instance) error {
	state := inst.State()
	if state == "loaded" {
		return nil
	}

	if state == "loading" {
		// Another caller (preload, prior dispatch) already kicked off the load.
		// Wait for it to finish rather than failing — failing here counts as a
		// load failure for the dispatcher, which after 3 attempts trips the
		// load circuit-breaker and pauses the model for 30s+. That's how we
		// ended up never getting concurrent dispatch on vLLM: the first job
		// triggers load, the 2nd-Nth jobs picked while still loading all
		// "fail", scheduler gives up, queue stalls.
		slog.Info("ensureLoaded.wait_for_in_progress_load", "instance", inst.InstanceID)
		deadline := time.Now().Add(10 * time.Minute)
		for time.Now().Before(deadline) {
			s2 := inst.State()
			if s2 == "loaded" {
				slog.Info("ensureLoaded.in_progress_load_completed", "instance", inst.InstanceID)
				return nil
			}
			if s2 == "stopped" || s2 == "error" || s2 == "unloaded" {
				slog.Warn("ensureLoaded.in_progress_load_failed",
					"instance", inst.InstanceID, "final_state", s2)
				return fmt.Errorf("instance %s in-progress load ended in state=%s", inst.InstanceID, s2)
			}
			time.Sleep(500 * time.Millisecond)
		}
		return fmt.Errorf("instance %s still loading after 10min", inst.InstanceID)
	}

	if state == "stopped" || state == "unloaded" || state == "error" {
		needed := inst.memoryGB
		freeGB := s.mgr.FreeGB()

		slog.Info("ensureLoaded: need VRAM", "instance", inst.InstanceID,
			"needed_gb", needed, "free_gb", freeGB, "state", state)

		if state == "error" {
			s.mgr.ReleaseMemoryFor(inst)
		}

		// Try reserve
		if !s.mgr.ReserveMemoryFor(inst, needed) {
			// Evict idle models. Use the queue-aware evictor so that a model
			// which still has queued/running work is preserved over a model
			// with nothing waiting for it.
			deficit := needed - s.mgr.FreeGB()
			if deficit > 0 {
				queuedJobs := make(map[string]int)
				for modelID := range s.config.Models {
					counts, err := s.store.CountByState(modelID)
					if err != nil {
						continue
					}
					queuedJobs[modelID] = counts["queued"] + counts["scheduled"] + counts["running"]
				}
				slog.Info("ensureLoaded: evicting idle models", "instance", inst.InstanceID, "deficit_gb", deficit)
				if _, err := s.mgr.EvictForGBWithQueueInfo(deficit, queuedJobs); err != nil {
					slog.Warn("ensureLoaded: queue-aware eviction insufficient",
						"instance", inst.InstanceID, "error", err)
				}
			}

			// Retry
			if !s.mgr.ReserveMemoryFor(inst, needed) {
				slog.Warn("ensureLoaded: can't reserve VRAM after eviction",
					"instance", inst.InstanceID, "needed_gb", needed, "free_gb", s.mgr.FreeGB())
				return insufficientMemoryError{
					instanceID: inst.InstanceID,
					neededGB:   needed,
					freeGB:     s.mgr.FreeGB(),
				}
			}
		}

		slog.Info("ensureLoaded: VRAM reserved, loading model",
			"instance", inst.InstanceID, "memory_gb", needed)
		s.logger.Log("model.load_start", map[string]any{
			"model_id":    inst.ModelID,
			"instance_id": inst.InstanceID,
			"memory_gb":   inst.memoryGB,
		})

		if err := inst.Load("cuda"); err != nil {
			s.mgr.ReleaseMemoryFor(inst)
			slog.Error("ensureLoaded: load failed", "instance", inst.InstanceID, "error", err)
			s.logger.Log("model.load_error", map[string]any{
				"model_id":    inst.ModelID,
				"instance_id": inst.InstanceID,
				"error":       err.Error(),
			})
			return err
		}

		slog.Info("ensureLoaded: model loaded successfully", "instance", inst.InstanceID)
		s.logger.Log("model.load_done", map[string]any{
			"model_id":    inst.ModelID,
			"instance_id": inst.InstanceID,
			"memory_gb":   inst.memoryGB,
		})
		s.mgr.AuditVRAMConsistency("post-load:" + inst.InstanceID)
		s.rescoreModel(inst.ModelID)
	}

	return nil
}

// dispatchJobToInstance loads the instance and runs inference.
// activeJobs and currentPressure are already incremented by the caller.
// This function owns both reservations and releases them when done.
func (s *Scheduler) dispatchJobToInstance(job *Job, inst *Instance, pressure float64) {
	defer func() {
		// Catch any unexpected panic so a single bad job can never crash the server.
		if r := recover(); r != nil {
			slog.Error("panic in dispatchJobToInstance — recovered", "panic", r, "job", job.ID, "model", job.ModelID)
			s.logger.Log("job.panic", map[string]any{"job_id": job.ID, "model_id": job.ModelID, "panic": fmt.Sprintf("%v", r)})
			s.store.UpdateState(job.ID, "failed", WithError(fmt.Sprintf("internal panic: %v", r)))
		}
		s.pressureMu.Lock()
		s.currentPressure -= pressure
		if s.currentPressure < 0 {
			s.currentPressure = 0
		}
		s.pressureMu.Unlock()
		s.mgr.ReleaseAndCheck(inst)
		s.rescoreModel(job.ModelID)
	}()

	slog.Info("dispatching job", "job_id", job.ID, "model", job.ModelID, "instance", inst.InstanceID)
	s.logger.Log("job.scheduled", map[string]any{"job_id": job.ID, "model_id": job.ModelID, "instance_id": inst.InstanceID})

	// Ensure loaded
	if err := s.ensureLoaded(inst); err != nil {
		if _, ok := err.(insufficientMemoryError); ok {
			slog.Warn("can't load instance yet, leaving job queued until memory is available",
				"instance", inst.InstanceID, "job", job.ID, "error", err)
			s.store.UpdateState(job.ID, "queued")
			s.cooldownMu.Lock()
			s.cooldownUntil[job.ModelID] = time.Now().Add(30 * time.Second)
			s.cooldownMu.Unlock()
			return
		}

		s.loadAttemptsMu.Lock()
		s.loadAttempts[job.ID]++
		attempts := s.loadAttempts[job.ID]
		s.loadAttemptsMu.Unlock()

		s.RecordLoadFailure(job.ModelID)
		s.cooldownMu.Lock()
		s.cooldownUntil[job.ModelID] = time.Now().Add(5 * time.Second)
		s.cooldownMu.Unlock()

		if attempts >= maxLoadAttempts {
			errMsg := fmt.Sprintf("load failed after %d attempts: %s", attempts, err)
			slog.Error("giving up on job after repeated load failures",
				"instance", inst.InstanceID, "job", job.ID, "attempts", attempts, "error", err)
			s.store.UpdateState(job.ID, "failed", WithError(errMsg), WithFinishedAt(nowTS()))
			if n := s.store.ResolveFollowers(job.ID, "failed", nil, errMsg, s.outputDir); n > 0 {
				slog.Info("resolved follower jobs", "original", job.ID, "followers", n, "state", "failed")
			}
			s.logger.Log("job.failed", map[string]any{
				"job_id":   job.ID,
				"model_id": job.ModelID,
				"error":    errMsg,
				"attempts": attempts,
			})
			s.loadAttemptsMu.Lock()
			delete(s.loadAttempts, job.ID)
			s.loadAttemptsMu.Unlock()
			s.cleanupJobInbox(job)
			return
		}

		slog.Warn("can't load instance, requeueing",
			"instance", inst.InstanceID, "job", job.ID, "attempt", attempts, "error", err)
		s.store.UpdateState(job.ID, "queued")
		return
	}
	s.loadAttemptsMu.Lock()
	delete(s.loadAttempts, job.ID)
	s.loadAttemptsMu.Unlock()
	s.RecordLoadSuccess(job.ModelID)

	// Mark running
	now := nowTS()
	s.store.UpdateState(job.ID, "running", WithStartedAt(now))
	s.logger.Log("job.started", map[string]any{
		"job_id":      job.ID,
		"model_id":    job.ModelID,
		"instance_id": inst.InstanceID,
	})

	// Streaming chat goes through the queue like everything else. The API
	// handler did the SSE proxying; the scheduler just holds the slot until
	// the handler signals done. Same activeJobs accounting as any other job.
	if job.JobType == "chat-completion-stream" {
		s.dispatchStreamHandoff(job, inst)
		return
	}

	// Run inference (blocking) — we use InferRaw which skips activeJobs management
	jobDir := filepath.Join(s.outputDir, "jobs", job.ID)
	os.MkdirAll(jobDir, 0o755)

	// Kill the worker if inference exceeds max_runtime_seconds.
	// This unblocks the goroutine (subprocess death → readLoop broadcasts → sendAndReceive returns).
	// The DB-level watchdog (RunJobWatchdog) is a secondary check; this is the real fix.
	if runtimeSec := s.config.Models[job.ModelID].MaxRuntimeSec; runtimeSec > 0 {
		killTimer := time.AfterFunc(time.Duration(runtimeSec)*time.Second, func() {
			slog.Warn("inference timeout — killing worker",
				"job", job.ID, "model", job.ModelID,
				"instance", inst.InstanceID, "max_runtime_seconds", runtimeSec)
			s.logger.Log("job.timeout", map[string]any{
				"job_id":              job.ID,
				"model_id":            job.ModelID,
				"instance_id":         inst.InstanceID,
				"max_runtime_seconds": runtimeSec,
			})
			inst.Kill()
		})
		defer killTimer.Stop()
	}

	start := time.Now()
	resp, err := inst.InferRaw(job.ID, job.JobType, job.Payload, jobDir)
	elapsed := time.Since(start).Seconds()

	if s.shouldRequeueForShutdown(err, resp) {
		s.store.UpdateState(job.ID, "queued")
		s.logger.Log("job.requeued", map[string]any{
			"job_id":            job.ID,
			"model_id":          job.ModelID,
			"reason":            "arbiter shutdown",
			"inference_seconds": elapsed,
		})
		slog.Warn("requeueing job due to shutdown", "job", job.ID, "model", job.ModelID)
		return
	}

	if err != nil {
		errMsg := fmt.Sprintf("inference error: %s", err)
		s.store.UpdateState(job.ID, "failed", WithError(errMsg), WithFinishedAt(nowTS()))
		if n := s.store.ResolveFollowers(job.ID, "failed", nil, errMsg, s.outputDir); n > 0 {
			slog.Info("resolved follower jobs", "original", job.ID, "followers", n, "state", "failed")
		}
		s.logger.Log("job.failed", map[string]any{
			"job_id":            job.ID,
			"model_id":          job.ModelID,
			"error":             errMsg,
			"inference_seconds": elapsed,
		})
		slog.Error("job failed", "job", job.ID, "error", err)
		if !isClientError(errMsg) {
			s.RecordFailure(job.ModelID)
		}
		s.cleanupJobInbox(job)
		return
	}

	if resp.Status == "cancelled" {
		s.store.UpdateState(job.ID, "cancelled", WithFinishedAt(nowTS()))
		s.logger.Log("job.cancelled", map[string]any{"job_id": job.ID, "model_id": job.ModelID})
		s.cleanupJobInbox(job)
		// Cancellation is intentional — don't penalise the model.
	} else if resp.Status == "error" {
		s.store.UpdateState(job.ID, "failed", WithError(resp.Error), WithFinishedAt(nowTS()))
		s.logger.Log("job.failed", map[string]any{
			"job_id":            job.ID,
			"model_id":          job.ModelID,
			"error":             resp.Error,
			"inference_seconds": elapsed,
		})
		if !isClientError(resp.Error) {
			s.RecordFailure(job.ModelID)
		}
		s.cleanupJobInbox(job)
	} else {
		// Relocate the job output to the share mount so spark's local disk
		// doesn't accumulate result files. Done before persisting state so
		// the stored result reflects the final path.
		newDir := relocateJobOutput(s.config, job.ID, jobDir)
		if newDir != jobDir {
			resp.Result = rewriteResultPaths(resp.Result, jobDir, newDir)
		}
		s.store.UpdateState(job.ID, "completed", WithResult(resp.Result), WithFinishedAt(nowTS()))
		rssEntry := map[string]any{
			"job_id":            job.ID,
			"model_id":          job.ModelID,
			"inference_seconds": elapsed,
		}
		if rss := inst.RSSAnon(); rss > 0 {
			rssEntry["worker_rss_anon_mb"] = rss
		}
		if newDir != jobDir {
			rssEntry["output_relocated_to"] = newDir
		}
		s.logger.Log("job.completed", rssEntry)
		s.RecordSuccess(job.ModelID)
		s.cleanupJobInbox(job)
	}

	// Resolve any follower jobs
	if resp != nil {
		var finalState string
		var finalResult *json.RawMessage
		var finalErr string
		if resp.Status == "cancelled" {
			finalState = "cancelled"
		} else if resp.Status == "error" {
			finalState = "failed"
			finalErr = resp.Error
		} else {
			finalState = "completed"
			finalResult = &resp.Result
		}
		if n := s.store.ResolveFollowers(job.ID, finalState, finalResult, finalErr, s.outputDir); n > 0 {
			slog.Info("resolved follower jobs", "original", job.ID, "followers", n, "state", finalState)
		}
	}

	s.Wake() // check for more work
}

// dispatchStreamHandoff hands the picked instance to the API handler that's
// waiting on a chat-completion-stream job, then blocks until the handler
// reports the SSE stream is finished. The slot is held for the entire
// duration via the same activeJobs reservation as any other job.
func (s *Scheduler) dispatchStreamHandoff(job *Job, inst *Instance) {
	const handoffWait = 5 * time.Second
	const streamMax = 30 * time.Minute

	h := s.waitForStreamHandoff(job.ID, handoffWait)
	if h == nil {
		errMsg := "stream handler did not register handoff"
		s.store.UpdateState(job.ID, "failed", WithError(errMsg), WithFinishedAt(nowTS()))
		s.logger.Log("job.failed", map[string]any{
			"job_id": job.ID, "model_id": job.ModelID, "error": errMsg,
		})
		slog.Warn("stream handoff timeout — handler abandoned", "job", job.ID)
		return
	}
	defer s.UnregisterStreamHandoff(job.ID)

	// Hand the instance to the API handler.
	h.instCh <- inst

	// Wait for the handler to finish proxying SSE (or error / disconnect).
	start := time.Now()
	select {
	case err := <-h.doneCh:
		elapsed := time.Since(start).Seconds()
		if err != nil {
			errMsg := fmt.Sprintf("stream error: %s", err)
			s.store.UpdateState(job.ID, "failed", WithError(errMsg), WithFinishedAt(nowTS()))
			s.logger.Log("job.failed", map[string]any{
				"job_id": job.ID, "model_id": job.ModelID,
				"error": errMsg, "inference_seconds": elapsed,
			})
			s.RecordFailure(job.ModelID)
			return
		}
		s.store.UpdateState(job.ID, "completed", WithFinishedAt(nowTS()))
		s.logger.Log("job.completed", map[string]any{
			"job_id": job.ID, "model_id": job.ModelID,
			"inference_seconds": elapsed, "stream": true,
		})
		s.RecordSuccess(job.ModelID)
	case <-time.After(streamMax):
		errMsg := fmt.Sprintf("stream exceeded %s", streamMax)
		s.store.UpdateState(job.ID, "failed", WithError(errMsg), WithFinishedAt(nowTS()))
		s.logger.Log("job.failed", map[string]any{
			"job_id": job.ID, "model_id": job.ModelID, "error": errMsg,
		})
		s.RecordFailure(job.ModelID)
		slog.Warn("stream handoff timeout — handler stuck", "job", job.ID)
	}
}

// tryPreload speculatively loads the next needed instance in the background.
// tryPreload speculatively loads every model that has queued work and fits
// in remaining free VRAM. Spare GPU memory is wasted memory — if there's
// room for the next thing we're going to need, load it now while the current
// thing is processing. The dispatcher will pick it up the moment a slot
// opens, no cold-start latency on the critical path.
//
// Greedy by design: iterates all distinct models with pending work, loading
// each that has an unloaded/stopped instance and fits. Multiple loads may
// kick off in parallel; ensureLoaded's reservation path is atomic so the
// race is benign — at worst one of them backs off when the budget is hit.
func (s *Scheduler) tryPreload() {
	pending := s.pendingJobsByModel()
	if len(pending) == 0 {
		return
	}

	// Reserve a small buffer so a real dispatch about to need VRAM isn't
	// starved by speculative loads. 1GB is enough for small adapters.
	const reserveBufferGB = 1.0

	for modelID, count := range pending {
		if count == 0 {
			continue
		}
		// Skip models the scheduler can't touch right now (circuit breakers,
		// pressure budget). We deliberately do NOT use getFullModels here —
		// "full" excludes models at MaxConcurrent, but those still have
		// loaded instances; preloading them is a no-op anyway because their
		// instances are already loaded. We just want to find unloaded
		// instances we can warm up.
		if paused, _ := s.IsModelLoadPaused(modelID); paused {
			continue
		}
		inst := s.mgr.PickInstance(modelID)
		if inst == nil {
			continue
		}
		state := inst.State()
		if state == "loaded" || state == "loading" {
			continue
		}
		freeGB := s.mgr.FreeGB() - reserveBufferGB
		if freeGB < inst.memoryGB {
			continue
		}
		slog.Info("preloading next-needed model",
			"instance", inst.InstanceID, "memory_gb", inst.memoryGB,
			"free_gb_after_buffer", freeGB, "queued_jobs", count)
		go func(i *Instance) {
			if err := s.ensureLoaded(i); err != nil {
				slog.Debug("preload failed", "instance", i.InstanceID, "error", err)
			}
		}(inst)
	}
}

// Run is the main scheduler loop.
func (s *Scheduler) Run(ctx context.Context) {
	slog.Info("scheduler started")
	s.rescoreAll()

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			slog.Info("scheduler stopped")
			return
		case <-s.wake:
		case <-ticker.C:
		}

		// Real queued work outranks warm residency. If there is backlog for any
		// model, immediately evict loaded idle models that have zero pending
		// work instead of waiting for keepalive expiry or explicit VRAM pressure.
		if evicted, err := s.mgr.EvictIdleNoQueueModels(s.pendingJobsByModel()); err == nil && evicted > 0 {
			s.rescoreAll()
		}

		// Pick and dispatch one job at a time
		full := s.getFullModels()
		job, err := s.store.PickNextJob(full)
		if err != nil {
			slog.Warn("scheduler.pick_next_job error", "error", err)
			continue
		}
		if job == nil {
			s.logIdleSkipIfBacklogged(full)
			continue
		}
		slog.Info("scheduler.picked_job",
			"job_id", job.ID, "model", job.ModelID, "type", job.JobType, "priority", job.Priority)

		// Mark scheduled so it won't be re-picked. Stamp the dispatch time so a
		// watchdog can recover orphaned scheduled jobs if the dispatch path wedges.
		s.store.UpdateState(job.ID, "scheduled", WithStartedAt(nowTS()))

		// Pick instance NOW (synchronous) so concurrent goroutines
		// don't race to pick the same instance
		inst := s.mgr.PickInstance(job.ModelID)
		if inst == nil {
			slog.Info("scheduler.requeue: no instance available",
				"job", job.ID, "model", job.ModelID,
				"reason", "PickInstance returned nil — all instances at max_concurrent or none exist")
			s.store.UpdateState(job.ID, "queued")
			continue
		}
		slog.Info("scheduler.dispatch",
			"job", job.ID, "model", job.ModelID,
			"instance", inst.InstanceID, "instance_state", inst.State(),
			"active_jobs_before", inst.ActiveJobs())
		slog.Info("picked instance for job", "job", job.ID, "model", job.ModelID,
			"instance", inst.InstanceID, "state", inst.State(), "active_jobs", inst.ActiveJobs())
		// Reserve the slot immediately so PickInstance won't return it again
		atomic.AddInt32(&inst.activeJobs, 1)

		// Reserve pressure immediately (main loop is single-threaded for dispatch decisions)
		pressure := *s.config.Models[job.ModelID].PressureIndex
		s.pressureMu.Lock()
		s.currentPressure += pressure
		s.pressureMu.Unlock()
		slog.Info("pressure reserved", "model", job.ModelID, "pressure", pressure, "total", s.currentPressure)

		go func(j *Job, inst *Instance, pressure float64) {
			s.dispatchJobToInstance(j, inst, pressure)
			s.Wake()
		}(job, inst, pressure)

		// Preload next instance in background
		s.tryPreload()
	}
}

// logIdleSkipIfBacklogged emits a diagnostic when the scheduler ticks,
// PickNextJob returns nil, but there is queued work for at least one model.
// That combination means every model with backlog was filtered out (paused,
// at capacity, or over the pressure budget) — historically silent, which
// made "queue not draining" impossible to diagnose from the log alone.
// Rate-limited to once per 5s so the 100ms tick doesn't flood the log.
func (s *Scheduler) logIdleSkipIfBacklogged(full map[string]bool) {
	pending := s.pendingJobsByModel()
	totalPending := 0
	for _, n := range pending {
		totalPending += n
	}
	if totalPending == 0 {
		return
	}

	s.lastIdleLogMu.Lock()
	if time.Since(s.lastIdleLog) < 5*time.Second {
		s.lastIdleLogMu.Unlock()
		return
	}
	s.lastIdleLog = time.Now()
	s.lastIdleLogMu.Unlock()

	s.pressureMu.Lock()
	cp := s.currentPressure
	s.pressureMu.Unlock()

	fullList := make([]string, 0, len(full))
	for m := range full {
		fullList = append(fullList, m)
	}
	pendingFull := make(map[string]int)
	for m, n := range pending {
		if n > 0 && full[m] {
			pendingFull[m] = n
		}
	}
	slog.Info("scheduler.idle_skip: backlog present but no job dispatched",
		"total_pending", totalPending,
		"pending_by_model", pending,
		"full_models", fullList,
		"pending_in_full_models", pendingFull,
		"current_pressure", cp)
}

// RunScheduledWatchdog requeues jobs stuck in "scheduled" long enough that
// they are almost certainly orphaned from a dead or wedged dispatch path.
func (s *Scheduler) RunScheduledWatchdog(ctx context.Context) {
	const (
		interval = 5 * time.Second
		staleSec = 15.0
	)

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}

		recovered, err := s.store.RecoverStuckScheduled(staleSec)
		if err != nil {
			slog.Warn("scheduled watchdog: failed to recover stuck jobs", "error", err)
			continue
		}
		if recovered > 0 {
			slog.Warn("scheduled watchdog: requeued stuck scheduled jobs", "count", recovered)
			s.Wake()
		}
	}
}

// RunKeepalive evicts idle models past their keep_alive_seconds.
//
// Safety rules (learned the hard way):
//   - NEVER evict a model that has queued or scheduled work. Idle time while
//     jobs are waiting is not "really" idle — it just means the scheduler is
//     paused for some reason (cooldown, in-flight dispatch, etc.).
//   - NEVER evict a model whose inference or load circuit-breaker is active.
//     Cooldown time is artificial idleness; evicting forces a cold reload
//     (minutes) the instant the cooldown expires.
//   - NEVER evict while active > 0 (existing rule).
func (s *Scheduler) RunKeepalive(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}

		now := time.Now()
		for modelID, cfg := range s.config.Models {
			// Skip if a circuit-breaker is currently holding dispatch back —
			// the model only looks idle because we stopped feeding it.
			if paused, until := s.IsModelPaused(modelID); paused {
				slog.Debug("keepalive skip: inference cooldown active",
					"model", modelID, "resume_at", until.Format(time.RFC3339))
				continue
			}
			if paused, until := s.IsModelLoadPaused(modelID); paused {
				slog.Debug("keepalive skip: load cooldown active",
					"model", modelID, "resume_at", until.Format(time.RFC3339))
				continue
			}

			// Skip if there is pending work queued for this model — evicting
			// now would force an expensive cold reload for jobs we already have.
			pending := 0
			if counts, err := s.store.CountByState(modelID); err == nil {
				pending = counts["queued"] + counts["scheduled"] + counts["running"]
			}
			if pending > 0 {
				slog.Debug("keepalive skip: pending work",
					"model", modelID, "pending_jobs", pending)
				continue
			}

			for _, inst := range s.mgr.GetModelInstances(modelID) {
				st := inst.State()
				active := inst.ActiveJobs()
				if active > 0 {
					continue // NEVER unload while active
				}
				if st == "loading" || st == "unloading" {
					continue // NEVER touch loading/unloading instances
				}
				if st != "loaded" {
					continue
				}
				la := inst.LastActive()
				if la.IsZero() {
					continue
				}
				if now.Sub(la) > time.Duration(cfg.KeepAliveSec)*time.Second {
					idle := now.Sub(la).Seconds()
					slog.Info("keepalive evicting", "instance", inst.InstanceID,
						"idle_seconds", idle, "keep_alive_seconds", cfg.KeepAliveSec)
					if err := inst.Unload(); err != nil {
						slog.Error("keepalive unload failed", "instance", inst.InstanceID, "error", err)
						continue
					}
					s.mgr.ReleaseMemoryFor(inst)
					s.logger.Log("model.evict_done", map[string]any{
						"model_id":    inst.ModelID,
						"instance_id": inst.InstanceID,
						"reason":      "keepalive_expired",
					})
					s.rescoreModel(modelID)
				}
			}
		}
	}
}

// RunJobWatchdog periodically checks for jobs stuck in "running" state past their
// model's max_runtime_seconds and marks them as failed.
func (s *Scheduler) RunJobWatchdog(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}

		jobs, err := s.store.GetRunningJobs()
		if err != nil {
			slog.Warn("watchdog: failed to query running jobs", "error", err)
			continue
		}

		now := nowTS()
		for _, job := range jobs {
			if job.StartedAt == nil {
				continue
			}

			cfg, ok := s.config.Models[job.ModelID]
			if !ok {
				continue
			}

			maxSec := float64(cfg.MaxRuntimeSec)
			elapsed := now - *job.StartedAt
			if maxSec == 0 || elapsed < maxSec {
				continue
			}

			errMsg := fmt.Sprintf("job timed out after %ds (limit %ds)", int(elapsed), cfg.MaxRuntimeSec)
			slog.Warn("watchdog: timing out stuck job",
				"job_id", job.ID,
				"model_id", job.ModelID,
				"elapsed_seconds", int(elapsed),
				"max_runtime_seconds", cfg.MaxRuntimeSec,
			)
			s.store.UpdateState(job.ID, "failed", WithError(errMsg), WithFinishedAt(now))
			s.logger.Log("job.timeout", map[string]any{
				"job_id":              job.ID,
				"model_id":            job.ModelID,
				"elapsed_seconds":     int(elapsed),
				"max_runtime_seconds": cfg.MaxRuntimeSec,
			})
			if n := s.store.ResolveFollowers(job.ID, "failed", nil, errMsg, s.outputDir); n > 0 {
				slog.Info("watchdog: resolved follower jobs", "original", job.ID, "followers", n)
			}
			s.cleanupJobInbox(job)
		}
	}
}

// RunModelHealthWatchdog periodically checks for models in broken states
// and resets them so the next job attempt starts fresh.
func (s *Scheduler) RunModelHealthWatchdog(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}

		for modelID, modelCfg := range s.config.Models {
			for _, inst := range s.mgr.GetModelInstances(modelID) {
				state := inst.State()

				// Check 1: Instance in "error" state with no active jobs — kill and
				// reset to stopped so next ensureLoaded starts fresh.
				if state == "error" && inst.ActiveJobs() == 0 {
					slog.Warn("health watchdog: resetting errored instance",
						"instance", inst.InstanceID, "model", modelID)
					inst.Kill()
					s.mgr.ReleaseMemoryFor(inst)
					s.logger.Log("model.health_reset", map[string]any{
						"model_id":    modelID,
						"instance_id": inst.InstanceID,
						"reason":      "error_state_reset",
					})
				}

				// Check 2: Instance stuck in "loading" for too long.
				// Allowance is the maximum of:
				//   • 3× the configured load_ms (so calibrated estimates win
				//     when they're high — e.g. ltx2-denoise1 at 420s)
				//   • size-based estimate: 60s framework overhead + 8s per GB
				//     of declared memory_gb. Covers reading weights from disk
				//     (~250 MB/s shared FS), framework init, and GPU upload.
				//
				// Examples with the size formula:
				//   1 GB  →  68 s    (small adapter)
				//   17 GB → 196 s   (moondream)
				//   23 GB → 244 s   (z-image-turbo)
				//   40 GB → 380 s   (gemma 26B with 0.32 util)
				//   80 GB → 700 s   (gemma 26B with 0.85 util)
				//
				// The previous 60-second floor was way too aggressive for any
				// non-trivial model — vLLM cold-start of a 13 GB checkpoint
				// alone takes ~30s before it even starts allocating KV cache.
				// Killing during legitimate loading caused the moondream
				// double-spawn cascade described in 2af5feb.
				//
				// The instance's lastActive is reset when state transitions to
				// "loading", so this measures from the start of THIS load
				// attempt only.
				if state == "loading" {
					maxLoadSec := modelCfg.LoadMs / 1000.0 * 3
					sizeBased := 60.0 + modelCfg.MemoryGB*8.0
					if sizeBased > maxLoadSec {
						maxLoadSec = sizeBased
					}
					la := inst.LastActive()
					if !la.IsZero() && time.Since(la).Seconds() > maxLoadSec {
						slog.Warn("health watchdog: loading stuck, killing instance",
							"instance", inst.InstanceID, "model", modelID,
							"loading_seconds", time.Since(la).Seconds(),
							"max_load_sec", maxLoadSec,
							"memory_gb", modelCfg.MemoryGB)
						inst.Kill()
						s.mgr.ReleaseMemoryFor(inst)
						s.logger.Log("model.health_reset", map[string]any{
							"model_id":     modelID,
							"instance_id":  inst.InstanceID,
							"reason":       "loading_stuck",
							"max_load_sec": maxLoadSec,
						})
					}
				}
			}
		}

		// Periodically reconcile following jobs whose originals are stuck
		if n := s.store.ReconcileFollowingJobs(s.outputDir); n > 0 {
			slog.Info("health watchdog: reconciled following jobs", "count", n)
		}
	}
}

// RunVRAMWatchdog periodically checks effective VRAM usage and proactively
// evicts idle models when memory pressure is too high, preventing OOM crashes.
func (s *Scheduler) RunVRAMWatchdog(ctx context.Context) {
	const (
		interval   = 15 * time.Second
		headroomGB = 2.0 // evict when effective usage is within 2GB of budget
	)

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}

		// Reclaim leaked bookkeeping from dead instances every tick. This is
		// the safety net that prevents the failure mode where workers crash
		// without releasing their reservation and the budget appears full
		// forever (queue stalls, 15min circuit-breaker repeats).
		if leaked := s.mgr.ReconcileFromInstances(); leaked > 0 {
			s.logger.Log("vram.reconciled_orphan", map[string]any{
				"freed_gb": leaked,
			})
			s.Wake()
		}

		effective, actual, allocated, reserved := s.mgr.EffectiveUsedGB()
		budget := s.mgr.BudgetGB()
		overage := effective - (budget - headroomGB)

		if overage <= 0 {
			// Reconcile bookkeeping drift even when not under pressure
			s.mgr.ReconcileUsedGB(actual, 1.0)
			continue
		}

		slog.Warn("vram watchdog: memory pressure detected",
			"effective_gb", effective,
			"actual_gb", actual,
			"allocated_gb", allocated,
			"reserved_gb", reserved,
			"budget_gb", budget,
			"overage_gb", overage,
		)
		s.logger.Log("vram.pressure", map[string]any{
			"effective_gb": effective,
			"actual_gb":    actual,
			"allocated_gb": allocated,
			"reserved_gb":  reserved,
			"budget_gb":    budget,
			"overage_gb":   overage,
		})

		// Reconcile bookkeeping before eviction
		s.mgr.ReconcileUsedGB(actual, 1.0)

		// Build queue counts for eviction priority. Include running jobs so
		// that a model actively working through a batch is still counted as
		// "has queued work" between individual job completions — otherwise a
		// model can momentarily show queued=0 (right as one job finishes and
		// the next is picked) and be evicted in favour of a model with zero
		// pending work.
		queuedJobs := make(map[string]int)
		for modelID := range s.config.Models {
			counts, err := s.store.CountByState(modelID)
			if err != nil {
				continue
			}
			queuedJobs[modelID] = counts["queued"] + counts["scheduled"] + counts["running"]
		}

		freed, err := s.mgr.EvictForGBWithQueueInfo(overage, queuedJobs)
		if err != nil {
			slog.Warn("vram watchdog: could not free enough memory",
				"needed_gb", overage, "freed_gb", freed, "error", err)
		} else {
			slog.Info("vram watchdog: eviction complete",
				"freed_gb", freed, "needed_gb", overage)
		}
		s.logger.Log("vram.eviction", map[string]any{
			"freed_gb":  freed,
			"needed_gb": overage,
			"success":   err == nil,
		})

		s.rescoreAll()
	}
}

// cleanupJobInbox is a no-op kept for call-site compatibility.
//
// It used to delete this job's inbox inputs the instant the job reached a
// terminal state, after checking only "is any *currently active* job still
// referencing this file?". That race deletes inputs out from under
// multi-stage clients that batch their submissions: e.g. ltx2 stages a
// boundary jpg, submits encode, encode completes, cleanup deletes the jpg,
// THEN ltx2 submits denoise1 referencing that jpg → "No such file".
//
// Inbox cleanup is now done exclusively by CleanupOrphanedInboxFiles
// (orphan sweep) which only deletes files that have no active references
// AND are older than inboxOrphanMinAge. That gives every file a grace
// period in which a follow-up job can reference it before it's eligible
// for deletion.
func (s *Scheduler) cleanupJobInbox(job *Job) {
	_ = job
}

// inboxOrphanMinAge is how old an inbox file must be before the orphan
// sweep is allowed to delete it. Sized to cover the worst-case pipeline
// latency under heavy queue contention — e.g. an ltx2 music_video that
// stages boundary jpgs, then submits encode → defect-check (gemma) →
// denoise1 → denoise2 in serial waves. With shared GPU contention from
// other workloads (tts queues etc.) the gap between first submission
// and last reference can run several hours. Will be replaced by the
// arbiter-client lease ledger; 24h is the safe-by-default conservative
// fallback in the meantime.
const inboxOrphanMinAge = 24 * time.Hour

// ValidateJobInputs returns a detailed error if any file path declared in
// the payload can't be opened. Checks every absolute path under known
// share-mount prefixes — both inbox/ (originally-staged inputs) AND
// output/ (outputs of previous pipeline stages, e.g. ltx2-encode →
// ltx2-denoise1 reading encoded.pt). Each path is os.Stat'd and the
// specific failure (missing / broken symlink / permission / I/O) is
// surfaced verbatim so the caller knows exactly what's wrong.
//
// Called at job-submission time so a job with bad paths is rejected with
// HTTP 400 BEFORE entering the queue — no model load, no dispatch, no
// circuit-breaker risk. Also called by the inbox watchdog as a safety
// net for paths that disappear after submission.
func (s *Scheduler) ValidateJobInputs(payload json.RawMessage) error {
	prefixes := s.checkablePathPrefixes()
	if len(prefixes) == 0 {
		return nil
	}
	paths := extractCheckablePaths(payload, prefixes)
	if len(paths) == 0 {
		return nil
	}
	var failures []string
	for _, p := range paths {
		if _, err := os.Stat(p); err != nil {
			// Build a precise reason. os.Stat returns *PathError whose
			// inner err encodes the syscall errno (ENOENT, EINVAL, etc.).
			// Surface the underlying error string verbatim so the caller
			// sees e.g. "no such file or directory" vs "invalid argument
			// (broken symlink on CIFS — target was deleted)".
			reason := classifyStatError(p, err)
			failures = append(failures, fmt.Sprintf("%s: %s", p, reason))
		}
	}
	if len(failures) > 0 {
		return fmt.Errorf("job rejected: %d input path(s) unreadable:\n  %s",
			len(failures), strings.Join(failures, "\n  "))
	}
	return nil
}

// checkablePathPrefixes returns the absolute path prefixes the validator
// will scan for. Currently the share mount root (covers inbox/ and
// output/) and the local output directory (for non-share deployments).
func (s *Scheduler) checkablePathPrefixes() []string {
	var out []string
	if s.config != nil && s.config.ShareMount != "" {
		out = append(out, strings.TrimRight(s.config.ShareMount, "/")+"/")
	}
	if s.outputDir != "" {
		clean := strings.TrimRight(s.outputDir, "/") + "/"
		// Avoid a duplicate when outputDir is under ShareMount.
		dup := false
		for _, p := range out {
			if strings.HasPrefix(clean, p) {
				dup = true
				break
			}
		}
		if !dup {
			out = append(out, clean)
		}
	}
	return out
}

// extractCheckablePaths walks the payload for any string starting with one
// of the supplied absolute prefixes.
func extractCheckablePaths(payload json.RawMessage, prefixes []string) []string {
	var v any
	if err := json.Unmarshal(payload, &v); err != nil {
		return nil
	}
	var out []string
	walkJSONMulti(v, prefixes, &out)
	return out
}

func walkJSONMulti(v any, prefixes []string, out *[]string) {
	switch t := v.(type) {
	case string:
		for _, p := range prefixes {
			if strings.HasPrefix(t, p) {
				*out = append(*out, t)
				return
			}
		}
	case map[string]any:
		for _, val := range t {
			walkJSONMulti(val, prefixes, out)
		}
	case []any:
		for _, val := range t {
			walkJSONMulti(val, prefixes, out)
		}
	}
}

// classifyStatError turns a stat() error into a human-readable reason
// that explains what's actually wrong, not just "stat failed".
func classifyStatError(path string, err error) string {
	if os.IsNotExist(err) {
		return "no such file or directory"
	}
	if os.IsPermission(err) {
		return "permission denied"
	}
	low := strings.ToLower(err.Error())
	switch {
	case strings.Contains(low, "invalid argument"):
		// On CIFS with symlink=native, EINVAL is what you get for a
		// symlink whose target was deleted. Spell that out — the caller
		// almost certainly cares (this is the dedup-symlink lifecycle bug).
		if lst, lerr := os.Lstat(path); lerr == nil && lst.Mode()&os.ModeSymlink != 0 {
			tgt, _ := os.Readlink(path)
			return fmt.Sprintf("invalid argument — broken symlink → %q (target gone or unreachable on this mount)", tgt)
		}
		return "invalid argument (likely broken symlink or unmounted path)"
	case strings.Contains(low, "stale file handle"):
		return "stale NFS/CIFS file handle (mount needs refresh)"
	case strings.Contains(low, "i/o error"):
		return "I/O error (storage / mount problem)"
	}
	return err.Error()
}

// isClientError returns true when an inference error string indicates a
// fatal client-side problem (bad input, missing/unreadable file, broken
// dependency from a previous pipeline stage) rather than a model fault.
// These should NOT count toward the inference circuit breaker — the model
// is fine, the caller submitted bad data. Penalising the model would pause
// it for 15 minutes and starve every other queued job.
//
// Watched errno values:
//   2  ENOENT  — no such file or directory
//   5  EIO     — I/O error (often stale NFS/CIFS handle)
//   13 EACCES  — permission denied
//   22 EINVAL  — invalid argument (broken symlink on CIFS, bad path syntax)
//   116 ESTALE — stale file handle (NFS specific)
//
// All of these mean "the bytes the worker wanted to read aren't reachable"
// — never a model bug, always upstream of the worker.
func isClientError(s string) bool {
	low := strings.ToLower(s)
	switch {
	case strings.Contains(low, "no such file or directory"),
		strings.Contains(low, "filenotfounderror"),
		strings.Contains(low, "input file(s) missing"),
		strings.Contains(low, "stale file handle"),
		strings.Contains(low, "permission denied"),
		strings.Contains(low, "i/o error"),
		strings.Contains(low, "invalid argument") && strings.Contains(low, "/mnt/") ||
			strings.Contains(low, "invalid argument") && strings.Contains(low, "/output/"),
		strings.Contains(low, "[errno 2]"),
		strings.Contains(low, "[errno 5]"),
		strings.Contains(low, "[errno 13]"),
		strings.Contains(low, "[errno 22]"),
		strings.Contains(low, "[errno 116]"):
		return true
	}
	return false
}

// RunInboxWatchdog periodically cleans orphaned inbox files AND fails any
// queued jobs whose declared inputs have disappeared. The two operations
// share the same active-job snapshot, and failing missing-input jobs here
// (rather than at dispatch) prevents 10 such jobs from cascading into a
// 15-minute inference circuit-breaker trip.
//
// Cadence: every 60 seconds. Cheap when the queue is small.
func (s *Scheduler) RunInboxWatchdog(ctx context.Context) {
	if s.inboxDir == "" {
		return
	}
	t := time.NewTicker(60 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if _, err := s.CleanupOrphanedInboxFiles(); err != nil {
				slog.Warn("inbox watchdog: cleanup failed", "error", err)
			}
		}
	}
}

// CleanupOrphanedInboxFiles removes files from the inbox that are not
// referenced by any queued, scheduled, running, or following job AND are
// older than inboxOrphanMinAge. After deletion runs, also fail any queued
// jobs whose declared input files no longer exist — that prevents the
// worker from cascading 10 "file not found" errors and tripping the
// inference circuit breaker.
func (s *Scheduler) CleanupOrphanedInboxFiles() (int, error) {
	if s.inboxDir == "" {
		return 0, nil
	}

	// Collect all inbox paths referenced by active jobs.
	activeJobs, err := s.store.GetActiveJobs()
	if err != nil {
		return 0, fmt.Errorf("get active jobs: %w", err)
	}
	referenced := make(map[string]bool)
	for _, j := range activeJobs {
		for _, f := range extractInboxPaths(j.Payload, s.inboxDir) {
			referenced[f] = true
		}
	}

	entries, err := os.ReadDir(s.inboxDir)
	if err != nil {
		return 0, fmt.Errorf("read inbox dir: %w", err)
	}

	deleted := 0
	cutoff := time.Now().Add(-inboxOrphanMinAge)
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		path := filepath.Join(s.inboxDir, e.Name())
		if referenced[path] {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if info.ModTime().After(cutoff) {
			// Too young — give multi-stage clients a chance to register
			// follow-up jobs that reference this file.
			continue
		}
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			slog.Warn("inbox cleanup: failed to remove orphan", "file", path, "error", err)
		} else if err == nil {
			deleted++
		}
	}

	// Inverse sweep: any queued job whose declared inputs are now missing
	// will fail the moment it dispatches. Better to fail it here with a
	// clear error than let 10 such jobs trip the circuit breaker. Always
	// runs (even when nothing was deleted this cycle) — files can also
	// disappear due to external causes (mount blip, manual cleanup).
	failed := s.failQueuedJobsWithMissingInputs(activeJobs)
	if failed > 0 {
		slog.Warn("inbox cleanup: failed queued jobs with missing inputs", "count", failed)
	}
	return deleted, nil
}

// failQueuedJobsWithMissingInputs scans the supplied (active) jobs and marks
// any in state "queued" as failed if any declared input file is unreadable.
// Uses the same ValidateJobInputs check as job submission, so the inverse
// sweep catches paths that were valid at submission but disappeared since
// (e.g. a previous pipeline stage's output dir cleaned up, mount blip, etc).
// Returns the number of jobs failed. Caller passes the list to avoid a
// redundant store query.
func (s *Scheduler) failQueuedJobsWithMissingInputs(active []*Job) int {
	if len(s.checkablePathPrefixes()) == 0 {
		return 0
	}
	failed := 0
	for _, j := range active {
		if j.State != "queued" {
			continue
		}
		err := s.ValidateJobInputs(j.Payload)
		if err == nil {
			continue
		}
		errMsg := err.Error()
		s.store.UpdateState(j.ID, "failed",
			WithError(errMsg), WithFinishedAt(nowTS()))
		s.logger.Log("job.failed", map[string]any{
			"job_id":   j.ID,
			"model_id": j.ModelID,
			"reason":   "unreachable_inputs",
			"detail":   errMsg,
		})
		slog.Warn("failing queued job — input(s) unreachable",
			"job", j.ID, "model", j.ModelID, "error", errMsg)
		failed++
	}
	return failed
}

// extractInboxPaths walks a JSON payload and returns every string value that
// starts with inboxDir.
func extractInboxPaths(payload json.RawMessage, inboxDir string) []string {
	var v any
	if err := json.Unmarshal(payload, &v); err != nil {
		return nil
	}
	prefix := inboxDir + "/"
	var out []string
	walkJSON(v, prefix, &out)
	return out
}

func walkJSON(v any, prefix string, out *[]string) {
	switch t := v.(type) {
	case string:
		if strings.HasPrefix(t, prefix) {
			*out = append(*out, t)
		}
	case map[string]any:
		for _, val := range t {
			walkJSON(val, prefix, out)
		}
	case []any:
		for _, val := range t {
			walkJSON(val, prefix, out)
		}
	}
}
