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
	dispatchPaused  atomic.Bool // benchmark mode: queue grows but no dispatch
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

// PauseDispatch halts new job dispatch. Queued jobs remain queued; in-flight
// jobs run to completion. Used for benchmark mode so external work pauses
// without being lost. Resets to false on process restart.
func (s *Scheduler) PauseDispatch() {
	s.dispatchPaused.Store(true)
}

// ResumeDispatch resumes normal dispatch. Wakes the scheduler so the queue
// drains immediately.
func (s *Scheduler) ResumeDispatch() {
	s.dispatchPaused.Store(false)
	s.Wake()
}

// IsDispatchPaused reports whether benchmark mode is active.
func (s *Scheduler) IsDispatchPaused() bool {
	return s.dispatchPaused.Load()
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
		active, _ := s.store.CountActive(modelID)
		capacity := *cfg.MaxInstances * cfg.MaxConcurrent
		if active >= capacity {
			slog.Debug("scheduler.full: model at capacity",
				"model", modelID, "active", active, "capacity", capacity,
				"max_concurrent", cfg.MaxConcurrent, "max_instances", *cfg.MaxInstances)
			full[modelID] = true
			continue
		}
		if cp+*cfg.PressureIndex > 1.0+1e-9 {
			slog.Debug("scheduler.full: model would exceed pressure budget",
				"model", modelID, "current_pressure", cp,
				"model_pressure_index", *cfg.PressureIndex,
				"sum", cp+*cfg.PressureIndex, "budget", 1.0)
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
		s.RecordFailure(job.ModelID)
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
		s.RecordFailure(job.ModelID)
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

// tryPreload speculatively loads the next needed instance in the background.
func (s *Scheduler) tryPreload() {
	full := s.getFullModels()
	job, _ := s.store.PickNextJob(full)
	if job == nil {
		return
	}

	inst := s.mgr.PickInstance(job.ModelID)
	if inst == nil {
		return
	}

	state := inst.State()
	if state == "loaded" || state == "loading" {
		return
	}

	if s.mgr.FreeGB() >= inst.memoryGB { // only preload if fits under budget
		slog.Debug("preloading", "instance", inst.InstanceID)
		go func() {
			if err := s.ensureLoaded(inst); err != nil {
				slog.Debug("preload failed", "instance", inst.InstanceID, "error", err)
			}
		}()
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

		// Benchmark mode: queue still accepts jobs but dispatch is suspended.
		// In-flight jobs already dispatched run to completion via their own
		// goroutines; only this picker stops handing out new work.
		if s.dispatchPaused.Load() {
			continue
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
		slog.Debug("pressure reserved", "model", job.ModelID, "pressure", pressure, "total", s.currentPressure)

		go func(j *Job, inst *Instance, pressure float64) {
			s.dispatchJobToInstance(j, inst, pressure)
			s.Wake()
		}(job, inst, pressure)

		// Preload next instance in background
		s.tryPreload()
	}
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
				// Allowance is 3× the configured load_ms with a 60-second floor.
				// No upper clamp: ltx2-denoise1's calibrated load_ms is 420s, so
				// the previous 600s clamp left almost no headroom for variance and
				// would kill perfectly healthy slow loads. The instance's
				// lastActive is reset when state transitions to "loading", so this
				// measures from the start of THIS load attempt only.
				if state == "loading" {
					maxLoadSec := modelCfg.LoadMs / 1000.0 * 3
					if maxLoadSec < 60 {
						maxLoadSec = 60
					}
					la := inst.LastActive()
					if !la.IsZero() && time.Since(la).Seconds() > maxLoadSec {
						slog.Warn("health watchdog: loading stuck, killing instance",
							"instance", inst.InstanceID, "model", modelID,
							"loading_seconds", time.Since(la).Seconds())
						inst.Kill()
						s.mgr.ReleaseMemoryFor(inst)
						s.logger.Log("model.health_reset", map[string]any{
							"model_id":    modelID,
							"instance_id": inst.InstanceID,
							"reason":      "loading_stuck",
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

// cleanupJobInbox deletes any files in the inbox directory that are referenced
// by this job's params, provided no other active job also references them.
// Called once a job reaches a terminal state (completed, failed, cancelled)
// but NOT on requeue.
func (s *Scheduler) cleanupJobInbox(job *Job) {
	if s.inboxDir == "" || len(job.Payload) == 0 {
		return
	}
	files := extractInboxPaths(job.Payload, s.inboxDir)
	if len(files) == 0 {
		return
	}

	// Build set of inbox paths still referenced by other active jobs.
	activeJobs, err := s.store.GetActiveJobs()
	if err != nil {
		slog.Warn("inbox cleanup: skipping, failed to query active jobs", "error", err)
		return
	}
	stillReferenced := make(map[string]bool)
	for _, j := range activeJobs {
		if j.ID == job.ID {
			continue // skip the job we're cleaning up
		}
		for _, f := range extractInboxPaths(j.Payload, s.inboxDir) {
			stillReferenced[f] = true
		}
	}

	for _, f := range files {
		if stillReferenced[f] {
			slog.Debug("inbox cleanup: skipping file still referenced by queued jobs", "file", f)
			continue
		}
		if err := os.Remove(f); err != nil && !os.IsNotExist(err) {
			slog.Warn("inbox cleanup: failed to remove file", "file", f, "error", err)
		} else if err == nil {
			slog.Debug("inbox cleanup: removed", "file", f)
		}
	}
}

// CleanupOrphanedInboxFiles removes files from the inbox that are not
// referenced by any queued, scheduled, running, or following job.
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
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		path := filepath.Join(s.inboxDir, e.Name())
		if referenced[path] {
			continue
		}
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			slog.Warn("inbox cleanup: failed to remove orphan", "file", path, "error", err)
		} else if err == nil {
			deleted++
		}
	}
	return deleted, nil
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
