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
}

func NewScheduler(cfg *Config, store *Store, mgr *InstanceManager, logger *EventLogger, outputDir string) *Scheduler {
	inboxDir := ""
	if cfg.ShareMount != "" {
		inboxDir = filepath.Join(cfg.ShareMount, "inbox")
	}
	return &Scheduler{
		config:               cfg,
		store:                store,
		mgr:                  mgr,
		logger:               logger,
		outputDir:            outputDir,
		inboxDir:             inboxDir,
		wake:                 make(chan struct{}, 1),
		cooldownUntil:            make(map[string]time.Time),
		failureCount:             make(map[string]int),
		failurePaused:            make(map[string]time.Time),
		failureCooldownLevel:     make(map[string]int),
		loadFailureCount:         make(map[string]int),
		loadFailurePaused:        make(map[string]time.Time),
		loadFailureCooldownLevel: make(map[string]int),
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
// Threshold: 3 failures → escalating pause. On activation, cancels queued+following jobs.
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
	slog.Warn("load circuit-breaker: model paused after consecutive load failures",
		"model", modelID, "threshold", threshold, "cooldown", dur,
		"resume_at", until.Format(time.RFC3339))
	s.logger.Log("model.load_circuit_breaker", map[string]any{
		"model_id": modelID,
		"cooldown": dur.String(),
	})
	// Cancel stuck jobs in background
	go func() {
		if n, err := s.store.CancelQueuedForModel(modelID); err == nil && n > 0 {
			slog.Warn("load circuit-breaker: cancelled queued jobs", "model", modelID, "count", n)
		}
		if n, err := s.store.CancelFollowingForModel(modelID, "model load circuit-breaker activated"); err == nil && n > 0 {
			slog.Warn("load circuit-breaker: cancelled following jobs", "model", modelID, "count", n)
		}
	}()
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
			full[modelID] = true
			continue
		}
		if cp+cfg.PressureIndex > 1.0+1e-9 {
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
		slog.Info("ensureLoaded: already loading", "instance", inst.InstanceID)
		return fmt.Errorf("instance %s is already loading", inst.InstanceID)
	}

	if state == "stopped" || state == "unloaded" || state == "error" {
		needed := inst.memoryGB
		freeGB := s.mgr.FreeGB()

		slog.Info("ensureLoaded: need VRAM", "instance", inst.InstanceID,
			"needed_gb", needed, "free_gb", freeGB, "state", state)

		if state == "error" {
			s.mgr.ReleaseMemory(needed)
		}

		// Try reserve
		if !s.mgr.ReserveMemory(needed) {
			// Evict idle models
			deficit := needed - s.mgr.FreeGB()
			if deficit > 0 {
				slog.Info("ensureLoaded: evicting idle models", "instance", inst.InstanceID, "deficit_gb", deficit)
				s.mgr.EvictForGB(deficit)
			}

			// Retry
			if !s.mgr.ReserveMemory(needed) {
				slog.Warn("ensureLoaded: can't reserve VRAM after eviction",
					"instance", inst.InstanceID, "needed_gb", needed, "free_gb", s.mgr.FreeGB())
				return fmt.Errorf("can't load %s: need %.1fGB, only %.1fGB free",
					inst.InstanceID, needed, s.mgr.FreeGB())
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
			s.mgr.ReleaseMemory(inst.memoryGB)
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
		slog.Warn("can't load instance, requeueing", "instance", inst.InstanceID, "job", job.ID, "error", err)
		s.store.UpdateState(job.ID, "queued")
		s.RecordLoadFailure(job.ModelID)
		// Cooldown: mark model as temporarily full to prevent scheduler spin
		s.cooldownMu.Lock()
		s.cooldownUntil[job.ModelID] = time.Now().Add(5 * time.Second)
		s.cooldownMu.Unlock()
		return
	}
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
		s.store.UpdateState(job.ID, "completed", WithResult(resp.Result), WithFinishedAt(nowTS()))
		rssEntry := map[string]any{
			"job_id":            job.ID,
			"model_id":          job.ModelID,
			"inference_seconds": elapsed,
		}
		if rss := inst.RSSAnon(); rss > 0 {
			rssEntry["worker_rss_anon_mb"] = rss
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

		// Pick and dispatch one job at a time
		full := s.getFullModels()
		job, err := s.store.PickNextJob(full)
		if err != nil || job == nil {
			continue
		}

		// Mark scheduled so it won't be re-picked
		s.store.UpdateState(job.ID, "scheduled")

		// Pick instance NOW (synchronous) so concurrent goroutines
		// don't race to pick the same instance
		inst := s.mgr.PickInstance(job.ModelID)
		if inst == nil {
			slog.Debug("no instance available, requeueing", "job", job.ID, "model", job.ModelID)
			s.store.UpdateState(job.ID, "queued")
			continue
		}
		slog.Info("picked instance for job", "job", job.ID, "model", job.ModelID,
			"instance", inst.InstanceID, "state", inst.State(), "active_jobs", inst.ActiveJobs())
		// Reserve the slot immediately so PickInstance won't return it again
		atomic.AddInt32(&inst.activeJobs, 1)

		// Reserve pressure immediately (main loop is single-threaded for dispatch decisions)
		pressure := s.config.Models[job.ModelID].PressureIndex
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

// RunKeepalive evicts idle models past their keep_alive_seconds.
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
					s.mgr.ReleaseMemory(inst.memoryGB)
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
			if elapsed < maxSec {
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
					s.mgr.ReleaseMemory(inst.memoryGB)
					s.logger.Log("model.health_reset", map[string]any{
						"model_id":    modelID,
						"instance_id": inst.InstanceID,
						"reason":      "error_state_reset",
					})
				}

				// Check 2: Instance stuck in "loading" for too long.
				if state == "loading" {
					maxLoadSec := modelCfg.LoadMs / 1000.0 * 2
					if maxLoadSec < 60 {
						maxLoadSec = 60
					}
					if maxLoadSec > 600 {
						maxLoadSec = 600
					}
					la := inst.LastActive()
					if !la.IsZero() && time.Since(la).Seconds() > maxLoadSec {
						slog.Warn("health watchdog: loading stuck, killing instance",
							"instance", inst.InstanceID, "model", modelID,
							"loading_seconds", time.Since(la).Seconds())
						inst.Kill()
						s.mgr.ReleaseMemory(inst.memoryGB)
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

		// Build queue counts for eviction priority
		queuedJobs := make(map[string]int)
		for modelID := range s.config.Models {
			counts, err := s.store.CountByState(modelID)
			if err != nil {
				continue
			}
			queuedJobs[modelID] = counts["queued"] + counts["scheduled"]
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
