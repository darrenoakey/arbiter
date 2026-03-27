package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"log/slog"
	"os"
	"path/filepath"
	"time"
)

type Scheduler struct {
	config   *Config
	store    *Store
	mgr      *InstanceManager
	logger   *EventLogger
	outputDir string
	wake          chan struct{}
	cooldownMu    sync.Mutex
	cooldownUntil map[string]time.Time // model -> skip until this time
}

func NewScheduler(cfg *Config, store *Store, mgr *InstanceManager, logger *EventLogger, outputDir string) *Scheduler {
	return &Scheduler{
		config:    cfg,
		store:     store,
		mgr:       mgr,
		logger:    logger,
		outputDir: outputDir,
		wake:          make(chan struct{}, 1),
		cooldownUntil: make(map[string]time.Time),
	}
}

// Wake signals the scheduler to check for new work.
func (s *Scheduler) Wake() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
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

// getFullModels returns model IDs that are at total capacity.
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
	for modelID, cfg := range s.config.Models {
		active, _ := s.store.CountActive(modelID)
		capacity := *cfg.MaxInstances * cfg.MaxConcurrent
		if active >= capacity {
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
// activeJobs is already incremented by the caller to reserve the slot.
// This function owns the reservation and releases it when done.
func (s *Scheduler) dispatchJobToInstance(job *Job, inst *Instance) {
	defer func() {
		s.mgr.ReleaseAndCheck(inst)
		s.rescoreModel(job.ModelID)
	}()

	slog.Info("dispatching job", "job_id", job.ID, "model", job.ModelID, "instance", inst.InstanceID)
	s.logger.Log("job.scheduled", map[string]any{"job_id": job.ID, "model_id": job.ModelID, "instance_id": inst.InstanceID})

	// Ensure loaded
	if err := s.ensureLoaded(inst); err != nil {
		slog.Warn("can't load instance, requeueing", "instance", inst.InstanceID, "job", job.ID, "error", err)
		s.store.UpdateState(job.ID, "queued")
		// Cooldown: mark model as temporarily full to prevent scheduler spin
		s.cooldownMu.Lock()
		s.cooldownUntil[job.ModelID] = time.Now().Add(5 * time.Second)
		s.cooldownMu.Unlock()
		return
	}

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

	start := time.Now()
	resp, err := inst.InferRaw(job.ID, job.JobType, job.Payload, jobDir)
	elapsed := time.Since(start).Seconds()

	if err != nil {
		errMsg := fmt.Sprintf("inference error: %s", err)
		s.store.UpdateState(job.ID, "failed", WithError(errMsg), WithFinishedAt(nowTS()))
		s.logger.Log("job.failed", map[string]any{
			"job_id":            job.ID,
			"model_id":         job.ModelID,
			"error":            errMsg,
			"inference_seconds": elapsed,
		})
		slog.Error("job failed", "job", job.ID, "error", err)
		return
	}

	if resp.Status == "cancelled" {
		s.store.UpdateState(job.ID, "cancelled", WithFinishedAt(nowTS()))
		s.logger.Log("job.cancelled", map[string]any{"job_id": job.ID, "model_id": job.ModelID})
	} else if resp.Status == "error" {
		s.store.UpdateState(job.ID, "failed", WithError(resp.Error), WithFinishedAt(nowTS()))
		s.logger.Log("job.failed", map[string]any{
			"job_id":            job.ID,
			"model_id":         job.ModelID,
			"error":            resp.Error,
			"inference_seconds": elapsed,
		})
	} else {
		s.store.UpdateState(job.ID, "completed", WithResult(resp.Result), WithFinishedAt(nowTS()))
		s.logger.Log("job.completed", map[string]any{
			"job_id":            job.ID,
			"model_id":         job.ModelID,
			"inference_seconds": elapsed,
		})
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

		go func(j *Job, inst *Instance) {
			s.dispatchJobToInstance(j, inst)
			s.Wake()
		}(job, inst)

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
