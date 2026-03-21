package main

import (
	"context"
	"fmt"
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
	wake     chan struct{}
}

func NewScheduler(cfg *Config, store *Store, mgr *InstanceManager, logger *EventLogger, outputDir string) *Scheduler {
	return &Scheduler{
		config:    cfg,
		store:     store,
		mgr:       mgr,
		logger:    logger,
		outputDir: outputDir,
		wake:      make(chan struct{}, 1),
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
	for modelID, cfg := range s.config.Models {
		active, _ := s.store.CountActive(modelID)
		capacity := cfg.MaxInstances * cfg.MaxConcurrent
		if active >= capacity {
			full[modelID] = true
		}
	}
	return full
}

// ensureLoaded makes sure an instance is loaded, evicting other models if needed.
func (s *Scheduler) ensureLoaded(inst *Instance) error {
	state := inst.State()
	if state == "loaded" {
		return nil
	}

	// Need to load — check VRAM budget
	if state == "stopped" || state == "unloaded" || state == "error" {
		if !s.mgr.ReserveMemory(inst.memoryGB) {
			// Try to evict
			deficit := inst.memoryGB - s.mgr.FreeGB()
			if err := s.mgr.EvictForGB(deficit); err != nil {
				return fmt.Errorf("can't load %s: %w", inst.InstanceID, err)
			}
			if !s.mgr.ReserveMemory(inst.memoryGB) {
				return fmt.Errorf("can't reserve memory for %s after eviction", inst.InstanceID)
			}
		}

		s.logger.Log("model.load_start", map[string]any{
			"model_id":    inst.ModelID,
			"instance_id": inst.InstanceID,
			"memory_gb":   inst.memoryGB,
		})

		if err := inst.Load("cuda"); err != nil {
			s.mgr.ReleaseMemory(inst.memoryGB)
			s.logger.Log("model.load_error", map[string]any{
				"model_id":    inst.ModelID,
				"instance_id": inst.InstanceID,
				"error":       err.Error(),
			})
			return err
		}

		s.logger.Log("model.load_done", map[string]any{
			"model_id":    inst.ModelID,
			"instance_id": inst.InstanceID,
			"memory_gb":   inst.memoryGB,
		})
		s.rescoreModel(inst.ModelID)
	}

	return nil
}

// dispatchJob picks an instance, loads it, and runs inference.
func (s *Scheduler) dispatchJob(job *Job) {
	modelCfg, ok := s.config.Models[job.ModelID]
	if !ok {
		s.store.UpdateState(job.ID, "failed", WithError("model not configured"), WithFinishedAt(nowTS()))
		return
	}
	_ = modelCfg

	s.logger.Log("job.scheduled", map[string]any{"job_id": job.ID, "model_id": job.ModelID})

	// Pick best instance
	inst := s.mgr.PickInstance(job.ModelID)
	if inst == nil {
		slog.Debug("no instance available, requeueing", "model", job.ModelID, "job", job.ID)
		s.store.UpdateState(job.ID, "queued")
		return
	}

	// Ensure loaded
	if err := s.ensureLoaded(inst); err != nil {
		slog.Warn("can't load instance, requeueing", "instance", inst.InstanceID, "error", err)
		s.store.UpdateState(job.ID, "queued")
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

	// Run inference (blocking)
	jobDir := filepath.Join(s.outputDir, "jobs", job.ID)
	os.MkdirAll(jobDir, 0o755)

	start := time.Now()
	resp, err := inst.Infer(job.ID, job.JobType, job.Payload, jobDir)
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

	s.rescoreModel(job.ModelID)
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

	if s.mgr.FreeGB() >= inst.memoryGB {
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

		// Dispatch blocks until inference completes — but runs in a goroutine
		// so the scheduler can continue picking more jobs
		go func(j *Job) {
			s.dispatchJob(j)
			s.Wake() // signal scheduler to check for more work
		}(job)

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
				if inst.State() != "loaded" || inst.ActiveJobs() > 0 {
					continue
				}
				la := inst.LastActive()
				if la.IsZero() {
					continue
				}
				if now.Sub(la) > time.Duration(cfg.KeepAliveSec)*time.Second {
					slog.Info("keepalive evicting", "instance", inst.InstanceID)
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
