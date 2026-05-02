package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// reapOrphanWorkers kills any worker subprocesses left over from a previous
// arbiter lifetime. When the Go server is killed (crash, SIGKILL, auto-restart
// timeout), its worker children are reparented to init and keep holding VRAM
// and ports. Next arbiter startup can't evict them because it doesn't own them,
// and can't reserve VRAM because they're consuming it — so every subsequent
// load fails with "VRAM reserve failed", the circuit breaker trips after 3
// tries, and every queued job is parked indefinitely.
//
// The reaper finds any llama-server, vllm-chat-worker, or arbiter.worker_main
// process that isn't a child of this arbiter pid and SIGKILLs it. This is safe:
// those binaries only run as arbiter-spawned workers. User-launched ones would
// be caught by pgrep too, but this is a machine-dedicated role (spark) so any
// match is an orphan from a prior arbiter.
func reapOrphanWorkers() {
	patterns := []string{
		"llama.cpp/build/bin/llama-server",
		"arbiter/vllm-chat-worker",
		"arbiter/llm-worker",
		"arbiter.worker_main",
	}
	selfPid := os.Getpid()
	killed := 0
	for _, pattern := range patterns {
		output, err := exec.Command("pgrep", "-f", pattern).Output()
		if err != nil {
			continue
		}
		for _, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
			if line == "" {
				continue
			}
			pid, err := strconv.Atoi(line)
			if err != nil || pid == selfPid {
				continue
			}
			if err := syscall.Kill(pid, syscall.SIGKILL); err == nil {
				slog.Warn("reaped orphan worker from prior arbiter", "pid", pid, "pattern", pattern)
				killed++
			}
		}
	}
	if killed > 0 {
		time.Sleep(2 * time.Second) // let VRAM actually release before the scheduler starts
	}
}

func main() {
	// Protect this process from the OOM killer — worker subprocesses are set to
	// OomScoreAdj=500 (in proc.go) so the kernel will kill them first.
	if err := os.WriteFile("/proc/self/oom_score_adj", []byte("-200"), 0o644); err != nil {
		// Non-fatal: may not have permission in all environments
		slog.Warn("could not set oom_score_adj", "error", err)
	}

	// Structured logging to stderr
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo})))

	// Kill any leftover worker subprocesses from a prior arbiter lifetime.
	// If this arbiter was hard-killed, its workers survive as init-children
	// and keep holding VRAM, which makes every subsequent model load fail.
	reapOrphanWorkers()

	// Find project root (directory containing go.mod)
	projectRoot, _ := filepath.Abs(".")
	if _, err := os.Stat(filepath.Join(projectRoot, "go.mod")); os.IsNotExist(err) {
		exe, _ := os.Executable()
		projectRoot = filepath.Dir(exe)
	}

	// Load config
	cfg, err := LoadConfig(projectRoot)
	if err != nil {
		slog.Error("failed to load config", "error", err)
		os.Exit(1)
	}

	// Output dirs — jobs/logs/refs go to ARBITER_OUTPUT_DIR if set (e.g. network share),
	// DB stays local (SQLite over network is unsafe).
	outputDir := filepath.Join(projectRoot, "output")
	if cfg.OutputDir != "" {
		outputDir = cfg.OutputDir
	}
	os.MkdirAll(filepath.Join(outputDir, "jobs"), 0o755)
	os.MkdirAll(filepath.Join(outputDir, "logs"), 0o755)
	os.MkdirAll(filepath.Join(outputDir, "refs"), 0o755)

	// Event logger
	eventLog := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer eventLog.Close()

	// Store — DB always on local disk regardless of ARBITER_OUTPUT_DIR
	dbPath := filepath.Join(projectRoot, "output", "arbiter.db")
	store, err := NewStore(dbPath)
	if err != nil {
		slog.Error("failed to open store", "error", err)
		os.Exit(1)
	}
	defer store.Close()

	recovered, _ := store.RecoverFromCrash()
	if recovered > 0 {
		slog.Info("recovered jobs from crash", "count", recovered)
	}

	// Python path
	pythonBin := filepath.Join(projectRoot, ".venv", "bin", "python")
	if _, err := os.Stat(pythonBin); os.IsNotExist(err) {
		pythonBin = "python3"
	}

	// Instance manager
	mgr := NewInstanceManager(cfg, pythonBin, projectRoot)
	setupInstances(cfg, mgr, pythonBin, projectRoot)

	// Register job type mappings for LLM models (restored from config)
	for modelID := range cfg.Models {
		if strings.HasPrefix(modelID, "llm:") {
			name := strings.TrimPrefix(modelID, "llm:")
			JobTypeToModel["chat-completion:"+name] = modelID
			slog.Info("registered LLM job type", "type", "chat-completion:"+name, "model", modelID)
		}
	}

	// Dedup cache
	store.InitDedup()

	// Re-populate dedup cache from queued jobs and remove duplicates
	deduped := store.DedupRecoveredJobs()
	if deduped > 0 {
		slog.Info("deduped recovered jobs", "removed", deduped)
	}
	reconciledFollowers := store.ReconcileFollowingJobs(outputDir)
	if reconciledFollowers > 0 {
		slog.Info("reconciled follower jobs", "count", reconciledFollowers)
	}

	// Scheduler
	sched := NewScheduler(cfg, store, mgr, eventLog, outputDir)

	// Remove inbox files that belong to no active job (runs in background so
	// it doesn't delay server startup — can be thousands of files after crashes).
	go func() {
		if n, err := sched.CleanupOrphanedInboxFiles(); err != nil {
			slog.Warn("inbox orphan cleanup failed", "error", err)
		} else if n > 0 {
			slog.Info("inbox orphan cleanup: removed files", "count", n)
		}
	}()

	// API
	api := NewAPI(cfg, store, mgr, sched, eventLog, outputDir, projectRoot)

	// Context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start background goroutines
	go sched.Run(ctx)
	go sched.RunScheduledWatchdog(ctx)
	go sched.RunKeepalive(ctx)
	go sched.RunJobWatchdog(ctx)
	go sched.RunModelHealthWatchdog(ctx)
	go sched.RunVRAMWatchdog(ctx)
	go sched.RunInboxWatchdog(ctx)
	go NewMemoryWatchdog(cfg, mgr, eventLog, projectRoot).Run(ctx, 30*time.Second)
	if cfg.ShareMount != "" {
		// Probe the mount root itself — contains inbox/ and output/ subdirs,
		// so ReadDir always gets a real server round-trip.
		go RunShareWatchdog(ctx, cfg.ShareMount, cfg.ShareMount, eventLog)
	}

	// Dedup cache cleanup (hourly)
	go func() {
		ticker := time.NewTicker(time.Hour)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				n := store.DedupCleanup(86400)
				if n > 0 {
					slog.Info("dedup cache cleanup", "removed", n)
				}
			}
		}
	}()

	done := make(chan struct{})
	go func() {
		<-ctx.Done()
		close(done)
	}()
	go api.RunPSCache(done)

	// Start HTTP server
	addr := fmt.Sprintf("%s:%d", cfg.Host, cfg.Port)
	srv := &http.Server{
		Addr:    addr,
		Handler: api.Handler(),
	}

	eventLog.Log("server.start", map[string]any{
		"vram_budget_gb":       cfg.VRAMBudgetGB,
		"recovered_jobs":       recovered,
		"reconciled_followers": reconciledFollowers,
	})
	slog.Info("arbiter started", "addr", addr, "vram_budget_gb", cfg.VRAMBudgetGB)

	// Handle shutdown signals
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		slog.Info("shutting down...")
		sched.MarkShuttingDown()
		cancel()
		srv.Shutdown(context.Background())
	}()

	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		slog.Error("server error", "error", err)
	}

	// Cleanup
	mgr.KillAll()
	// Catch any grandchildren that got reparented to init when their direct
	// worker parent died — same pattern as startup reaper. Without this pass,
	// a shutdown leaves llama-server / vllm processes holding VRAM forever.
	reapOrphanWorkers()
	eventLog.Log("server.stop", map[string]any{
		"uptime_seconds": time.Since(api.startTime).Seconds(),
	})
	slog.Info("arbiter stopped")
}

// setupInstances creates Instance objects for all configured models.
func setupInstances(cfg *Config, mgr *InstanceManager, pythonBin, projectRoot string) {
	for modelID, modelCfg := range cfg.Models {
		n := *modelCfg.MaxInstances
		mgr.EnsureModel(modelID)
		for i := 0; i < n; i++ {
			instanceID := modelID
			if n > 1 {
				instanceID = fmt.Sprintf("%s#%d", modelID, i)
			}
			inst := NewInstance(
				modelID, instanceID,
				modelCfg.MaxConcurrent,
				modelCfg.MemoryGB,
				pythonBin, projectRoot,
			)
			if len(modelCfg.WorkerCmd) > 0 {
				inst.workerCmd = modelCfg.WorkerCmd
			}
			if modelCfg.AdapterParams != nil {
				for k, v := range modelCfg.AdapterParams {
					inst.workerEnv = append(inst.workerEnv, k+"="+v)
				}
			}
			mgr.Register(inst)
		}
		if n > 1 {
			slog.Info("registered multi-instance model",
				"model", modelID,
				"instances", n,
				"memory_each_gb", modelCfg.MemoryGB,
			)
		}
	}
}
