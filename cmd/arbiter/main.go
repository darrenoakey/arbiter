package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"path/filepath"
	"syscall"
	"time"
)

func main() {
	// Structured logging to stderr
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo})))

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

	// Output dirs
	outputDir := filepath.Join(projectRoot, "output")
	os.MkdirAll(filepath.Join(outputDir, "jobs"), 0o755)
	os.MkdirAll(filepath.Join(outputDir, "logs"), 0o755)
	os.MkdirAll(filepath.Join(outputDir, "refs"), 0o755)

	// Event logger
	eventLog := NewEventLogger(filepath.Join(outputDir, "logs"))
	defer eventLog.Close()

	// Store
	dbPath := filepath.Join(outputDir, "arbiter.db")
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
	mgr := NewInstanceManager(cfg.VRAMBudgetGB, pythonBin, projectRoot)
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

	// Scheduler
	sched := NewScheduler(cfg, store, mgr, eventLog, outputDir)

	// API
	api := NewAPI(cfg, store, mgr, sched, eventLog, outputDir, projectRoot)

	// Context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start background goroutines
	go sched.Run(ctx)
	go sched.RunKeepalive(ctx)

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
		"vram_budget_gb": cfg.VRAMBudgetGB,
		"recovered_jobs": recovered,
	})
	slog.Info("arbiter started", "addr", addr, "vram_budget_gb", cfg.VRAMBudgetGB)

	// Handle shutdown signals
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		slog.Info("shutting down...")
		cancel()
		srv.Shutdown(context.Background())
	}()

	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		slog.Error("server error", "error", err)
	}

	// Cleanup
	mgr.KillAll()
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
