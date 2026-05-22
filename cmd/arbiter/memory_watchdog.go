package main

import (
	"context"
	"log/slog"
	"sync"
	"time"
)

// MemoryWatchdog samples actual per-instance memory usage (VRAM + system RAM
// across the worker process tree) and:
//
//  1. emits a `model.memory_drift` event when actual exceeds the configured
//     `memory_gb` by a wide margin for a sustained window (drift detection);
//  2. patches `local/config.json` `memory_gb` to a new high-water mark on the
//     same condition (dynamic config writeback) — capped per session;
//  3. emits `system.memory_pressure` when total tree-RSS approaches
//     `system_ram_budget_gb`, the unified-memory ceiling.
//
// The watchdog is read-mostly: it never evicts instances on its own (existing
// VRAM watchdog handles that). Its job is to make accounting truthful.
type MemoryWatchdog struct {
	cfg         *Config
	mgr         *InstanceManager
	logger      *EventLogger
	projectRoot string

	driftMu        sync.Mutex
	consecutive    map[string]int     // model_id -> consecutive intervals over threshold (upward)
	underUtil      map[string]int     // instance_id -> consecutive intervals well below configured (downward)
	writebackDone  map[string]bool    // model_id -> already patched this session
	pressureLastAt time.Time          // throttle system.memory_pressure events
	lastWriteback  map[string]float64 // model_id -> last patched memory_gb
}

const (
	driftRatio          = 1.5              // actual must exceed configured by this much to count
	driftConsecutive    = 2                // sustained over this many checks (so 60s at 30s cadence)
	driftWritebackBuf   = 1.2              // patched value = observed_max * 1.2
	driftWritebackCap   = 0.8              // never write a value larger than VRAMBudgetGB * cap
	pressureThreshold   = 0.95             // emit pressure event when total RAM > budget * threshold
	pressureMinInterval = 30 * time.Second // throttle pressure events
	// Downward shrink — reclaim over-reservation from instances that have
	// stabilised well below their declared memory_gb (e.g. vLLM with
	// gpu-memory-utilization=0.30 declared at 80GB but using 25GB).
	shrinkRatio       = 0.6 // shrink only if observed < configured * this (i.e. >40% headroom wasted)
	shrinkConsecutive = 4   // require 4 stable samples (~2 min at 30s cadence) before shrinking
	shrinkSafetyBuf   = 1.3 // new memoryGB = observed * this (30% safety margin above peak observed)
	shrinkMinFreedGB  = 2.0 // don't bother for tiny gains
)

// NewMemoryWatchdog constructs a watchdog. Caller starts it with Run.
func NewMemoryWatchdog(cfg *Config, mgr *InstanceManager, logger *EventLogger, projectRoot string) *MemoryWatchdog {
	return &MemoryWatchdog{
		cfg:           cfg,
		mgr:           mgr,
		logger:        logger,
		projectRoot:   projectRoot,
		consecutive:   make(map[string]int),
		underUtil:     make(map[string]int),
		writebackDone: make(map[string]bool),
		lastWriteback: make(map[string]float64),
	}
}

// Run polls every interval until ctx is done. interval=0 uses 30s.
func (w *MemoryWatchdog) Run(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		interval = 30 * time.Second
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			w.tick()
		}
	}
}

// tick performs a single sampling pass.
//
// Note: there is no downward shrink path. The declared memory_gb is the
// worst-case ceiling — what the worker COULD grow to under load (e.g. vLLM
// with gpu-memory-utilization=0.66 will eventually pre-allocate ~80GB of
// KV pool even if it currently uses 17GB). Shrinking the reservation based
// on observed-low usage would over-promise free VRAM to other models, then
// when the original worker grows into its declared budget the GPU OOMs and
// everything dies. Keep the conservative reservation; the upward drift
// watchdog catches under-declaration but the inverse is safer left alone.
func (w *MemoryWatchdog) tick() {
	pidVRAM := GetPerProcessVRAM()
	snaps := w.mgr.snapshotInstanceMemory(pidVRAM)
	var totalRAM float64
	for _, s := range snaps {
		totalRAM += s.TreeRSSGB
		w.checkDrift(s)
	}
	w.checkSystemPressure(totalRAM)
}

func (w *MemoryWatchdog) checkDrift(s instanceMemSnapshot) {
	configured := s.ConfiguredGB
	if configured <= 0 {
		return // no configured value to drift from
	}
	// memory_gb is the GPU-VRAM reservation, so drift is measured against
	// actual GPU VRAM only. Host RSS is governed separately by
	// checkSystemPressure. On unified-memory hardware (GB10) a vLLM worker's
	// large host-side RSS used to leak into this figure via a max(), inflating
	// the VRAM reservation far above the model's true GPU ceiling — e.g. gemma,
	// hard-capped by --gpu-memory-utilization at ~43GB, got recorded at ~75GB
	// (a ~62GB RSS+VRAM peak × 1.2), which then made it impossible to
	// co-schedule with any other large model.
	observed := s.TreeVRAMGB
	w.driftMu.Lock()
	defer w.driftMu.Unlock()

	if observed <= configured*driftRatio {
		w.consecutive[s.ModelID] = 0
		return
	}
	w.consecutive[s.ModelID]++
	if w.consecutive[s.ModelID] < driftConsecutive {
		return
	}

	// Sustained drift. Always emit the event.
	target := observed * driftWritebackBuf
	if cap := w.cfg.VRAMBudgetGB * driftWritebackCap; cap > 0 && target > cap {
		target = cap
	}
	patched := false
	if !w.writebackDone[s.ModelID] && target > configured {
		if err := PatchModelMemoryGB(w.projectRoot, s.ModelID, target); err != nil {
			slog.Warn("memory_watchdog: writeback failed", "model", s.ModelID, "err", err)
		} else {
			w.writebackDone[s.ModelID] = true
			w.lastWriteback[s.ModelID] = target
			w.mgr.UpdateModelMemoryGB(s.ModelID, target)
			patched = true
		}
	}
	w.logger.Log("model.memory_drift", map[string]any{
		"model_id":            s.ModelID,
		"configured_gb":       configured,
		"observed_max_gb":     observed,
		"tree_vram_gb":        s.TreeVRAMGB,
		"tree_rss_gb":         s.TreeRSSGB,
		"ratio":               observed / configured,
		"writeback_attempted": !w.writebackDone[s.ModelID] || patched,
		"writeback_applied":   patched,
		"new_memory_gb":       target,
	})
	w.consecutive[s.ModelID] = 0 // reset so we don't spam after writeback
}

func (w *MemoryWatchdog) checkSystemPressure(totalRAMGB float64) {
	budget := w.cfg.SystemRAMBudgetGB
	if budget <= 0 {
		return // disabled
	}
	if totalRAMGB <= budget*pressureThreshold {
		return
	}
	w.driftMu.Lock()
	defer w.driftMu.Unlock()
	if time.Since(w.pressureLastAt) < pressureMinInterval {
		return
	}
	w.pressureLastAt = time.Now()
	w.logger.Log("system.memory_pressure", map[string]any{
		"total_tree_rss_gb": totalRAMGB,
		"budget_gb":         budget,
		"threshold_gb":      budget * pressureThreshold,
		"over_budget":       totalRAMGB > budget,
	})
}
