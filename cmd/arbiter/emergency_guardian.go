package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"
)

// EmergencyGuardian is the arbiter's last line of defence against taking the
// host down. It watches the kernel's MemAvailable — the ONE number that
// unified-memory (UVM/CUDA) allocations cannot hide from, unlike cgroup
// accounting and per-process RSS — and when it crosses the floor it
// force-kills the worst-offending adapter instance, ACTIVE JOBS INCLUDED.
//
// This is deliberately different from every other watchdog in this file set:
//
//   - VRAMWatchdog works from the arbiter's own bookkeeping and only evicts
//     idle instances (Unload refuses busy ones).
//   - MemoryWatchdog observes and patches accounting; it never kills.
//   - The per-worker CUDA cap (worker_main._apply_cuda_memory_cap) bounds GPU
//     allocations made through torch, but CPU-side allocations (safetensors
//     host buffers during load, numpy, ffmpeg children) are uncapped and come
//     out of the same 128GB unified pool.
//
// So a busy or loading worker whose real footprint blows past its declaration
// previously had NOTHING arbiter-side standing between it and the documented
// GB10 failure mode: pool exhausted -> kernel OOM killer can't attribute the
// memory -> whole-machine livelock needing a physical power cycle (NVIDIA
// forum threads 353752 / 358951). Killing the adapter is always the right
// trade: its in-flight jobs fail cleanly (pending requests get "subprocess
// died" via markSubprocessExited, callers retry per their own policy) and the
// machine lives.
//
// Victim selection: the instance most OVER its declared memory_gb
// (tree RSS + tree VRAM - declared), i.e. the one lying to the scheduler. If
// every instance is within its declaration the pressure is partly external,
// but we still shed the largest instance — the arbiter is the only memory
// consumer on the box that can shed tens of GB in under a second, and a
// machine with no free memory is everyone's problem. earlyoom (SIGTERM at 5%
// MemAvailable) and the kernel panic sysctls sit below this floor as host
// backstops; the guardian fires first so the choice of casualty is informed.
type EmergencyGuardian struct {
	mgr     *InstanceManager
	logger  *EventLogger
	floorGB float64

	// injectable for tests
	readAvailableGB func() (float64, error)
	killInstance    func(victim instanceMemSnapshot) bool

	lastKill time.Time
}

const (
	emergencyDefaultFloorGB = 8.0
	emergencyTickInterval   = 2 * time.Second
	// After a kill, give the kernel time to actually reclaim the tree's pages
	// before deciding the kill "didn't help" and shooting the next instance.
	emergencyKillCooldown = 10 * time.Second
	// Killing an instance holding less than this frees nothing meaningful;
	// at that point the pressure is external and earlyoom owns the problem.
	emergencyMinFootprintGB = 1.0
)

func NewEmergencyGuardian(cfg *Config, mgr *InstanceManager, logger *EventLogger) *EmergencyGuardian {
	floor := cfg.EmergencyFloorGB
	if floor <= 0 {
		floor = emergencyDefaultFloorGB
	}
	g := &EmergencyGuardian{
		mgr:             mgr,
		logger:          logger,
		floorGB:         floor,
		readAvailableGB: readMemAvailableGB,
	}
	g.killInstance = g.forceKill
	return g
}

// Run ticks until ctx is done. interval<=0 uses the default 2s.
func (g *EmergencyGuardian) Run(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		interval = emergencyTickInterval
	}
	slog.Info("emergency guardian up", "floor_gb", g.floorGB, "interval", interval)
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			g.tick()
		}
	}
}

func (g *EmergencyGuardian) tick() {
	avail, err := g.readAvailableGB()
	if err != nil {
		return // not on linux / proc unreadable — nothing we can do
	}
	if avail >= g.floorGB {
		return
	}
	slog.Error("emergency guardian: MemAvailable below floor",
		"available_gb", avail, "floor_gb", g.floorGB)
	if time.Since(g.lastKill) < emergencyKillCooldown {
		return // let the previous kill's reclaim land before shooting again
	}

	snaps := g.mgr.snapshotKillableInstances(GetPerProcessVRAM())
	victim, ok := pickEmergencyVictim(snaps)
	if !ok {
		// Nothing of ours worth killing — pressure is external. earlyoom and
		// the host sysctls are the remaining layers; make sure it's on record.
		g.logger.Log("system.emergency_pressure_external", map[string]any{
			"available_gb": avail,
			"floor_gb":     g.floorGB,
			"instances":    len(snaps),
		})
		g.lastKill = time.Now() // throttle the event, not just kills
		return
	}

	if g.killInstance(victim) {
		g.lastKill = time.Now()
		g.logger.Log("system.emergency_kill", map[string]any{
			"instance_id":   victim.InstanceID,
			"model_id":      victim.ModelID,
			"tree_rss_gb":   victim.TreeRSSGB,
			"tree_vram_gb":  victim.TreeVRAMGB,
			"configured_gb": victim.ConfiguredGB,
			"available_gb":  avail,
			"floor_gb":      g.floorGB,
		})
	}
}

// forceKill obliterates the victim's process tree regardless of active jobs
// and releases its reservation. Pending requests resolve as errors via
// markSubprocessExited, so jobs fail cleanly rather than hanging.
func (g *EmergencyGuardian) forceKill(victim instanceMemSnapshot) bool {
	inst := g.mgr.Get(victim.InstanceID)
	if inst == nil {
		return false
	}
	slog.Error("emergency guardian: force-killing instance to protect the host",
		"instance", victim.InstanceID,
		"model", victim.ModelID,
		"tree_rss_gb", victim.TreeRSSGB,
		"tree_vram_gb", victim.TreeVRAMGB,
		"configured_gb", victim.ConfiguredGB,
		"active_jobs", inst.ActiveJobs())
	inst.Kill()
	g.mgr.ReleaseMemoryFor(inst)
	return true
}

// pickEmergencyVictim chooses the instance most over its declared memory_gb
// (tree RSS+VRAM vs configured). If nobody is over-declaration, it falls back
// to the largest absolute footprint. Returns ok=false when there is no
// instance whose death would free a meaningful amount of memory.
func pickEmergencyVictim(snaps []instanceMemSnapshot) (instanceMemSnapshot, bool) {
	var best instanceMemSnapshot
	bestOverage := 0.0
	bestTotal := 0.0
	found := false
	for _, s := range snaps {
		total := s.TreeRSSGB + s.TreeVRAMGB
		if total < emergencyMinFootprintGB {
			continue
		}
		overage := total - s.ConfiguredGB
		better := false
		switch {
		case overage > 0 && bestOverage > 0:
			better = overage > bestOverage
		case overage > 0 && bestOverage <= 0:
			better = true // an over-budget instance always outranks within-budget ones
		case overage <= 0 && bestOverage > 0:
			better = false
		default:
			better = total > bestTotal
		}
		if !found || better {
			best = s
			bestOverage = overage
			bestTotal = total
			found = true
		}
	}
	return best, found
}

// snapshotKillableInstances is snapshotInstanceMemory widened to include
// instances mid-load. A runaway LOAD (weights bigger than declared, host-side
// safetensors buffers) is precisely the historical machine-killer — the
// per-worker CUDA cap bounds its GPU side but not its host RSS — so the
// guardian must be able to see and shoot loading instances too.
func (m *InstanceManager) snapshotKillableInstances(pidVRAM map[int]int64) []instanceMemSnapshot {
	type pending struct {
		id      string
		modelID string
		pid     int
	}
	m.mu.RLock()
	var pendings []pending
	for _, inst := range m.instances {
		state := inst.State()
		if state != "loaded" && state != "active" && state != "loading" {
			continue
		}
		inst.mu.Lock()
		if inst.cmd != nil && inst.cmd.Process != nil {
			pendings = append(pendings, pending{
				id:      inst.InstanceID,
				modelID: inst.ModelID,
				pid:     inst.cmd.Process.Pid,
			})
		}
		inst.mu.Unlock()
	}
	configuredByModel := make(map[string]float64)
	for id, mc := range m.config.Models {
		configuredByModel[id] = mc.MemoryGB
	}
	m.mu.RUnlock()

	out := make([]instanceMemSnapshot, 0, len(pendings))
	for _, p := range pendings {
		out = append(out, instanceMemSnapshot{
			InstanceID:   p.id,
			ModelID:      p.modelID,
			PID:          p.pid,
			TreeVRAMGB:   float64(treeVRAMBytes(p.pid, pidVRAM)) / (1024 * 1024 * 1024),
			TreeRSSGB:    treeRSSAnonMB(p.pid) / 1024,
			ConfiguredGB: configuredByModel[p.modelID],
		})
	}
	return out
}

// readMemAvailableGB reads the kernel's estimate of allocatable memory. On
// the GB10 this is the ground truth that CUDA unified-memory allocations
// deplete even though they are invisible to cgroups and worker RSS.
func readMemAvailableGB() (float64, error) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, err
	}
	defer f.Close()
	return parseMemAvailableGB(f)
}

func parseMemAvailableGB(r io.Reader) (float64, error) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0, fmt.Errorf("malformed MemAvailable line: %q", line)
		}
		kb, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			return 0, fmt.Errorf("parse MemAvailable: %w", err)
		}
		return kb / (1024 * 1024), nil
	}
	if err := scanner.Err(); err != nil {
		return 0, err
	}
	return 0, fmt.Errorf("MemAvailable not found in meminfo")
}
