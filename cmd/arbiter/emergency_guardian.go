package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// EmergencyGuardian is the arbiter's last line of defence against taking the
// host down. It watches two kernel signals — MemAvailable (which
// unified-memory/CUDA allocations cannot hide from, unlike cgroup accounting
// and per-process RSS) and MemFree (because MemAvailable itself is blind to
// the GB10 pathology where page cache the NVIDIA driver cannot use inflates
// it; see emergencyDefaultMemFreeFloorGB) — and when either crosses its floor
// it first drops page cache if there is real cache to reclaim, then
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
	mgr            *InstanceManager
	logger         *EventLogger
	floorGB        float64
	memFreeFloorGB float64

	// injectable for tests
	readMeminfo  func() (meminfoGB, error)
	killInstance func(victim instanceMemSnapshot) bool
	dropCaches   func() error

	lastKill time.Time
	lastDrop time.Time
	// lastDropRecovered records whether the most recent cache drop lifted
	// MemFree back over its floor. A floor re-trip while this is true (and
	// cache is still large) is proven-benign refill — e.g. a loader streaming
	// a 145 GB BF16 repo — so the guardian re-drops instead of killing.
	lastDropRecovered bool
}

// meminfoGB is the slice of /proc/meminfo the guardian needs, in GB.
type meminfoGB struct {
	AvailableGB float64
	FreeGB      float64
	CachedGB    float64
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
	// MemFree co-trigger. MemAvailable is blind to the GB10 pathology where
	// tens of GB of page cache inflate it while the NVIDIA driver cannot use
	// that cache and reclaim stalls wedge the box (2026-06-10 03:58 host
	// death: MemFree pinned at 8-12GB with 25-65GB Cached, MemAvailable
	// healthy, guardian and earlyoom both silent). When MemFree itself is
	// critically low the box is one allocation burst from livelock no matter
	// what MemAvailable claims.
	emergencyDefaultMemFreeFloorGB = 4.0
	// Dropping page cache is the BENIGN remedy for a MemFree deficit, so the
	// decision to try it must be relative to the deficit — not an absolute
	// "lots of cache" threshold. This was an absolute 16GB slack until
	// 2026-08-24, which created a dead zone that killed jobs for nothing: a
	// render worker was force-killed 9 times with MemFree ~3.3GB against a 4GB
	// floor (a 0.7GB deficit) while 13-15GB of reclaimable cache sat there,
	// because 13-15GB < 16GB meant the drop was never even ATTEMPTED. Every
	// one of those kills destroyed ~28min of GPU work, and MemAvailable
	// (11.5-13.9GB) was never below its own 8GB floor — the host was in no
	// danger at all. Dropping cache in that state recovers ~10GB (measured
	// 8->25GB), so the only correct question is "would dropping plausibly
	// cover the shortfall", i.e. cache >= deficit * margin.
	emergencyDropCacheMargin = 2.0
	// Never shell out to sudo drop_caches for less reclaimable cache than
	// this; below it a drop cannot move MemFree meaningfully and the pressure
	// is real rather than cache-inflated.
	emergencyMinDropCacheGB = 2.0
	emergencyDropCooldown   = 60 * time.Second
	// Minimum gap between cache drops when the previous drop recovered. The
	// drop call is synchronous inside tick(), so this only bounds how often
	// sudo drop_caches runs — it must stay well under emergencyDropCooldown so
	// refill racing a recovered drop never falls through to the kill path.
	emergencyReDropCooldown = 5 * time.Second
)

func NewEmergencyGuardian(cfg *Config, mgr *InstanceManager, logger *EventLogger) *EmergencyGuardian {
	floor := cfg.EmergencyFloorGB
	if floor <= 0 {
		floor = emergencyDefaultFloorGB
	}
	memFreeFloor := cfg.EmergencyMemFreeFloorGB
	if memFreeFloor == 0 {
		memFreeFloor = emergencyDefaultMemFreeFloorGB
	} // negative disables the MemFree co-trigger
	g := &EmergencyGuardian{
		mgr:            mgr,
		logger:         logger,
		floorGB:        floor,
		memFreeFloorGB: memFreeFloor,
		readMeminfo:    readMeminfoGB,
		dropCaches:     sudoDropCaches,
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

// shouldDropCache reports whether dropping page cache is a plausible remedy for
// the current MemFree deficit. The comparison is deliberately RELATIVE: a 0.7GB
// shortfall against 15GB of reclaimable cache is the easiest possible recovery,
// and refusing to try it because 15GB fell under some absolute "lots of cache"
// bar is how the guardian ended up force-killing long-running jobs while the
// host was in no danger (see emergencyDropCacheMargin).
func shouldDropCache(mi meminfoGB, memFreeFloorGB float64) bool {
	deficit := memFreeFloorGB - mi.FreeGB
	if deficit <= 0 {
		return false // not actually short on MemFree
	}
	if mi.CachedGB < emergencyMinDropCacheGB {
		return false // nothing worth reclaiming; the pressure is real
	}
	return mi.CachedGB >= deficit*emergencyDropCacheMargin
}

func (g *EmergencyGuardian) tick() {
	mi, err := g.readMeminfo()
	if err != nil {
		return // not on linux / proc unreadable — nothing we can do
	}
	trigger := ""
	if mi.AvailableGB < g.floorGB {
		trigger = "mem_available"
	}
	if trigger == "" && g.memFreeFloorGB > 0 && mi.FreeGB < g.memFreeFloorGB {
		// MemAvailable looks fine but MemFree is critical — the cache-inflated
		// blind spot. If there is real cache to reclaim, drop it first: that is
		// the benign remedy and also covers a dead gpu-mem-governor.
		if shouldDropCache(mi, g.memFreeFloorGB) {
			since := time.Since(g.lastDrop)
			if since > emergencyDropCooldown ||
				(g.lastDropRecovered && since > emergencyReDropCooldown) {
				g.lastDrop = time.Now()
				dropErr := g.dropCaches()
				recovered := false
				if dropErr == nil {
					if after, err2 := g.readMeminfo(); err2 == nil && after.FreeGB >= g.memFreeFloorGB {
						recovered = true
					}
				}
				g.lastDropRecovered = recovered
				g.logger.Log("system.emergency_cache_drop", map[string]any{
					"mem_free_gb": mi.FreeGB,
					"cached_gb":   mi.CachedGB,
					"floor_gb":    g.memFreeFloorGB,
					"drop_failed": dropErr != nil,
					"recovered":   recovered,
				})
				if recovered {
					return
				}
			} else if g.lastDropRecovered {
				// Too soon to re-drop, but the previous drop proved this cache
				// is reclaimable: the re-trip is refill racing our own recovery
				// window, not the 2026-06-10 hazard. Hold fire this tick; the
				// next tick re-drops. (Historically this fell through to kill,
				// which shot H3 loaders 3–9 s after successful drops.)
				return
			}
		}
		trigger = "mem_free"
	}
	if trigger == "" {
		return
	}
	slog.Error("emergency guardian: memory below floor",
		"trigger", trigger, "available_gb", mi.AvailableGB, "mem_free_gb", mi.FreeGB,
		"cached_gb", mi.CachedGB, "floor_gb", g.floorGB, "memfree_floor_gb", g.memFreeFloorGB)
	if time.Since(g.lastKill) < emergencyKillCooldown {
		return // let the previous kill's reclaim land before shooting again
	}

	snaps := g.mgr.snapshotKillableInstances(GetPerProcessVRAM())
	victim, ok := pickEmergencyVictim(snaps)
	if !ok {
		// Nothing of ours worth killing — pressure is external. earlyoom and
		// the host sysctls are the remaining layers; make sure it's on record.
		g.logger.Log("system.emergency_pressure_external", map[string]any{
			"available_gb": mi.AvailableGB,
			"mem_free_gb":  mi.FreeGB,
			"trigger":      trigger,
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
			"available_gb":  mi.AvailableGB,
			"mem_free_gb":   mi.FreeGB,
			"trigger":       trigger,
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
	m.config.RangeModels(func(id string, mc ModelConfig) bool {
		configuredByModel[id] = mc.MemoryGB
		return true
	})
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

// readMeminfoGB reads the guardian's slice of /proc/meminfo. MemAvailable is
// the kernel's estimate of allocatable memory — the ground truth that CUDA
// unified-memory allocations deplete even though they are invisible to
// cgroups and worker RSS. MemFree and Cached are read alongside it because
// MemAvailable counts reclaimable cache the NVIDIA driver cannot actually
// use, so MemFree is the number that exposes the GB10 starvation pathology.
func readMeminfoGB() (meminfoGB, error) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return meminfoGB{}, err
	}
	defer func() {
		if err := f.Close(); err != nil {
			slog.Debug("close meminfo", "error", err)
		}
	}()
	return parseMeminfoGB(f)
}

func parseMeminfoGB(r io.Reader) (meminfoGB, error) {
	var mi meminfoGB
	seenAvailable := false
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 {
			continue
		}
		kb, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			continue
		}
		gb := kb / (1024 * 1024)
		switch fields[0] {
		case "MemAvailable:":
			mi.AvailableGB = gb
			seenAvailable = true
		case "MemFree:":
			mi.FreeGB = gb
		case "Cached:":
			mi.CachedGB = gb
		}
	}
	if err := scanner.Err(); err != nil {
		return meminfoGB{}, err
	}
	if !seenAvailable {
		return meminfoGB{}, fmt.Errorf("MemAvailable not found in meminfo")
	}
	return mi, nil
}

// sudoDropCaches drops clean page cache (not dentries/inodes), same remedy as
// gpu-mem-governor.sh. Requires passwordless sudo; -n fails instead of
// prompting if that ever changes.
func sudoDropCaches() error {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	return exec.CommandContext(ctx, "sudo", "-n", "sh", "-c",
		"sync; echo 1 > /proc/sys/vm/drop_caches").Run()
}
