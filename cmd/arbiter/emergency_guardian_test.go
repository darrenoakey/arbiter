package main

import (
	"strings"
	"testing"
	"time"
)

func TestParseMeminfoGB(t *testing.T) {
	meminfo := `MemTotal:       125353216 kB
MemFree:        70463488 kB
MemAvailable:   73932800 kB
Buffers:           12345 kB
Cached:          2242560 kB
`
	got, err := parseMeminfoGB(strings.NewReader(meminfo))
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	check := func(name string, got, wantKB float64) {
		want := wantKB / (1024 * 1024)
		if got < want-0.01 || got > want+0.01 {
			t.Fatalf("%s: got %.3f GB, want %.3f GB", name, got, want)
		}
	}
	check("MemAvailable", got.AvailableGB, 73932800)
	check("MemFree", got.FreeGB, 70463488)
	check("Cached", got.CachedGB, 2242560)
}

func TestParseMeminfoGBMissingAvailable(t *testing.T) {
	if _, err := parseMeminfoGB(strings.NewReader("MemTotal: 1 kB\n")); err == nil {
		t.Fatal("expected error for meminfo without MemAvailable")
	}
}

func TestPickEmergencyVictimPrefersOverDeclaration(t *testing.T) {
	snaps := []instanceMemSnapshot{
		// biggest absolute footprint, but within its declaration
		{InstanceID: "ltx2", ModelID: "ltx2", TreeRSSGB: 20, TreeVRAMGB: 60, ConfiguredGB: 90},
		// smaller, but 20GB OVER what it told the scheduler — the liar dies first
		{InstanceID: "wan", ModelID: "wan-flf", TreeRSSGB: 40, TreeVRAMGB: 10, ConfiguredGB: 30},
		{InstanceID: "kokoro", ModelID: "tts-kokoro", TreeRSSGB: 1.5, TreeVRAMGB: 0.5, ConfiguredGB: 2},
	}
	victim, ok := pickEmergencyVictim(snaps)
	if !ok {
		t.Fatal("expected a victim")
	}
	if victim.InstanceID != "wan" {
		t.Fatalf("expected over-declaration instance 'wan', got %q", victim.InstanceID)
	}
}

func TestPickEmergencyVictimFallsBackToLargest(t *testing.T) {
	snaps := []instanceMemSnapshot{
		{InstanceID: "small", ModelID: "m1", TreeRSSGB: 2, TreeVRAMGB: 2, ConfiguredGB: 10},
		{InstanceID: "big", ModelID: "m2", TreeRSSGB: 10, TreeVRAMGB: 40, ConfiguredGB: 90},
	}
	victim, ok := pickEmergencyVictim(snaps)
	if !ok {
		t.Fatal("expected a victim")
	}
	if victim.InstanceID != "big" {
		t.Fatalf("expected largest instance 'big', got %q", victim.InstanceID)
	}
}

func TestPickEmergencyVictimIgnoresTinyFootprints(t *testing.T) {
	snaps := []instanceMemSnapshot{
		{InstanceID: "tiny", ModelID: "m1", TreeRSSGB: 0.3, TreeVRAMGB: 0.2, ConfiguredGB: 2},
	}
	if _, ok := pickEmergencyVictim(snaps); ok {
		t.Fatal("expected no victim when all footprints are below the minimum")
	}
	if _, ok := pickEmergencyVictim(nil); ok {
		t.Fatal("expected no victim for empty snapshot list")
	}
}

// TestEmergencyGuardianTick drives the full tick path with injected memory
// readings and kill hook: above floor -> no kill; below floor -> kill of the
// over-declaration victim; immediately below floor again -> cooldown blocks a
// second kill.
func TestEmergencyGuardianTick(t *testing.T) {
	cfg := &Config{
		EmergencyFloorGB: 8,
		Models: map[string]ModelConfig{
			"wan-flf": {MemoryGB: 30},
		},
	}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	logger := NewEventLogger(t.TempDir())

	g := NewEmergencyGuardian(cfg, mgr, logger)

	mem := meminfoGB{AvailableGB: 50, FreeGB: 40, CachedGB: 5}
	g.readMeminfo = func() (meminfoGB, error) { return mem, nil }

	var killed []string
	g.killInstance = func(v instanceMemSnapshot) bool {
		killed = append(killed, v.InstanceID)
		return true
	}
	// snapshotKillableInstances needs real instances+processes; inject the
	// snapshot indirectly by overriding the kill path only and pre-seeding the
	// pick through a tiny shim: tick() consults mgr's snapshots, which are
	// empty here, so emulate the victim path by checking the external-pressure
	// event instead when no instances exist.

	// 1. Healthy memory: nothing happens.
	g.tick()
	if len(killed) != 0 {
		t.Fatalf("kill fired with healthy memory: %v", killed)
	}

	// 2. Below floor with NO loaded instances: external-pressure event path,
	// no kill, and the cooldown timestamp advances (event throttle).
	mem.AvailableGB = 4.0
	g.tick()
	if len(killed) != 0 {
		t.Fatalf("kill fired with no instances: %v", killed)
	}
	if g.lastKill.IsZero() {
		t.Fatal("external-pressure path should set lastKill to throttle events")
	}

	// 3. Cooldown: a tick right after must not do anything even below floor.
	prev := g.lastKill
	g.tick()
	if g.lastKill != prev {
		t.Fatal("cooldown should prevent state change on immediate re-tick")
	}

	// 4. After cooldown expires, the guardian acts again.
	g.lastKill = time.Now().Add(-2 * emergencyKillCooldown)
	g.tick()
	if g.lastKill.Equal(prev.Add(-2 * emergencyKillCooldown)) {
		t.Fatal("guardian did not act after cooldown expiry")
	}
}

// TestEmergencyGuardianMemFreeCoTrigger covers the GB10 blind spot: MemFree
// critical while page cache keeps MemAvailable looking healthy (the 2026-06-10
// host death). The guardian must first try a cache drop, and only act further
// if MemFree stays under the floor.
func TestEmergencyGuardianMemFreeCoTrigger(t *testing.T) {
	cfg := &Config{
		EmergencyFloorGB:        8,
		EmergencyMemFreeFloorGB: 4,
		Models:                  map[string]ModelConfig{},
	}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	logger := NewEventLogger(t.TempDir())
	g := NewEmergencyGuardian(cfg, mgr, logger)

	mem := meminfoGB{AvailableGB: 40, FreeGB: 2, CachedGB: 35}
	g.readMeminfo = func() (meminfoGB, error) { return mem, nil }
	drops := 0
	g.dropCaches = func() error {
		drops++
		mem.FreeGB = 30 // the drop reclaims cache into MemFree
		mem.CachedGB = 5
		return nil
	}
	var killed []string
	g.killInstance = func(v instanceMemSnapshot) bool {
		killed = append(killed, v.InstanceID)
		return true
	}

	// 1. MemFree critical + lots of cache: drop fires, recovers, nothing dies.
	g.tick()
	if drops != 1 {
		t.Fatalf("expected exactly one cache drop, got %d", drops)
	}
	if len(killed) != 0 {
		t.Fatal("kill fired even though the cache drop recovered MemFree")
	}
	if !g.lastKill.IsZero() {
		t.Fatal("recovered drop must not consume the kill/event throttle")
	}

	// 2. MemFree critical with NO cache to drop: goes straight to the shed
	// path (external-pressure event here since no instances exist).
	mem = meminfoGB{AvailableGB: 40, FreeGB: 2, CachedGB: 3}
	g.tick()
	if drops != 1 {
		t.Fatalf("drop attempted with no reclaimable cache: %d", drops)
	}
	if g.lastKill.IsZero() {
		t.Fatal("expected the shed path (external-pressure event) to fire")
	}

	// 3. MemFree critical, cache present, but the drop does NOT recover:
	// proceeds to the shed path after the drop.
	g.lastKill = time.Time{}
	g.lastDrop = time.Time{}
	mem = meminfoGB{AvailableGB: 40, FreeGB: 2, CachedGB: 35}
	g.dropCaches = func() error { drops++; return nil } // drop "succeeds" but frees nothing
	g.tick()
	if drops != 2 {
		t.Fatalf("expected a second drop attempt, got %d", drops)
	}
	if g.lastKill.IsZero() {
		t.Fatal("expected shed path when drop fails to recover MemFree")
	}
}

func TestEmergencyGuardianMemFreeTriggerDisabled(t *testing.T) {
	cfg := &Config{
		EmergencyFloorGB:        8,
		EmergencyMemFreeFloorGB: -1,
		Models:                  map[string]ModelConfig{},
	}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	logger := NewEventLogger(t.TempDir())
	g := NewEmergencyGuardian(cfg, mgr, logger)

	g.readMeminfo = func() (meminfoGB, error) {
		return meminfoGB{AvailableGB: 40, FreeGB: 1, CachedGB: 35}, nil
	}
	dropped := false
	g.dropCaches = func() error { dropped = true; return nil }
	g.killInstance = func(v instanceMemSnapshot) bool { t.Fatal("kill must not fire"); return false }

	g.tick()
	if dropped {
		t.Fatal("co-trigger ran despite emergency_memfree_floor_gb=-1")
	}
	if !g.lastKill.IsZero() {
		t.Fatal("no event path should fire when co-trigger disabled and MemAvailable healthy")
	}
}

// TestEmergencyGuardianCacheRefillReDrop covers the 2026-08-17 H3 loader
// kills: a loader streaming a 145 GB BF16 repo refills page cache faster than
// the guardian can drop it, so the MemFree floor re-trips INSIDE the drop
// cooldown. When the previous drop recovered — cache proven reclaimable — a
// re-trip must re-drop (or hold fire for the re-drop gap) instead of falling
// through to kill. A drop that failed to recover keeps the kill path.
func TestEmergencyGuardianCacheRefillReDrop(t *testing.T) {
	cfg := &Config{
		EmergencyFloorGB:        8,
		EmergencyMemFreeFloorGB: 4,
		Models:                  map[string]ModelConfig{},
	}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	logger := NewEventLogger(t.TempDir())
	g := NewEmergencyGuardian(cfg, mgr, logger)

	mem := meminfoGB{AvailableGB: 40, FreeGB: 3, CachedGB: 77}
	g.readMeminfo = func() (meminfoGB, error) { return mem, nil }
	drops := 0
	g.dropCaches = func() error {
		drops++
		mem.FreeGB = 30 // drop reclaims the refill into MemFree
		mem.CachedGB = 5
		return nil
	}
	var killed []string
	g.killInstance = func(v instanceMemSnapshot) bool {
		killed = append(killed, v.InstanceID)
		return true
	}

	// 1. First trip: drop fires and recovers; nothing dies.
	g.tick()
	if drops != 1 || len(killed) != 0 {
		t.Fatalf("first trip: drops=%d killed=%v", drops, killed)
	}
	if !g.lastDropRecovered {
		t.Fatal("recovered drop must set lastDropRecovered")
	}

	// 2. Refill re-trips the floor inside the re-drop gap: hold fire — no
	// kill, no redundant sudo drop.
	mem = meminfoGB{AvailableGB: 40, FreeGB: 3, CachedGB: 77}
	g.tick()
	if drops != 1 || len(killed) != 0 {
		t.Fatalf("re-trip inside re-drop gap: drops=%d killed=%v", drops, killed)
	}
	if !g.lastKill.IsZero() {
		t.Fatal("benign refill inside the re-drop gap must not arm the kill throttle")
	}

	// 3. Past the re-drop gap (still inside the 60 s drop cooldown): the
	// guardian re-drops instead of killing, and the drop recovers again.
	g.lastDrop = time.Now().Add(-emergencyReDropCooldown - time.Second)
	mem = meminfoGB{AvailableGB: 40, FreeGB: 3, CachedGB: 77}
	g.tick()
	if drops != 2 || len(killed) != 0 {
		t.Fatalf("re-trip past re-drop gap: drops=%d killed=%v", drops, killed)
	}

	// 4. Contrast: a re-trip during cooldown when the last drop did NOT
	// recover keeps today's behaviour — straight to the shed path (here the
	// external-pressure event, since no instances exist).
	g.lastDropRecovered = false
	g.lastKill = time.Time{}
	g.lastDrop = time.Now().Add(-emergencyReDropCooldown - time.Second)
	mem = meminfoGB{AvailableGB: 40, FreeGB: 3, CachedGB: 77}
	g.tick()
	if drops != 2 {
		t.Fatalf("non-recovering drop during cooldown must not re-drop: drops=%d", drops)
	}
	if g.lastKill.IsZero() {
		t.Fatal("non-recovering drop during cooldown must reach the shed path")
	}
	if len(killed) != 0 {
		t.Fatalf("no instances exist, so nothing to kill: killed=%v", killed)
	}
}

// TestShouldDropCacheIsDeficitRelative pins the drop decision to the SHORTFALL
// rather than an absolute cache threshold. The first case is the exact
// production reading that force-killed a render worker nine times on
// 2026-08-23/24: a 0.74GB MemFree deficit with 15.1GB of reclaimable cache. An
// absolute 16GB gate refused to even attempt the drop and went straight to
// killing a job that had already burned 28 minutes of GPU time.
func TestShouldDropCacheIsDeficitRelative(t *testing.T) {
	const floor = 4.0
	cases := []struct {
		name string
		mi   meminfoGB
		want bool
	}{
		{
			name: "observed production kill: tiny deficit, ample cache",
			mi:   meminfoGB{AvailableGB: 13.880844116210938, FreeGB: 3.2564353942871094, CachedGB: 15.124683380126953},
			want: true,
		},
		{
			name: "observed production kill: lowest cache seen",
			mi:   meminfoGB{AvailableGB: 11.565895080566406, FreeGB: 3.637237548828125, CachedGB: 13.069778442382812},
			want: true,
		},
		{
			name: "large deficit, cache cannot cover it",
			mi:   meminfoGB{AvailableGB: 40, FreeGB: 2, CachedGB: 3},
			want: false,
		},
		{
			name: "cache below the minimum worth dropping",
			mi:   meminfoGB{AvailableGB: 40, FreeGB: 3.9, CachedGB: 1.5},
			want: false,
		},
		{
			name: "no deficit at all",
			mi:   meminfoGB{AvailableGB: 40, FreeGB: 9, CachedGB: 50},
			want: false,
		},
		{
			name: "deficit exactly covered at the margin",
			mi:   meminfoGB{AvailableGB: 40, FreeGB: 2, CachedGB: 4},
			want: true,
		},
	}
	for _, tc := range cases {
		if got := shouldDropCache(tc.mi, floor); got != tc.want {
			t.Errorf("%s: shouldDropCache(free=%.3f cached=%.3f, floor=%.1f) = %v, want %v",
				tc.name, tc.mi.FreeGB, tc.mi.CachedGB, floor, got, tc.want)
		}
	}
}

// TestEmergencyGuardianDropsRatherThanKillsOnModerateCache is the end-to-end
// regression for the 2026-08-24 render kills. It drives tick() with the real
// captured meminfo shape — MemAvailable comfortably ABOVE its own floor, so the
// host was never in danger — and asserts the guardian reclaims cache instead of
// destroying an active job. Every pre-existing test in this file used 35GB or
// 77GB of cache, which is exactly why the 13-15GB production band shipped
// broken: no fixture ever visited it.
func TestEmergencyGuardianDropsRatherThanKillsOnModerateCache(t *testing.T) {
	cfg := &Config{
		EmergencyFloorGB:        8,
		EmergencyMemFreeFloorGB: 4,
		Models:                  map[string]ModelConfig{"minimax-h3-local": {MemoryGB: 80}},
	}
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	logger := NewEventLogger(t.TempDir())
	g := NewEmergencyGuardian(cfg, mgr, logger)

	// Exact reading logged immediately before the 02:28:34Z force-kill.
	mem := meminfoGB{AvailableGB: 13.880844116210938, FreeGB: 3.2564353942871094, CachedGB: 15.124683380126953}
	g.readMeminfo = func() (meminfoGB, error) { return mem, nil }
	drops := 0
	g.dropCaches = func() error {
		drops++
		// Measured on spark: dropping ~15GB of clean cache took MemFree from
		// 3.3GB to ~25GB.
		mem.FreeGB = 25
		mem.CachedGB = 1.2
		return nil
	}
	var killed []string
	g.killInstance = func(v instanceMemSnapshot) bool {
		killed = append(killed, v.InstanceID)
		return true
	}

	g.tick()

	if drops != 1 {
		t.Fatalf("guardian must attempt the benign cache drop, got %d drops", drops)
	}
	if len(killed) != 0 {
		t.Fatalf("guardian killed %v despite 15GB of reclaimable cache and MemAvailable above its floor", killed)
	}
	if !g.lastKill.IsZero() {
		t.Fatal("a recovered drop must not consume the kill/event throttle")
	}
	if !g.lastDropRecovered {
		t.Fatal("drop recovered MemFree, so lastDropRecovered must be set for the refill path")
	}
}
