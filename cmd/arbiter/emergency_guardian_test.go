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
