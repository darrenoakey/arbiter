package main

import "testing"

func modelInstanceIDs(mgr *InstanceManager, modelID string) []string {
	instances := mgr.GetModelInstances(modelID)
	out := make([]string, 0, len(instances))
	for _, inst := range instances {
		out = append(out, inst.InstanceID)
	}
	return out
}

func containsString(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}

func TestReloadModelReplacesDispatchInstances(t *testing.T) {
	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 70}, "python3", t.TempDir())
	cfg := ModelConfig{
		MemoryGB:      4,
		MaxConcurrent: 1,
		MaxInstances:  intPtr(2),
	}

	initial := mgr.ScaleModel("demo", 2, cfg)
	if initial["added"].(int) != 2 {
		t.Fatalf("initial scale added = %v, want 2", initial["added"])
	}
	before := modelInstanceIDs(mgr, "demo")

	updated := cfg
	updated.WorkerCmd = []string{"custom-worker"}
	reloaded := mgr.ReloadModel("demo", 2, updated)
	after := modelInstanceIDs(mgr, "demo")

	if reloaded["added"].(int) != 2 {
		t.Fatalf("reload added = %v, want 2", reloaded["added"])
	}
	if reloaded["removed"].(int) != 2 {
		t.Fatalf("reload removed = %v, want 2", reloaded["removed"])
	}
	if len(after) != 2 {
		t.Fatalf("dispatch instances after reload = %d, want 2", len(after))
	}
	for _, id := range before {
		if containsString(after, id) {
			t.Fatalf("old instance %s still in dispatch set after reload", id)
		}
	}
	for _, inst := range mgr.GetModelInstances("demo") {
		if len(inst.workerCmd) != 1 || inst.workerCmd[0] != "custom-worker" {
			t.Fatalf("replacement instance has wrong worker cmd: %+v", inst.workerCmd)
		}
	}
}

// remotePlacementTestConfig builds a config with one remote-only model
// ("llm:remote-chat" on h1+h2) and one mixed local+remote model
// ("mixed-chat" on spark+h1), matching the fleet-placement shape that
// wedged on 2026-07-11: DELETE /v1/models/<id>/workers destroyed the
// "@host" instances and recreated only local "#N" slots, so every queued
// job requeued forever ("PickInstanceForJob returned nil").
func remotePlacementTestConfig() *Config {
	return &Config{
		VRAMBudgetGB: 100,
		Hosts: map[string]HostConfig{
			"h1": {Addr: "http://10.255.255.1:11434", Kind: "mlx", BudgetGB: 64},
			"h2": {Addr: "http://10.255.255.2:11434", Kind: "mlx", BudgetGB: 64},
		},
		Models: map[string]ModelConfig{
			"llm:remote-chat": {
				MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
				Placements:    []string{"h1", "h2"},
				AdapterParams: map[string]string{"remote_model_tag": "chat:latest"},
			},
			"mixed-chat": {
				MemoryGB: 1, MaxConcurrent: 1, MaxInstances: intPtr(1),
				PressureIndex: pi(),
				Placements:    []string{"spark", "h1"},
			},
		},
	}
}

func TestHardKillModelRecreatesRemotePlacements(t *testing.T) {
	cfg := remotePlacementTestConfig()
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())

	mcfg := cfg.Models["llm:remote-chat"]
	result := mgr.HardKillModel("llm:remote-chat", true, &mcfg)
	if result["killed"].(int) != 2 {
		t.Fatalf("killed = %v, want 2 (both remote placements)", result["killed"])
	}

	after := modelInstanceIDs(mgr, "llm:remote-chat")
	if !containsString(after, "llm:remote-chat@h1") || !containsString(after, "llm:remote-chat@h2") {
		t.Fatalf("remote placements not recreated after hard kill: %v", after)
	}
	for _, inst := range mgr.GetModelInstances("llm:remote-chat") {
		if !inst.isRemote() {
			t.Fatalf("hard kill recreated a LOCAL instance %s for a remote-only model (no adapter exists — it would crash-loop)", inst.InstanceID)
		}
	}

	// The real invariant: the model must still be schedulable.
	job := &Job{ID: "j1", ModelID: "llm:remote-chat", State: "queued"}
	if got := mgr.PickInstanceForJob(job, true); got == nil {
		t.Fatal("PickInstanceForJob returned nil after hard kill — model is wedged until restart")
	}
}

func TestHardKillModelRecreatesMixedPlacements(t *testing.T) {
	cfg := remotePlacementTestConfig()
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())

	mcfg := cfg.Models["mixed-chat"]
	result := mgr.HardKillModel("mixed-chat", true, &mcfg)
	if result["killed"].(int) != 2 {
		t.Fatalf("killed = %v, want 2 (local + remote)", result["killed"])
	}
	if result["recreated"].(int) != 2 {
		t.Fatalf("recreated = %v, want 2 (local + remote)", result["recreated"])
	}

	var haveLocal, haveRemote bool
	for _, inst := range mgr.GetModelInstances("mixed-chat") {
		if inst.isRemote() {
			haveRemote = true
		} else {
			haveLocal = true
		}
	}
	if !haveLocal || !haveRemote {
		t.Fatalf("mixed model missing a placement after hard kill: local=%v remote=%v ids=%v",
			haveLocal, haveRemote, modelInstanceIDs(mgr, "mixed-chat"))
	}
}

func TestScaleModelLeavesRemotePlacementsAlone(t *testing.T) {
	cfg := remotePlacementTestConfig()
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())

	// mixed-chat has 1 local + 1 remote instance. Scaling the local pool to 1
	// must be a no-op — the remote instance must not be counted or retired.
	mcfg := cfg.Models["mixed-chat"]
	result := mgr.ScaleModel("mixed-chat", 1, mcfg)
	if result["added"].(int) != 0 || result["removed"].(int) != 0 || result["condemned"].(int) != 0 {
		t.Fatalf("scale to current local count mutated instances: %v", result)
	}
	if !containsString(modelInstanceIDs(mgr, "mixed-chat"), "mixed-chat@h1") {
		t.Fatal("remote placement lost after local pool scale")
	}
}

func TestReloadModelHealsMissingRemotePlacements(t *testing.T) {
	cfg := remotePlacementTestConfig()
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())

	// Simulate the wedged state: all instances destroyed, none recreated.
	mcfg := cfg.Models["llm:remote-chat"]
	mgr.HardKillModel("llm:remote-chat", false, &mcfg)
	if len(modelInstanceIDs(mgr, "llm:remote-chat")) != 0 {
		t.Fatal("precondition: expected no instances after recreate=false hard kill")
	}

	reloaded := mgr.ReloadModel("llm:remote-chat", *mcfg.MaxInstances, mcfg)
	if reloaded["added"].(int) != 2 {
		t.Fatalf("reload added = %v, want 2 remote placements", reloaded["added"])
	}
	after := modelInstanceIDs(mgr, "llm:remote-chat")
	if !containsString(after, "llm:remote-chat@h1") || !containsString(after, "llm:remote-chat@h2") {
		t.Fatalf("reload did not heal remote placements: %v", after)
	}
	for _, inst := range mgr.GetModelInstances("llm:remote-chat") {
		if !inst.isRemote() {
			t.Fatalf("reload created a LOCAL instance %s for a remote-only model", inst.InstanceID)
		}
	}
}

func TestScaleModelCreatesNoLocalPoolForRemoteOnlyModel(t *testing.T) {
	cfg := remotePlacementTestConfig()
	mgr := NewInstanceManager(cfg, "python3", t.TempDir())
	setupInstances(cfg, mgr, "python3", t.TempDir())

	// The auto-wake guard un-parks a model by calling ScaleModel(1). For a
	// remote-only model that must NOT create a local "#0" worker — there is
	// no local adapter for it, so the worker would crash-loop.
	mcfg := cfg.Models["llm:remote-chat"]
	result := mgr.ScaleModel("llm:remote-chat", 1, mcfg)
	if result["added"].(int) != 0 {
		t.Fatalf("scale-up added %v local instances for a remote-only model, want 0", result["added"])
	}
	for _, inst := range mgr.GetModelInstances("llm:remote-chat") {
		if !inst.isRemote() {
			t.Fatalf("local instance %s exists for remote-only model", inst.InstanceID)
		}
	}
}

func TestHardKillModelRecreatesConfiguredSlots(t *testing.T) {
	mgr := NewInstanceManager(&Config{VRAMBudgetGB: 70}, "python3", t.TempDir())
	cfg := ModelConfig{
		MemoryGB:      4,
		MaxConcurrent: 1,
		MaxInstances:  intPtr(2),
	}

	initial := mgr.ScaleModel("demo", 2, cfg)
	if initial["added"].(int) != 2 {
		t.Fatalf("initial scale added = %v, want 2", initial["added"])
	}
	before := modelInstanceIDs(mgr, "demo")

	result := mgr.HardKillModel("demo", true, &cfg)
	after := modelInstanceIDs(mgr, "demo")

	if result["killed"].(int) != 2 {
		t.Fatalf("hard kill killed = %v, want 2", result["killed"])
	}
	if result["recreated"].(int) != 2 {
		t.Fatalf("hard kill recreated = %v, want 2", result["recreated"])
	}
	if len(after) != 2 {
		t.Fatalf("dispatch instances after hard kill = %d, want 2", len(after))
	}
	if len(before) != len(after) {
		t.Fatalf("instance count changed unexpectedly: before=%d after=%d", len(before), len(after))
	}
}
