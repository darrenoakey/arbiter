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
	mgr := NewInstanceManager(70, "python3", t.TempDir())
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

func TestHardKillModelRecreatesConfiguredSlots(t *testing.T) {
	mgr := NewInstanceManager(70, "python3", t.TempDir())
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
