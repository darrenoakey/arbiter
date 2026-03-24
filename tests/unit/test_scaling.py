"""Tests for runtime model scaling (max_instances changes)."""
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from arbiter.config import save_model_config_field
from arbiter.memory import EvictionImpossible, MemoryManager, ModelState
from arbiter.store import JobStore


def make_adapter(load_time=0):
    """Create a mock adapter."""
    adapter = MagicMock()
    def fake_load(device="cuda"):
        if load_time:
            time.sleep(load_time)
    adapter.load = fake_load
    adapter.unload = MagicMock()
    return adapter


def make_adapter_cls(load_time=0):
    """Return a callable that produces mock adapters."""
    def factory():
        return make_adapter(load_time)
    return factory


@pytest.fixture
def executor():
    pool = ThreadPoolExecutor(max_workers=4)
    yield pool
    pool.shutdown(wait=False)


@pytest.fixture
def mm(executor, mock_logger):
    return MemoryManager(budget_gb=100, executor=executor, event_logger=mock_logger)


# --- Store: cancel_queued_for_model / cancel_all_queued ---

class TestStoreCancelQueued:
    def test_cancel_queued_for_model(self, store):
        store.create_job("model-a", "t", {})
        store.create_job("model-a", "t", {})
        store.create_job("model-b", "t", {})

        cancelled = store.cancel_queued_for_model("model-a")
        assert cancelled == 2

        # model-b untouched
        remaining = store.list_jobs(state="queued")
        assert len(remaining) == 1
        assert remaining[0].model_id == "model-b"

    def test_cancel_queued_for_model_skips_running(self, store):
        j1 = store.create_job("model-a", "t", {})
        j2 = store.create_job("model-a", "t", {})
        store.update_state(j1.id, "running", started_at=time.time())

        cancelled = store.cancel_queued_for_model("model-a")
        assert cancelled == 1  # only the queued one

        fetched = store.get_job(j1.id)
        assert fetched.state == "running"  # untouched

    def test_cancel_queued_for_model_empty(self, store):
        cancelled = store.cancel_queued_for_model("nonexistent")
        assert cancelled == 0

    def test_cancel_all_queued(self, store):
        store.create_job("model-a", "t", {})
        store.create_job("model-b", "t", {})
        j3 = store.create_job("model-c", "t", {})
        store.update_state(j3.id, "running", started_at=time.time())

        cancelled = store.cancel_all_queued()
        assert cancelled == 2  # only the queued ones

        # running job untouched
        fetched = store.get_job(j3.id)
        assert fetched.state == "running"

    def test_cancel_all_queued_empty(self, store):
        assert store.cancel_all_queued() == 0


# --- Memory: scale_model ---

class TestScaleUp:
    @pytest.mark.asyncio
    async def test_scale_up_adds_instances(self, mm):
        """Scaling up from 2 to 5 adds 3 unloaded instances."""
        for i in range(2):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")

        result = await mm.scale_model(
            "mdl", 5, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )
        assert result["added"] == 3
        assert result["removed"] == 0
        assert result["condemned"] == 0

        instances = mm.get_model_instances("mdl")
        assert len(instances) == 5
        # New instances start unloaded
        for iid in instances[2:]:
            assert mm.get_slot(iid).state == ModelState.UNLOADED

    @pytest.mark.asyncio
    async def test_scale_up_from_single_instance(self, mm):
        """Scaling up from 1 (bare model_id) to 3 adds 2 new instances."""
        mm.register("mdl", make_adapter(), memory_gb=4.0)  # instance_id=None -> key="mdl"

        result = await mm.scale_model(
            "mdl", 3, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )
        assert result["added"] == 2

        instances = mm.get_model_instances("mdl")
        assert len(instances) == 3
        assert instances[0] == "mdl"  # original
        assert instances[1] == "mdl#1"
        assert instances[2] == "mdl#2"

    @pytest.mark.asyncio
    async def test_scale_up_no_change(self, mm):
        """Scaling to the same count is a no-op."""
        for i in range(3):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")

        result = await mm.scale_model(
            "mdl", 3, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )
        assert result == {"added": 0, "removed": 0, "condemned": 0}

    @pytest.mark.asyncio
    async def test_scale_up_new_instances_usable(self, mm):
        """New instances can be loaded and used."""
        mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id="mdl#0")
        await mm.scale_model(
            "mdl", 3, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )

        # Load a new instance
        await mm.ensure_loaded("mdl#1")
        assert mm.get_slot("mdl#1").state == ModelState.LOADED
        assert mm.get_slot("mdl#1").active_count == 1


class TestScaleDown:
    @pytest.mark.asyncio
    async def test_scale_down_removes_unloaded(self, mm):
        """Unloaded excess instances are removed immediately."""
        for i in range(4):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")

        result = await mm.scale_model(
            "mdl", 2, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )
        assert result["removed"] == 2
        assert result["condemned"] == 0

        instances = mm.get_model_instances("mdl")
        assert len(instances) == 2
        assert instances == ["mdl#0", "mdl#1"]
        # Slots are gone
        assert mm.get_slot("mdl#2") is None
        assert mm.get_slot("mdl#3") is None

    @pytest.mark.asyncio
    async def test_scale_down_evicts_idle_loaded(self, mm):
        """Loaded but idle instances are evicted and removed immediately."""
        adapters = []
        for i in range(4):
            a = make_adapter()
            adapters.append(a)
            mm.register("mdl", a, memory_gb=4.0, instance_id=f"mdl#{i}")

        # Load instances #2 and #3, then release them (make idle)
        await mm.ensure_loaded("mdl#2")
        mm.release("mdl#2")
        await mm.ensure_loaded("mdl#3")
        mm.release("mdl#3")
        assert mm.used_gb == 8.0

        result = await mm.scale_model(
            "mdl", 2, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )
        assert result["removed"] == 2
        assert result["condemned"] == 0

        # VRAM freed
        assert mm.used_gb == 0.0
        # Adapters were unloaded
        assert adapters[2].unload.called
        assert adapters[3].unload.called

    @pytest.mark.asyncio
    async def test_scale_down_condemns_active(self, mm):
        """Active instances are condemned, not killed."""
        for i in range(4):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")

        # Load and keep active (simulate running job)
        await mm.ensure_loaded("mdl#2")
        await mm.ensure_loaded("mdl#3")
        # active_count = 1 on both

        result = await mm.scale_model(
            "mdl", 2, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )
        assert result["removed"] == 0
        assert result["condemned"] == 2

        # Active instances still have their slots
        assert mm.get_slot("mdl#2") is not None
        assert mm.get_slot("mdl#3") is not None
        # But removed from dispatch list
        assert "mdl#2" not in mm.get_model_instances("mdl")
        assert "mdl#3" not in mm.get_model_instances("mdl")

    @pytest.mark.asyncio
    async def test_condemned_auto_evicts_on_release(self, mm):
        """Condemned instances auto-evict when their jobs finish."""
        adapter = make_adapter()
        for i in range(3):
            mm.register("mdl", make_adapter() if i != 2 else adapter, memory_gb=4.0, instance_id=f"mdl#{i}")

        # Load #2 and keep active
        await mm.ensure_loaded("mdl#2")
        assert mm.used_gb == 4.0

        # Scale down — #2 gets condemned
        result = await mm.scale_model(
            "mdl", 2, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )
        assert result["condemned"] == 1

        # Simulate job finishing
        mm.release("mdl#2")

        # Give the async cleanup task a moment to run
        await asyncio.sleep(0.1)

        # Instance should be evicted and slot removed
        assert mm.get_slot("mdl#2") is None
        assert adapter.unload.called
        assert mm.used_gb == 0.0

    @pytest.mark.asyncio
    async def test_scale_down_mixed(self, mm):
        """Mix of unloaded, idle-loaded, and active instances."""
        adapters = []
        for i in range(6):
            a = make_adapter()
            adapters.append(a)
            mm.register("mdl", a, memory_gb=4.0, instance_id=f"mdl#{i}")

        # #0, #1: untouched (unloaded)
        # #2: loaded and idle
        await mm.ensure_loaded("mdl#2")
        mm.release("mdl#2")
        # #3: loaded and active
        await mm.ensure_loaded("mdl#3")
        # #4: loaded and idle
        await mm.ensure_loaded("mdl#4")
        mm.release("mdl#4")
        # #5: loaded and active
        await mm.ensure_loaded("mdl#5")

        # Scale down to 2 — removes #2, #3, #4, #5
        result = await mm.scale_model(
            "mdl", 2, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )

        # #2 and #4 are idle-loaded -> removed (evicted)
        # #3 and #5 are active -> condemned
        assert result["removed"] == 2
        assert result["condemned"] == 2
        assert mm.get_model_instances("mdl") == ["mdl#0", "mdl#1"]


class TestScaleToZero:
    @pytest.mark.asyncio
    async def test_scale_to_zero_all_unloaded(self, mm):
        """Scale to 0 removes all unloaded instances."""
        for i in range(3):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")

        result = await mm.scale_model(
            "mdl", 0, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )
        assert result["removed"] == 3
        assert mm.get_model_instances("mdl") == []

    @pytest.mark.asyncio
    async def test_scale_to_zero_with_active(self, mm):
        """Scale to 0 condemns active instances, they finish gracefully."""
        adapter = make_adapter()
        for i in range(2):
            mm.register("mdl", make_adapter() if i != 0 else adapter, memory_gb=4.0, instance_id=f"mdl#{i}")

        await mm.ensure_loaded("mdl#0")  # active

        result = await mm.scale_model(
            "mdl", 0, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )
        assert result["condemned"] == 1  # #0 is active
        assert result["removed"] == 1   # #1 is unloaded
        assert mm.get_model_instances("mdl") == []

        # pick_instance returns None (no instances in dispatch list)
        assert mm.pick_instance("mdl") is None

    @pytest.mark.asyncio
    async def test_scale_zero_then_back_up(self, mm):
        """Scale to 0, then back to 3 — new instances are created."""
        for i in range(2):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")

        await mm.scale_model(
            "mdl", 0, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )
        assert mm.get_model_instances("mdl") == []

        result = await mm.scale_model(
            "mdl", 3, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )
        assert result["added"] == 3
        instances = mm.get_model_instances("mdl")
        assert len(instances) == 3
        # New instances are usable
        await mm.ensure_loaded(instances[0])
        assert mm.get_slot(instances[0]).state == ModelState.LOADED


class TestScaleDownThenUp:
    @pytest.mark.asyncio
    async def test_scale_down_then_up_reuses_indices(self, mm):
        """Scale 4->2->4 assigns correct indices without collision."""
        for i in range(4):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")

        await mm.scale_model(
            "mdl", 2, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )
        assert mm.get_model_instances("mdl") == ["mdl#0", "mdl#1"]

        result = await mm.scale_model(
            "mdl", 4, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )
        assert result["added"] == 2
        instances = mm.get_model_instances("mdl")
        assert len(instances) == 4
        # Should get indices 2 and 3
        assert "mdl#2" in instances
        assert "mdl#3" in instances


class TestSnapshotWithCondemned:
    @pytest.mark.asyncio
    async def test_snapshot_shows_condemned(self, mm):
        """Condemned instances appear in snapshot with condemned flag."""
        for i in range(3):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")

        await mm.ensure_loaded("mdl#2")  # active

        await mm.scale_model(
            "mdl", 2, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )

        snap = mm.snapshot()
        model_entry = snap["models"][0]
        assert "instances" in model_entry
        assert model_entry.get("condemned_instances") == 1

        # Find the condemned instance
        condemned = [i for i in model_entry["instances"] if i.get("condemned")]
        assert len(condemned) == 1
        assert condemned[0]["instance_id"] == "mdl#2"


# --- Config: save_model_config_field ---

class TestSaveModelConfigField:
    def test_creates_config_from_default(self, tmp_path):
        """Creates config.json from config.default.json when it doesn't exist."""
        local_dir = tmp_path / "local"
        local_dir.mkdir()
        default = {
            "vram_budget_gb": 100,
            "models": {
                "model-a": {"memory_gb": 4.0, "max_instances": 1},
                "model-b": {"memory_gb": 8.0, "max_instances": 2},
            },
        }
        (local_dir / "config.default.json").write_text(json.dumps(default))

        save_model_config_field("model-a", "max_instances", 5, project_root=tmp_path)

        config_path = local_dir / "config.json"
        assert config_path.exists()
        saved = json.loads(config_path.read_text())
        assert saved["models"]["model-a"]["max_instances"] == 5
        # Other fields preserved
        assert saved["vram_budget_gb"] == 100
        assert saved["models"]["model-b"]["max_instances"] == 2

    def test_updates_existing_config(self, tmp_path):
        """Updates an existing config.json without losing other fields."""
        local_dir = tmp_path / "local"
        local_dir.mkdir()
        existing = {
            "vram_budget_gb": 100,
            "models": {
                "model-a": {"memory_gb": 4.0, "max_instances": 1, "keep_alive_seconds": 600},
            },
        }
        (local_dir / "config.json").write_text(json.dumps(existing))

        save_model_config_field("model-a", "max_instances", 10, project_root=tmp_path)

        saved = json.loads((local_dir / "config.json").read_text())
        assert saved["models"]["model-a"]["max_instances"] == 10
        assert saved["models"]["model-a"]["keep_alive_seconds"] == 600  # preserved

    def test_no_config_files_creates_minimal(self, tmp_path):
        """Creates config.json even when neither config file exists."""
        save_model_config_field("model-a", "max_instances", 3, project_root=tmp_path)

        config_path = tmp_path / "local" / "config.json"
        assert config_path.exists()
        saved = json.loads(config_path.read_text())
        assert saved["models"]["model-a"]["max_instances"] == 3

    def test_zero_value_persists(self, tmp_path):
        """max_instances=0 is correctly persisted (not treated as falsy)."""
        local_dir = tmp_path / "local"
        local_dir.mkdir()
        (local_dir / "config.default.json").write_text(json.dumps({
            "models": {"model-a": {"memory_gb": 4.0, "max_instances": 4}},
        }))

        save_model_config_field("model-a", "max_instances", 0, project_root=tmp_path)

        saved = json.loads((local_dir / "config.json").read_text())
        assert saved["models"]["model-a"]["max_instances"] == 0


class TestNextInstanceIndex:
    @pytest.mark.asyncio
    async def test_next_index_with_numbered(self, mm):
        for i in range(3):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")
        assert mm._next_instance_index("mdl") == 3

    @pytest.mark.asyncio
    async def test_next_index_with_bare_model_id(self, mm):
        mm.register("mdl", make_adapter(), memory_gb=4.0)  # bare "mdl"
        assert mm._next_instance_index("mdl") == 1

    @pytest.mark.asyncio
    async def test_next_index_empty(self, mm):
        assert mm._next_instance_index("mdl") == 0

    @pytest.mark.asyncio
    async def test_next_index_skips_condemned(self, mm):
        """Condemned slots still in _slots prevent index reuse."""
        for i in range(4):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")

        # Load and keep #3 active, then scale down so #3 is condemned
        await mm.ensure_loaded("mdl#3")
        await mm.scale_model(
            "mdl", 2, make_adapter_cls(), memory_gb=4.0, keep_alive_s=300, max_concurrent=1,
        )
        # mdl#2 removed (was unloaded), mdl#3 condemned (still in _slots)
        # next index should be 4, not 3
        assert mm._next_instance_index("mdl") >= 3
