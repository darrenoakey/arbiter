"""Tests for GPU Memory Manager."""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from arbiter.memory import EvictionImpossible, MemoryManager, ModelState


@pytest.fixture
def executor():
    pool = ThreadPoolExecutor(max_workers=4)
    yield pool
    pool.shutdown(wait=False)


@pytest.fixture
def mm(executor, mock_logger):
    return MemoryManager(budget_gb=24, executor=executor, event_logger=mock_logger)


def make_adapter(load_time=0):
    """Create a mock adapter."""
    adapter = MagicMock()
    def fake_load(device="cuda"):
        if load_time:
            time.sleep(load_time)
    adapter.load = fake_load
    adapter.unload = MagicMock()
    return adapter


class TestMemoryManager:
    @pytest.mark.asyncio
    async def test_register_and_load(self, mm):
        adapter = make_adapter()
        mm.register("test-model", adapter, memory_gb=4.0)

        await mm.ensure_loaded("test-model")
        slot = mm.get_slot("test-model")
        assert slot.state == ModelState.LOADED
        assert slot.active_count == 1
        assert mm.used_gb == 4.0

    @pytest.mark.asyncio
    async def test_already_loaded_increments(self, mm):
        adapter = make_adapter()
        mm.register("test-model", adapter, memory_gb=4.0)

        await mm.ensure_loaded("test-model")
        await mm.ensure_loaded("test-model")
        slot = mm.get_slot("test-model")
        assert slot.active_count == 2

    @pytest.mark.asyncio
    async def test_release_decrements(self, mm):
        adapter = make_adapter()
        mm.register("test-model", adapter, memory_gb=4.0)

        await mm.ensure_loaded("test-model")
        mm.release("test-model")
        slot = mm.get_slot("test-model")
        assert slot.active_count == 0
        assert slot.last_active > 0

    @pytest.mark.asyncio
    async def test_eviction_when_full(self, mm):
        a1 = make_adapter()
        a2 = make_adapter()
        mm.register("big-model", a1, memory_gb=20.0, keep_alive_s=300)
        mm.register("other-model", a2, memory_gb=10.0, keep_alive_s=300)

        # Load big model
        await mm.ensure_loaded("big-model")
        mm.release("big-model")

        # Loading other-model should evict big-model (20 + 10 > 24)
        await mm.ensure_loaded("other-model")
        big_slot = mm.get_slot("big-model")
        assert big_slot.state == ModelState.UNLOADED
        assert a1.unload.called

    @pytest.mark.asyncio
    async def test_eviction_impossible(self, mm):
        a1 = make_adapter()
        a2 = make_adapter()
        mm.register("active-model", a1, memory_gb=20.0)
        mm.register("huge-model", a2, memory_gb=20.0)

        await mm.ensure_loaded("active-model")
        # active-model has active_count=1, can't be evicted

        with pytest.raises(EvictionImpossible):
            await mm.ensure_loaded("huge-model")

    @pytest.mark.asyncio
    async def test_multiple_models_fit(self, mm):
        a1 = make_adapter()
        a2 = make_adapter()
        a3 = make_adapter()
        mm.register("m1", a1, memory_gb=4.0)
        mm.register("m2", a2, memory_gb=8.0)
        mm.register("m3", a3, memory_gb=10.0)

        await mm.ensure_loaded("m1")
        await mm.ensure_loaded("m2")
        await mm.ensure_loaded("m3")

        assert mm.used_gb == 22.0  # 4 + 8 + 10
        for mid in ("m1", "m2", "m3"):
            assert mm.get_slot(mid).state == ModelState.LOADED

    @pytest.mark.asyncio
    async def test_lru_eviction_order(self, mm):
        a1 = make_adapter()
        a2 = make_adapter()
        a3 = make_adapter()
        mm.register("m1", a1, memory_gb=10.0)
        mm.register("m2", a2, memory_gb=10.0)
        mm.register("m3", a3, memory_gb=10.0)

        await mm.ensure_loaded("m1")
        mm.release("m1")
        await asyncio.sleep(0.01)

        await mm.ensure_loaded("m2")
        mm.release("m2")
        await asyncio.sleep(0.01)

        # m1 is oldest idle, m2 is newer idle
        # Loading m3 needs 10GB, budget is 24, used is 20 -> need to evict 6GB
        # Should evict m1 (oldest, 10GB)
        await mm.ensure_loaded("m3")

        assert mm.get_slot("m1").state == ModelState.UNLOADED  # evicted (LRU)
        assert mm.get_slot("m2").state == ModelState.LOADED  # kept
        assert mm.get_slot("m3").state == ModelState.LOADED

    @pytest.mark.asyncio
    async def test_snapshot(self, mm):
        adapter = make_adapter()
        mm.register("test-model", adapter, memory_gb=4.0)
        await mm.ensure_loaded("test-model")

        snap = mm.snapshot()
        assert snap["vram_budget_gb"] == 24
        assert snap["vram_configured_gb"] == 4.0
        assert len(snap["models"]) == 1
        assert snap["models"][0]["id"] == "test-model"
        assert snap["models"][0]["active_jobs"] == 1

    @pytest.mark.asyncio
    async def test_unregistered_model_raises(self, mm):
        with pytest.raises(KeyError, match="not registered"):
            await mm.ensure_loaded("nonexistent")

    @pytest.mark.asyncio
    async def test_free_gb(self, mm):
        adapter = make_adapter()
        mm.register("m", adapter, memory_gb=10.0)
        assert mm.free_gb == 24.0
        await mm.ensure_loaded("m")
        assert mm.free_gb == 14.0


class TestMultiInstance:
    """Tests for multi-instance model support."""

    @pytest.mark.asyncio
    async def test_register_multiple_instances(self, mm):
        for i in range(3):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")
        assert mm.get_model_instances("mdl") == ["mdl#0", "mdl#1", "mdl#2"]

    @pytest.mark.asyncio
    async def test_pick_instance_prefers_loaded(self, mm):
        for i in range(3):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")

        # Load instance #1
        await mm.ensure_loaded("mdl#1")
        mm.release("mdl#1")

        # pick_instance should prefer the loaded one
        picked = mm.pick_instance("mdl", max_concurrent=1)
        assert picked == "mdl#1"

    @pytest.mark.asyncio
    async def test_pick_instance_skips_busy(self, mm):
        for i in range(3):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")

        # Load and keep active on instance #0
        await mm.ensure_loaded("mdl#0")
        # active_count is now 1, max_concurrent=1 -> full

        # Should pick an unloaded instance
        picked = mm.pick_instance("mdl", max_concurrent=1)
        assert picked == "mdl#1"  # first unloaded

    @pytest.mark.asyncio
    async def test_pick_instance_least_busy(self, mm):
        for i in range(3):
            mm.register("mdl", make_adapter(), memory_gb=4.0, max_concurrent=2, instance_id=f"mdl#{i}")

        # Load both #0 and #1
        await mm.ensure_loaded("mdl#0")
        await mm.ensure_loaded("mdl#1")
        # #0 has active_count=1, #1 has active_count=1
        # Add another to #0
        await mm.ensure_loaded("mdl#0")
        # #0 has active_count=2, #1 has active_count=1

        picked = mm.pick_instance("mdl", max_concurrent=2)
        assert picked == "mdl#1"  # least busy

    @pytest.mark.asyncio
    async def test_multi_instance_independent_load_unload(self, mm):
        for i in range(2):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")

        await mm.ensure_loaded("mdl#0")
        await mm.ensure_loaded("mdl#1")
        assert mm.used_gb == 8.0

        mm.release("mdl#0")
        mm.release("mdl#1")

        # Each instance is independent
        assert mm.get_slot("mdl#0").state == ModelState.LOADED
        assert mm.get_slot("mdl#1").state == ModelState.LOADED

    @pytest.mark.asyncio
    async def test_is_loaded_any_instance(self, mm):
        for i in range(2):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")

        assert not mm.is_loaded("mdl")
        await mm.ensure_loaded("mdl#0")
        assert mm.is_loaded("mdl")

    @pytest.mark.asyncio
    async def test_multi_instance_eviction(self, mm):
        """Idle instances can be evicted independently."""
        for i in range(2):
            mm.register("mdl", make_adapter(), memory_gb=10.0, instance_id=f"mdl#{i}")
        mm.register("other", make_adapter(), memory_gb=10.0)

        # Load both instances
        await mm.ensure_loaded("mdl#0")
        mm.release("mdl#0")
        await asyncio.sleep(0.01)
        await mm.ensure_loaded("mdl#1")
        mm.release("mdl#1")

        # Budget=24, used=20. Loading "other" (10GB) needs eviction.
        # mdl#0 is oldest idle -> evicted first
        await mm.ensure_loaded("other")
        assert mm.get_slot("mdl#0").state == ModelState.UNLOADED
        assert mm.get_slot("mdl#1").state == ModelState.LOADED

    @pytest.mark.asyncio
    async def test_multi_instance_snapshot(self, mm):
        for i in range(2):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")

        await mm.ensure_loaded("mdl#0")
        snap = mm.snapshot()

        # Should be grouped as one model entry
        assert len(snap["models"]) == 1
        entry = snap["models"][0]
        assert entry["id"] == "mdl"
        assert entry["total_instances"] == 2
        assert entry["loaded_instances"] == 1
        assert entry["active_jobs"] == 1
        assert len(entry["instances"]) == 2

    @pytest.mark.asyncio
    async def test_total_capacity(self, mm):
        for i in range(3):
            mm.register("mdl", make_adapter(), memory_gb=4.0, instance_id=f"mdl#{i}")
        assert mm.total_capacity("mdl", max_concurrent=2) == 6
