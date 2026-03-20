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
        assert snap["vram_used_gb"] == 4.0
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
