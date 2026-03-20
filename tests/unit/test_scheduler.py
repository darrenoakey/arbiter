"""Tests for the scheduler's SJF scoring and priority logic."""
import pytest

from arbiter.config import ArbiterConfig
from arbiter.scheduler import Scheduler
from arbiter.store import JobStore
from unittest.mock import MagicMock


@pytest.fixture
def scheduler_deps(sample_config, tmp_db, mock_logger):
    store = JobStore(tmp_db)
    memory = MagicMock()
    memory.is_loaded = MagicMock(return_value=False)
    memory.get_slot = MagicMock(return_value=None)
    memory.free_gb = 100.0
    worker = MagicMock()
    sched = Scheduler(
        config=sample_config, store=store, memory=memory,
        worker_pool=worker, event_logger=mock_logger,
    )
    return sched, store, memory


class TestSJFScoring:
    def test_priority_unloaded(self, scheduler_deps):
        sched, _, memory = scheduler_deps
        memory.is_loaded.return_value = False
        # model-a: avg=500 + load=2000 = 2500
        p = sched.compute_priority("model-a")
        assert p == 2500

    def test_priority_loaded(self, scheduler_deps):
        sched, _, memory = scheduler_deps
        memory.is_loaded.return_value = True
        # model-a: avg=500 + load=0 = 500
        p = sched.compute_priority("model-a")
        assert p == 500

    def test_shorter_job_wins(self, scheduler_deps):
        sched, store, memory = scheduler_deps
        memory.is_loaded.return_value = False

        # model-c: 100 + 500 = 600 (shortest)
        # model-a: 500 + 2000 = 2500
        # model-b: 2000 + 5000 = 7000

        store.create_job("model-a", "t", {}, priority=sched.compute_priority("model-a"))
        store.create_job("model-b", "t", {}, priority=sched.compute_priority("model-b"))
        store.create_job("model-c", "t", {}, priority=sched.compute_priority("model-c"))

        job = store.pick_next_job()
        assert job.model_id == "model-c"

    def test_loaded_model_preferred(self, scheduler_deps):
        sched, store, memory = scheduler_deps

        def is_loaded(mid):
            return mid == "model-b"
        memory.is_loaded = is_loaded

        # model-a unloaded: 500 + 2000 = 2500
        # model-b loaded: 2000 + 0 = 2000  <-- wins
        store.create_job("model-a", "t", {}, priority=sched.compute_priority("model-a"))
        store.create_job("model-b", "t", {}, priority=sched.compute_priority("model-b"))

        job = store.pick_next_job()
        assert job.model_id == "model-b"

    def test_rescore_on_load(self, scheduler_deps):
        sched, store, memory = scheduler_deps
        memory.is_loaded.return_value = False

        store.create_job("model-b", "t", {}, priority=sched.compute_priority("model-b"))
        store.create_job("model-c", "t", {}, priority=sched.compute_priority("model-c"))

        # model-c wins (priority 600 vs 7000)
        assert store.pick_next_job().model_id == "model-c"

        # Now model-b gets loaded — rescore
        memory.is_loaded = lambda mid: mid == "model-b"
        sched.rescore_model("model-b")

        # model-c: 600, model-b: 2000 (no load cost)
        # model-c still wins
        assert store.pick_next_job().model_id == "model-c"

    def test_unknown_model_returns_inf(self, scheduler_deps):
        sched, _, _ = scheduler_deps
        p = sched.compute_priority("nonexistent")
        assert p == float("inf")
