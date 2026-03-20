"""Tests for SQLite job store."""
import time

import pytest

from arbiter.store import JobStore, Job


class TestJobStore:
    def test_create_and_get(self, store):
        job = store.create_job("model-a", "image-generate", {"prompt": "test"}, priority=100)
        assert job.id
        assert job.state == "queued"
        assert job.model_id == "model-a"
        assert job.payload == {"prompt": "test"}

        fetched = store.get_job(job.id)
        assert fetched is not None
        assert fetched.id == job.id
        assert fetched.payload == {"prompt": "test"}

    def test_get_nonexistent(self, store):
        assert store.get_job("nonexistent") is None

    def test_list_jobs(self, store):
        store.create_job("model-a", "type-a", {}, priority=1)
        store.create_job("model-b", "type-b", {}, priority=2)
        store.create_job("model-a", "type-a", {}, priority=3)

        all_jobs = store.list_jobs()
        assert len(all_jobs) == 3

        model_a = store.list_jobs(model_id="model-a")
        assert len(model_a) == 2

    def test_list_by_state(self, store):
        j1 = store.create_job("model-a", "t", {})
        j2 = store.create_job("model-b", "t", {})
        store.update_state(j1.id, "running", started_at=time.time())

        queued = store.list_jobs(state="queued")
        assert len(queued) == 1
        assert queued[0].id == j2.id

        running = store.list_jobs(state="running")
        assert len(running) == 1
        assert running[0].id == j1.id

    def test_pick_next_job_priority(self, store):
        store.create_job("model-a", "t", {}, priority=100)
        store.create_job("model-b", "t", {}, priority=10)
        store.create_job("model-c", "t", {}, priority=50)

        job = store.pick_next_job()
        assert job.model_id == "model-b"  # lowest priority score

    def test_pick_next_excludes_models(self, store):
        store.create_job("model-a", "t", {}, priority=10)
        store.create_job("model-b", "t", {}, priority=20)

        job = store.pick_next_job(exclude_models={"model-a"})
        assert job.model_id == "model-b"

    def test_pick_next_empty(self, store):
        assert store.pick_next_job() is None

    def test_update_state(self, store):
        job = store.create_job("model-a", "t", {})
        store.update_state(job.id, "running", started_at=time.time())
        fetched = store.get_job(job.id)
        assert fetched.state == "running"
        assert fetched.started_at is not None

    def test_update_result(self, store):
        job = store.create_job("model-a", "t", {})
        store.update_state(job.id, "completed", result={"format": "png"}, finished_at=time.time())
        fetched = store.get_job(job.id)
        assert fetched.state == "completed"
        assert fetched.result == {"format": "png"}

    def test_update_error(self, store):
        job = store.create_job("model-a", "t", {})
        store.update_state(job.id, "failed", error="OOM", finished_at=time.time())
        fetched = store.get_job(job.id)
        assert fetched.error == "OOM"

    def test_cancel_queued(self, store):
        job = store.create_job("model-a", "t", {})
        assert store.cancel_job(job.id) is True
        fetched = store.get_job(job.id)
        assert fetched.state == "cancelled"

    def test_cancel_running_fails(self, store):
        job = store.create_job("model-a", "t", {})
        store.update_state(job.id, "running", started_at=time.time())
        assert store.cancel_job(job.id) is False  # can't cancel running via store

    def test_count_by_state(self, store):
        store.create_job("model-a", "t", {})
        store.create_job("model-a", "t", {})
        j3 = store.create_job("model-b", "t", {})
        store.update_state(j3.id, "running", started_at=time.time())

        counts = store.count_by_state()
        assert counts.get("queued", 0) == 2
        assert counts.get("running", 0) == 1

    def test_count_running(self, store):
        j1 = store.create_job("model-a", "t", {})
        j2 = store.create_job("model-a", "t", {})
        store.update_state(j1.id, "running", started_at=time.time())
        assert store.count_running("model-a") == 1

    def test_update_priority(self, store):
        store.create_job("model-a", "t", {}, priority=100)
        store.create_job("model-a", "t", {}, priority=200)
        store.create_job("model-b", "t", {}, priority=50)

        count = store.update_priority("model-a", 5)
        assert count == 2

        job = store.pick_next_job()
        assert job.model_id == "model-a"  # now has priority 5

    def test_crash_recovery(self, tmp_db):
        store1 = JobStore(tmp_db)
        j1 = store1.create_job("model-a", "t", {})
        j2 = store1.create_job("model-b", "t", {})
        store1.update_state(j1.id, "running", started_at=time.time())
        store1.update_state(j2.id, "scheduled")
        store1.close()

        # Simulate restart
        store2 = JobStore(tmp_db)
        recovered = store2.recover_from_crash()
        assert recovered == 2

        f1 = store2.get_job(j1.id)
        f2 = store2.get_job(j2.id)
        assert f1.state == "queued"
        assert f2.state == "queued"
        assert f1.started_at is None
        store2.close()

    def test_persistence(self, tmp_db):
        store1 = JobStore(tmp_db)
        job = store1.create_job("model-a", "t", {"x": 1})
        store1.close()

        store2 = JobStore(tmp_db)
        fetched = store2.get_job(job.id)
        assert fetched is not None
        assert fetched.payload == {"x": 1}
        store2.close()
