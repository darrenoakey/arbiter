"""Tests for job state transitions."""
import time

import pytest

from arbiter.store import JobStore


class TestJobLifecycle:
    def test_queued_to_running_to_completed(self, store):
        job = store.create_job("m", "t", {})
        assert job.state == "queued"

        store.update_state(job.id, "scheduled")
        assert store.get_job(job.id).state == "scheduled"

        store.update_state(job.id, "running", started_at=time.time())
        assert store.get_job(job.id).state == "running"

        store.update_state(job.id, "completed", result={"ok": True}, finished_at=time.time())
        j = store.get_job(job.id)
        assert j.state == "completed"
        assert j.result == {"ok": True}
        assert j.finished_at is not None

    def test_queued_to_failed(self, store):
        job = store.create_job("m", "t", {})
        store.update_state(job.id, "failed", error="load error", finished_at=time.time())
        j = store.get_job(job.id)
        assert j.state == "failed"
        assert j.error == "load error"

    def test_queued_to_cancelled(self, store):
        job = store.create_job("m", "t", {})
        assert store.cancel_job(job.id) is True
        assert store.get_job(job.id).state == "cancelled"

    def test_running_to_cancelled(self, store):
        job = store.create_job("m", "t", {})
        store.update_state(job.id, "running", started_at=time.time())
        # Store.cancel_job only works for queued/scheduled
        assert store.cancel_job(job.id) is False
        # But direct update_state works (scheduler does this)
        store.update_state(job.id, "cancelled", finished_at=time.time())
        assert store.get_job(job.id).state == "cancelled"

    def test_timestamps_populated(self, store):
        job = store.create_job("m", "t", {})
        assert job.created_at > 0
        assert job.started_at is None
        assert job.finished_at is None

        now = time.time()
        store.update_state(job.id, "running", started_at=now)
        j = store.get_job(job.id)
        assert abs(j.started_at - now) < 1

        store.update_state(job.id, "completed", finished_at=time.time())
        j = store.get_job(job.id)
        assert j.finished_at is not None
