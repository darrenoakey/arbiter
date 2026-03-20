"""Integration tests for the Arbiter API using FastAPI TestClient."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_with_mocks(tmp_path):
    """Create a test app with mock config and adapters."""
    # Create local/config.default.json in tmp dir
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    config = {
        "vram_budget_gb": 24,
        "host": "127.0.0.1",
        "port": 18400,
        "default_keep_alive_seconds": 60,
        "models": {
            "birefnet": {
                "memory_gb": 2, "max_concurrent": 1,
                "avg_inference_ms": 200, "load_ms": 1000,
            },
        },
    }
    (local_dir / "config.default.json").write_text(json.dumps(config))

    # Create output dirs
    (tmp_path / "output" / "jobs").mkdir(parents=True)
    (tmp_path / "output" / "logs").mkdir(parents=True)

    # Patch the project root
    with patch("arbiter.server._PROJECT_ROOT", tmp_path):
        with patch("arbiter.config._PROJECT_ROOT", tmp_path):
            from arbiter.server import app
            yield app


@pytest.fixture
def client(app_with_mocks):
    """TestClient that goes through lifespan."""
    with TestClient(app_with_mocks) as c:
        yield c


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data


class TestPsEndpoint:
    def test_ps(self, client):
        resp = client.get("/v1/ps")
        assert resp.status_code == 200
        data = resp.json()
        assert "vram_budget_gb" in data
        assert "vram_used_gb" in data
        assert "models" in data


class TestJobSubmission:
    def test_submit_valid_job(self, client):
        resp = client.post("/v1/jobs", json={
            "type": "background-remove",
            "params": {"image": "base64data"},
        })
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["model"] == "birefnet"

    def test_submit_unknown_type(self, client):
        resp = client.post("/v1/jobs", json={
            "type": "nonexistent",
            "params": {},
        })
        assert resp.status_code == 422  # Pydantic validation for enum

    def test_get_job_status(self, client):
        resp = client.post("/v1/jobs", json={
            "type": "background-remove",
            "params": {"image": "base64data"},
        })
        job_id = resp.json()["job_id"]

        resp2 = client.get(f"/v1/jobs/{job_id}")
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("queued", "scheduled", "running", "completed", "failed")

    def test_get_nonexistent_job(self, client):
        resp = client.get("/v1/jobs/nonexistent")
        assert resp.status_code == 404

    def test_list_jobs(self, client):
        client.post("/v1/jobs", json={"type": "background-remove", "params": {"image": "x"}})
        client.post("/v1/jobs", json={"type": "background-remove", "params": {"image": "y"}})

        resp = client.get("/v1/jobs")
        assert resp.status_code == 200
        jobs = resp.json()
        assert len(jobs) >= 2

    def test_cancel_queued_job(self, client):
        resp = client.post("/v1/jobs", json={
            "type": "background-remove",
            "params": {"image": "x"},
        })
        job_id = resp.json()["job_id"]

        resp2 = client.delete(f"/v1/jobs/{job_id}")
        assert resp2.status_code == 200

    def test_cancel_nonexistent(self, client):
        resp = client.delete("/v1/jobs/nonexistent")
        assert resp.status_code == 404
