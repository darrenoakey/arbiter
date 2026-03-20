"""Integration tests for moondream via Arbiter API — uses real image from example.com."""
from __future__ import annotations

import base64
import json
import time
from urllib.request import Request, urlopen

import pytest

ARBITER_URL = "http://localhost:8400"


def _api(method, path, data=None):
    url = f"{ARBITER_URL}{path}"
    body = json.dumps(data).encode() if data else None
    req = Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _poll(job_id, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = _api("GET", f"/v1/jobs/{job_id}")
        if resp["status"] in ("completed", "failed", "cancelled"):
            return resp
        time.sleep(2.0)
    raise TimeoutError(f"Job {job_id} timed out")


def _get_test_image_b64():
    """Download a real photo from Lorem Picsum and return as base64."""
    req = Request("https://picsum.photos/512/512", headers={"User-Agent": "arbiter-test"})
    with urlopen(req, timeout=15) as resp:
        return base64.b64encode(resp.read()).decode()


@pytest.fixture(scope="module")
def image_b64():
    """Cached test image for the whole module."""
    return _get_test_image_b64()


@pytest.fixture(scope="module")
def arbiter_available():
    """Check if Arbiter is running."""
    try:
        resp = _api("GET", "/v1/health")
        return resp.get("status") == "ok"
    except Exception:
        return False


@pytest.mark.integration
class TestMoondreamCaption:
    def test_caption_job(self, arbiter_available, image_b64):
        if not arbiter_available:
            pytest.skip("Arbiter not running")

        resp = _api("POST", "/v1/jobs", {
            "type": "caption",
            "params": {"image": image_b64, "length": "short"},
        })
        assert resp["status"] == "queued"
        assert resp["model"] == "moondream"
        job_id = resp["job_id"]

        result = _poll(job_id)
        assert result["status"] == "completed", f"Job failed: {result.get('error')}"
        assert "caption" in result["result"], f"No caption in result: {result['result']}"
        assert len(result["result"]["caption"]) > 0
        print(f"Caption: {result['result']['caption']}")


@pytest.mark.integration
class TestMoondreamQuery:
    def test_query_job(self, arbiter_available, image_b64):
        if not arbiter_available:
            pytest.skip("Arbiter not running")

        resp = _api("POST", "/v1/jobs", {
            "type": "query",
            "params": {"image": image_b64, "question": "What colors do you see in this image?"},
        })
        assert resp["status"] == "queued"
        assert resp["model"] == "moondream"
        job_id = resp["job_id"]

        result = _poll(job_id)
        assert result["status"] == "completed", f"Job failed: {result.get('error')}"
        assert "answer" in result["result"], f"No answer in result: {result['result']}"
        assert len(result["result"]["answer"]) > 0
        print(f"Answer: {result['result']['answer']}")


@pytest.mark.integration
class TestMoondreamDetect:
    def test_detect_job(self, arbiter_available, image_b64):
        if not arbiter_available:
            pytest.skip("Arbiter not running")

        resp = _api("POST", "/v1/jobs", {
            "type": "detect",
            "params": {"image": image_b64, "object": "dice"},
        })
        assert resp["status"] == "queued"
        job_id = resp["job_id"]

        result = _poll(job_id)
        assert result["status"] == "completed", f"Job failed: {result.get('error')}"
        assert "objects" in result["result"], f"No objects in result: {result['result']}"
        print(f"Detected: {result['result']['objects']}")


@pytest.mark.integration
class TestMoondreamPoint:
    def test_point_job(self, arbiter_available, image_b64):
        if not arbiter_available:
            pytest.skip("Arbiter not running")

        resp = _api("POST", "/v1/jobs", {
            "type": "point",
            "params": {"image": image_b64, "object": "dice"},
        })
        assert resp["status"] == "queued"
        job_id = resp["job_id"]

        result = _poll(job_id)
        assert result["status"] == "completed", f"Job failed: {result.get('error')}"
        assert "points" in result["result"], f"No points in result: {result['result']}"
        print(f"Points: {result['result']['points']}")


@pytest.mark.integration
class TestJobTypeDispatch:
    """Verify that different job types for the same model dispatch correctly."""

    def test_caption_returns_caption_not_answer(self, arbiter_available, image_b64):
        if not arbiter_available:
            pytest.skip("Arbiter not running")

        resp = _api("POST", "/v1/jobs", {
            "type": "caption",
            "params": {"image": image_b64},
        })
        result = _poll(resp["job_id"])
        assert result["status"] == "completed"
        assert result["result"]["task"] == "caption"
        assert "caption" in result["result"]

    def test_query_returns_answer_not_caption(self, arbiter_available, image_b64):
        if not arbiter_available:
            pytest.skip("Arbiter not running")

        resp = _api("POST", "/v1/jobs", {
            "type": "query",
            "params": {"image": image_b64, "question": "Describe this image"},
        })
        result = _poll(resp["job_id"])
        assert result["status"] == "completed"
        assert result["result"]["task"] == "query"
        assert "answer" in result["result"]
