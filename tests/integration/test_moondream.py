"""Integration tests for moondream via Arbiter API using a real local image."""

from __future__ import annotations

import base64
import json
import threading
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
        threading.Event().wait(2.0)
    raise TimeoutError(f"Job {job_id} timed out")


def _write_test_image(path):
    """Write a deterministic RGB gradient in the portable pixmap format."""
    pixels = bytearray()
    for row in range(256):
        for column in range(256):
            pixels.extend((column, row, (column + row) // 2))
    path.write_bytes(b"P6\n256 256\n255\n" + pixels)


@pytest.fixture(scope="module")
def image_b64(tmp_path_factory):
    """Create one real image file for the whole module."""
    image_path = tmp_path_factory.mktemp("moondream-media") / "gradient.ppm"
    _write_test_image(image_path)
    return base64.b64encode(image_path.read_bytes()).decode()


@pytest.fixture(scope="module")
def arbiter_health():
    """Require the real Arbiter API used by this integration module."""
    response = _api("GET", "/v1/health")
    assert response.get("status") == "ok"
    return response


@pytest.fixture(scope="module")
def moondream_results(arbiter_health, image_b64):
    """Submit the real adapter jobs together so one loaded worker serves the batch."""
    requests = {
        "caption": {
            "type": "caption",
            "params": {"image": image_b64, "length": "short"},
        },
        "query": {
            "type": "query",
            "params": {
                "image": image_b64,
                "question": "What colors do you see in this image?",
            },
        },
        "detect": {
            "type": "detect",
            "params": {"image": image_b64, "object": "gradient"},
        },
        "point": {
            "type": "point",
            "params": {"image": image_b64, "object": "gradient"},
        },
    }
    submissions = {
        name: _api("POST", "/v1/jobs", request) for name, request in requests.items()
    }
    assert all(
        submission["model"] == "moondream" for submission in submissions.values()
    )
    return {
        name: _poll(submission["job_id"]) for name, submission in submissions.items()
    }


@pytest.mark.integration
class TestMoondreamCaption:
    def test_caption_job(self, moondream_results):
        result = moondream_results["caption"]
        assert result["status"] == "completed", f"Job failed: {result.get('error')}"
        assert "caption" in result["result"], (
            f"No caption in result: {result['result']}"
        )
        assert len(result["result"]["caption"]) > 0
        print(f"Caption: {result['result']['caption']}")


@pytest.mark.integration
class TestMoondreamQuery:
    def test_query_job(self, moondream_results):
        result = moondream_results["query"]
        assert result["status"] == "completed", f"Job failed: {result.get('error')}"
        assert "answer" in result["result"], f"No answer in result: {result['result']}"
        assert len(result["result"]["answer"]) > 0
        print(f"Answer: {result['result']['answer']}")


@pytest.mark.integration
class TestMoondreamDetect:
    def test_detect_job(self, moondream_results):
        result = moondream_results["detect"]
        assert result["status"] == "completed", f"Job failed: {result.get('error')}"
        assert "objects" in result["result"], (
            f"No objects in result: {result['result']}"
        )
        print(f"Detected: {result['result']['objects']}")


@pytest.mark.integration
class TestMoondreamPoint:
    def test_point_job(self, moondream_results):
        result = moondream_results["point"]
        assert result["status"] == "completed", f"Job failed: {result.get('error')}"
        assert "points" in result["result"], f"No points in result: {result['result']}"
        print(f"Points: {result['result']['points']}")


@pytest.mark.integration
class TestJobTypeDispatch:
    """Verify that different job types for the same model dispatch correctly."""

    def test_caption_returns_caption_not_answer(self, moondream_results):
        result = moondream_results["caption"]
        assert result["status"] == "completed"
        assert result["result"]["task"] == "caption"
        assert "caption" in result["result"]

    def test_query_returns_answer_not_caption(self, moondream_results):
        result = moondream_results["query"]
        assert result["status"] == "completed"
        assert result["result"]["task"] == "query"
        assert "answer" in result["result"]
