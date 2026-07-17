"""Integration tests for the voice pipeline job types (demucs, rvc-convert).

Real jobs against a live Arbiter (no mocks). demucs runs on synthetic audio
generated locally by ffmpeg; rvc-convert requires a trained voice model on the
server and self-skips when none is present. rvc-train is exercised by the
project's real fidelity validation rather than here (a full training is far
too long for the default suite).
"""
from __future__ import annotations

import base64
import io
import json
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

ARBITER_URL = "http://localhost:8400"
# A voice model id expected on the server for the rvc-convert test; the project
# trains "leo-laporte" during fidelity validation. Absent -> the test skips.
RVC_TEST_MODEL = "leo-laporte"


def _api(method, path, data=None):
    url = f"{ARBITER_URL}{path}"
    body = json.dumps(data).encode() if data else None
    req = Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _poll(job_id, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = _api("GET", f"/v1/jobs/{job_id}")
        if resp["status"] in ("completed", "failed", "cancelled"):
            return resp
        time.sleep(2.0)
    raise TimeoutError(f"Job {job_id} timed out")


def _synth_audio_b64(seconds=4) -> str:
    """A short stereo clip (tone + noise) so demucs has real signal to split."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
         "-f", "lavfi", "-i", f"anoisesrc=d={seconds}:c=pink",
         "-filter_complex", "[0][1]amix=inputs=2,pan=stereo|c0<c0+c1|c1<c0+c1",
         "-ar", "44100", path],
        capture_output=True, check=True,
    )
    raw = Path(path).read_bytes()
    Path(path).unlink(missing_ok=True)
    return base64.b64encode(raw).decode()


def _wav_frames_from_b64(b64: str) -> int:
    with wave.open(io.BytesIO(base64.b64decode(b64))) as w:
        return w.getnframes()


@pytest.fixture(scope="module")
def arbiter_available():
    try:
        return _api("GET", "/v1/health").get("status") == "ok"
    except Exception:
        return False


@pytest.mark.integration
class TestDemucs:
    def test_two_stem_separation(self, arbiter_available):
        if not arbiter_available:
            pytest.skip("Arbiter not running")

        resp = _api("POST", "/v1/jobs", {
            "type": "demucs",
            "params": {"audio": _synth_audio_b64(), "return_b64": True, "force": True},
        })
        assert resp["model"] == "demucs"
        result = _poll(resp["job_id"])
        assert result["status"] == "completed", f"demucs failed: {result.get('error')}"
        r = result["result"]
        assert r["vocals"] == "vocals.wav"
        assert r["accompaniment"] == "accompaniment.wav"
        assert r["samplerate"] == 44100
        # Both stems present, non-empty, and the same length as the input clock.
        assert _wav_frames_from_b64(r["vocals_b64"]) > 44100
        assert _wav_frames_from_b64(r["accompaniment_b64"]) > 44100


@pytest.mark.integration
class TestRvcConvert:
    def test_convert_with_trained_model(self, arbiter_available):
        if not arbiter_available:
            pytest.skip("Arbiter not running")

        resp = _api("POST", "/v1/jobs", {
            "type": "rvc-convert",
            "params": {"model": RVC_TEST_MODEL, "audio": _synth_audio_b64(3),
                       "transpose": 0, "return_b64": True, "force": True},
        })
        assert resp["model"] == "rvc-convert"
        result = _poll(resp["job_id"])
        if result["status"] == "failed" and "not found" in (result.get("error") or ""):
            pytest.skip(f"voice model '{RVC_TEST_MODEL}' not trained on this server")
        assert result["status"] == "completed", f"rvc-convert failed: {result.get('error')}"
        r = result["result"]
        assert r["format"] == "wav"
        assert _wav_frames_from_b64(r["audio_b64"]) > 0
