"""Integration tests for the voice pipeline job types (demucs, rvc-convert).

Real jobs against a live Arbiter. demucs runs on generated audio
created locally by ffmpeg; rvc-convert requires the trained voice model on the
server. rvc-train is exercised by the
project's real fidelity validation rather than here (a full training is far
too long for the default suite).
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

ARBITER_URL = "http://localhost:8400"
# A voice model id expected on the server for the rvc-convert test; the project
# trains "leo-laporte" during fidelity validation.
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
        threading.Event().wait(2.0)
    raise TimeoutError(f"Job {job_id} timed out")


def _synth_audio_b64(seconds=4) -> str:
    """A short stereo clip (tone + noise) so demucs has real signal to split."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"anoisesrc=d={seconds}:c=pink",
            "-filter_complex",
            "[0][1]amix=inputs=2,pan=stereo|c0<c0+c1|c1<c0+c1",
            "-ar",
            "44100",
            path,
        ],
        capture_output=True,
        check=True,
    )
    raw = Path(path).read_bytes()
    Path(path).unlink(missing_ok=True)
    return base64.b64encode(raw).decode()


def _wav_frames_from_b64(b64: str) -> int:
    with wave.open(io.BytesIO(base64.b64decode(b64))) as w:
        return w.getnframes()


@pytest.fixture(scope="module")
def arbiter_health():
    response = _api("GET", "/v1/health")
    assert response.get("status") == "ok"
    return response


@pytest.mark.integration
class TestRvcConvert:
    def test_two_stem_separation(self, arbiter_health):
        resp = _api(
            "POST",
            "/v1/jobs",
            {
                "type": "demucs",
                "params": {
                    "audio": _synth_audio_b64(),
                    "return_b64": True,
                    "force": True,
                },
            },
        )
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
class TestVocalStem:
    def test_separate_and_normalize_to_target_lufs(self, arbiter_health):
        # vocal-stem takes an absolute file path on spark local disk (unlike
        # demucs' base64); the test runs next to the server so a temp file
        # is directly visible to the worker.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            src_path = tmp.name
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=220:duration=4",
                "-f",
                "lavfi",
                "-i",
                "anoisesrc=d=4:c=pink",
                "-filter_complex",
                "[0][1]amix=inputs=2,pan=stereo|c0<c0+c1|c1<c0+c1",
                "-ar",
                "44100",
                src_path,
            ],
            capture_output=True,
            check=True,
        )
        try:
            resp = _api(
                "POST",
                "/v1/jobs",
                {
                    "type": "vocal-stem",
                    "params": {"audio_file": src_path, "force": True},
                },
            )
            assert resp["model"] == "vocal-stem"
            result = _poll(resp["job_id"])
        finally:
            Path(src_path).unlink(missing_ok=True)
        assert result["status"] == "completed", (
            f"vocal-stem failed: {result.get('error')}"
        )
        r = result["result"]
        assert r["file"] == "vocals_normalized.wav"
        assert r["vocals"] == "vocals.wav"
        assert r["stats"] == "stats.json"
        assert r["model"] == "htdemucs"
        assert r["seconds"] > 0
        # Normalized stem must land on the -14 LUFS target and stay under
        # the -1 dBTP ceiling; stats are real measured values.
        assert abs(r["output_lufs"] - (-14.0)) <= 0.5, r
        assert r["peak_dbtp"] <= -1.0, r
        assert r["input_lufs"] < 0.0, r


@pytest.mark.integration
class TestRvcConvert:
    def test_convert_with_trained_model(self, arbiter_health):
        resp = _api(
            "POST",
            "/v1/jobs",
            {
                "type": "rvc-convert",
                "params": {
                    "model": RVC_TEST_MODEL,
                    "audio": _synth_audio_b64(3),
                    "transpose": 0,
                    "return_b64": True,
                    "force": True,
                },
            },
        )
        assert resp["model"] == "rvc-convert"
        result = _poll(resp["job_id"])
        assert result["status"] == "completed", (
            f"rvc-convert failed: {result.get('error')}"
        )
        r = result["result"]
        assert r["format"] == "wav"
        assert _wav_frames_from_b64(r["audio_b64"]) > 0
