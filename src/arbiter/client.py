"""Arbiter Python client — submit jobs, poll results, decode outputs.

Usage:
    from arbiter.client import ArbiterClient

    client = ArbiterClient()  # defaults to http://localhost:8400

    # Submit and wait for result
    result = client.run("image-generate", prompt="a red fox", steps=4)
    with open("fox.png", "wb") as f:
        f.write(result["data_bytes"])

    # Or async: submit and poll separately
    job_id = client.submit("transcribe", audio=base64_audio)
    status = client.poll(job_id)  # returns when done
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_DEFAULT_URL = "http://localhost:8400"


class ArbiterError(Exception):
    """Error from Arbiter API."""
    def __init__(self, message: str, status_code: int = 0, job_id: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.job_id = job_id


class ArbiterClient:
    """Synchronous client for the Arbiter API."""

    def __init__(self, base_url: str = _DEFAULT_URL, timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, data: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode() if data else None
        req = Request(url, data=body, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            body = e.read().decode() if e.fp else ""
            raise ArbiterError(f"HTTP {e.code}: {body}", status_code=e.code)
        except URLError as e:
            raise ArbiterError(f"Connection failed: {e}")

    # --- Core API ---

    def submit(self, job_type: str, **params) -> str:
        """Submit a job. Returns job_id."""
        resp = self._request("POST", "/v1/jobs", {"type": job_type, "params": params})
        return resp["job_id"]

    def status(self, job_id: str) -> dict:
        """Get job status."""
        return self._request("GET", f"/v1/jobs/{job_id}")

    def cancel(self, job_id: str) -> dict:
        """Cancel a job."""
        return self._request("DELETE", f"/v1/jobs/{job_id}")

    def poll(self, job_id: str, interval: float = 1.0, timeout: float = 600) -> dict:
        """Poll until job completes. Returns full status with result.

        Raises ArbiterError if job fails or times out.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = self.status(job_id)
            state = s["status"]
            if state == "completed":
                # Decode base64 data if present
                if s.get("result") and "data" in s["result"]:
                    s["result"]["data_bytes"] = base64.b64decode(s["result"]["data"])
                return s
            elif state == "failed":
                raise ArbiterError(
                    f"Job {job_id} failed: {s.get('error', 'unknown')}",
                    job_id=job_id,
                )
            elif state == "cancelled":
                raise ArbiterError(f"Job {job_id} was cancelled", job_id=job_id)
            time.sleep(interval)
        raise ArbiterError(f"Job {job_id} timed out after {timeout}s", job_id=job_id)

    def run(self, job_type: str, timeout: float = 600, poll_interval: float = 1.0, **params) -> dict:
        """Submit a job and wait for result. Returns result dict with data_bytes if applicable."""
        job_id = self.submit(job_type, **params)
        return self.poll(job_id, interval=poll_interval, timeout=timeout)

    # --- Convenience methods ---

    def image_generate(self, prompt: str, **kwargs) -> bytes:
        """Generate an image. Returns PNG bytes."""
        result = self.run("image-generate", prompt=prompt, **kwargs)
        return result["result"]["data_bytes"]

    def background_remove(self, image_b64: str) -> bytes:
        """Remove background from image. Returns RGBA PNG bytes."""
        result = self.run("background-remove", image=image_b64)
        return result["result"]["data_bytes"]

    def caption(self, image_b64: str, length: str = "normal") -> str:
        """Caption an image. Returns caption text."""
        result = self.run("caption", image=image_b64, length=length)
        return result["result"].get("caption", "")

    def query(self, image_b64: str, question: str) -> str:
        """Ask a question about an image. Returns answer text."""
        result = self.run("query", image=image_b64, question=question)
        return result["result"].get("answer", "")

    def detect(self, image_b64: str, object: str) -> list:
        """Detect objects in an image. Returns list of bboxes."""
        result = self.run("detect", image=image_b64, object=object)
        return result["result"].get("objects", [])

    def transcribe(self, audio_b64: str, language: str = "en") -> dict:
        """Transcribe audio. Returns {text, segments}."""
        result = self.run("transcribe", audio=audio_b64, language=language)
        return result["result"]

    def tts(self, text: str, mode: str = "custom", speaker: str = "Aiden",
            ref_audio: str = "", ref_text: str = "",
            voice_description: str = "", **kwargs) -> bytes:
        """Text to speech. Returns WAV bytes."""
        if mode == "custom":
            result = self.run("tts-custom", text=text, speaker=speaker, **kwargs)
        elif mode == "clone":
            result = self.run("tts-clone", text=text, ref_audio=ref_audio,
                            ref_text=ref_text, **kwargs)
        elif mode == "design":
            result = self.run("tts-design", text=text,
                            voice_description=voice_description, **kwargs)
        else:
            raise ValueError(f"Unknown TTS mode: {mode}")
        return result["result"]["data_bytes"]

    def talking_head(self, image_b64: str, audio_b64: str, **kwargs) -> bytes:
        """Generate talking head video. Returns MP4 bytes."""
        result = self.run("talking-head", timeout=300,
                         image=image_b64, audio=audio_b64, **kwargs)
        return result["result"]["data_bytes"]

    def video_generate(self, segments: list[dict], audio_b64: str, **kwargs) -> bytes:
        """Generate video from segments + audio. Returns MP4 bytes."""
        result = self.run("video-generate", timeout=1800,
                         segments=segments, audio_b64=audio_b64, **kwargs)
        return result["result"]["data_bytes"]

    # --- System ---

    def health(self) -> dict:
        return self._request("GET", "/v1/health")

    def ps(self) -> dict:
        return self._request("GET", "/v1/ps")

    @staticmethod
    def file_to_b64(path: str | Path) -> str:
        """Read a file and return its base64 encoding."""
        return base64.b64encode(Path(path).read_bytes()).decode()

    @staticmethod
    def b64_to_file(b64_data: str, path: str | Path):
        """Decode base64 data and write to a file."""
        Path(path).write_bytes(base64.b64decode(b64_data))
