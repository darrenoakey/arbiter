"""Spark-local arbiter client for the photo-enhance worker.

The worker runs ON spark, so staging writes straight into the shared inbox
directory and results are read from spark-local paths — no SMB mapping and
no base64 inline payloads for the big images. Everything else mirrors the
Mac-side photo-enhance client protocol exactly.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

from .cache_shim import image_bytes
from .settings import Settings

log = logging.getLogger(__name__)

POLL_SECONDS = 0.5
HTTP_TIMEOUT = 30
RESULT_WAIT_SECONDS = 30


class ArbiterClient:
    """Thin typed client for the spark GPU job server (localhost)."""

    def __init__(self, settings: Settings, why: str) -> None:
        self.settings = settings
        self.why = why

    # ##################################################################
    # stage image
    # save a PIL image into the shared inbox under a unique name; the path
    # is valid on spark itself and in every spark-local worker.
    def stage_image(self, image: Image.Image, label: str) -> str:
        name = f"{uuid.uuid4().hex[:12]}_{label}.png"
        path = Path(self.settings.inbox_local) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")
        return str(path)

    # ##################################################################
    # submit
    def submit(self, job_type: str, params: dict) -> str:
        body = {
            "type": job_type,
            "params": params,
            "source": {"who": "photo-enhance-worker", "why": self.why},
        }
        request = urllib.request.Request(
            f"{self.settings.arbiter_url}/v1/jobs",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            return json.load(response)["job_id"]

    # ##################################################################
    # wait
    def wait(self, job_id: str) -> dict:
        pause = threading.Event()
        while True:
            with urllib.request.urlopen(f"{self.settings.arbiter_url}/v1/jobs/{job_id}", timeout=HTTP_TIMEOUT) as r:
                record = json.load(r)
            if record["status"] == "completed":
                return record
            if record["status"] in ("failed", "cancelled"):
                raise RuntimeError(f"arbiter job {job_id} {record['status']}: {record.get('error')}")
            pause.wait(POLL_SECONDS)

    # ##################################################################
    # run
    def run(self, job_type: str, params: dict) -> dict:
        return self.wait(self.submit(job_type, params))["result"]

    # ##################################################################
    # result image
    # prefer inlined base64; otherwise read the spark-local result file.
    def result_image(self, result: dict) -> Image.Image:
        if result.get("data"):
            return Image.open(io.BytesIO(base64.b64decode(result["data"])))
        path = Path(result["result_path"])
        pause = threading.Event()
        waited = 0.0
        while not path.is_file() and waited < RESULT_WAIT_SECONDS:
            pause.wait(POLL_SECONDS)
            waited += POLL_SECONDS
        if not path.is_file():
            raise FileNotFoundError(f"result {path} not visible after {RESULT_WAIT_SECONDS}s")
        image = Image.open(path)
        image.load()
        return image

    # ##################################################################
    # aesthetic score
    def aesthetic_score(self, image: Image.Image) -> dict[str, float]:
        encoded = base64.b64encode(image_bytes(image)).decode("ascii")
        result = self.run("aesthetic-score", {"image": encoded})
        return {key: float(value) for key, value in result["scores"].items()}

    # ##################################################################
    # remove background
    def remove_background(self, image: Image.Image) -> Image.Image:
        staged = self.stage_image(image, "segment")
        result = self.run("background-remove", {"image_file": staged, "force": True})
        return self.result_image(result)

    # ##################################################################
    # reference edit
    def submit_reference_edit(self, image: Image.Image, prompt: str, seed: int, steps: int, guidance: float) -> str:
        staged = self.stage_image(image, "reference")
        params = {"prompt": prompt, "image_file": staged, "seed": seed, "steps": steps, "guidance_scale": guidance}
        return self.submit("reference-image-edit", params)

    def reference_edit_result(self, job_id: str) -> Image.Image:
        return self.result_image(self.wait(job_id)["result"]).convert("RGB")
