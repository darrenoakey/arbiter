"""Real audited GPU-worker test; run detached after deploying the adapter."""

import base64
import io
import json
import math
import threading
import time
import zipfile

import pytest
from PIL import Image

from arbiter.adapters.aesthetic_scorer import _SCORE_NAMES


def _completed(client, job_id):
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        job = client.status(job_id)
        if job["status"] in ("completed", "failed", "cancelled"):
            assert job["status"] == "completed", job
            print(json.dumps({"job_id": job_id, "result": job["result"]}))
            return job["result"]
        threading.Event().wait(1)
    raise TimeoutError(f"GPU job {job_id} did not complete")


@pytest.mark.integration
@pytest.mark.gpu
def test_real_gpu_zip_batches_match_legacy_single_image(tmp_path):
    from arbiter_client import ArbiterClient, stage_file

    client = ArbiterClient()
    image = io.BytesIO()
    Image.new("RGB", (91, 79), (35, 90, 160)).save(image, "JPEG")
    first = image.getvalue()
    path = tmp_path / "originals.zip"
    names = [f"originals/{index:02}.jpg" for index in range(63)]
    names.insert(17, "broken.heic")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("originals/", b"")
        for index, name in enumerate(names):
            if name == "broken.heic":
                archive.writestr(name, b"corrupt original")
            else:
                pixels = io.BytesIO()
                Image.new("RGB", (91, 79), (35 + index, 90, 160)).save(pixels, "JPEG")
                archive.writestr(name, first if index == 0 else pixels.getvalue())
    batch_id = client.submit("aesthetic-score", images_file=stage_file(path), force=True)
    single_id = client.submit(
        "aesthetic-score", image=base64.b64encode(first).decode(), force=True
    )
    batch = _completed(client, batch_id)
    single = _completed(client, single_id)
    assert "scores" not in batch and "items" not in single
    assert [item["name"] for item in batch["items"]] == names
    assert "error" in batch["items"][17] and "scores" not in batch["items"][17]
    assert batch["score_names"] == single["score_names"] == list(_SCORE_NAMES)
    assert batch["elapsed_ms"] < 60_000
    for index, item in enumerate(batch["items"]):
        if index == 17:
            continue
        assert "error" not in item
        assert set(item["scores"]) == set(_SCORE_NAMES)
        assert all(math.isfinite(value) for value in item["scores"].values())
    assert batch["items"][0]["scores"] == pytest.approx(single["scores"], abs=1e-4)
    assert batch["items"][0]["scores"] != batch["items"][-1]["scores"]
