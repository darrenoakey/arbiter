"""TRELLIS.2 image-to-3d adapter tests (no GPU, no model load)."""

from __future__ import annotations

import base64
import io
import threading
from pathlib import Path

import pytest

from arbiter.adapters.base import InferenceError
from arbiter.adapters.trellis2 import (
    HF_ID,
    Trellis2Adapter,
    load_job_image,
    pipeline_type_for_resolution,
    resolve_weights_path,
)
from arbiter.schemas import JOB_TYPE_PARAMS, JOB_TYPE_TO_MODEL, ImageTo3DParams, JobType


def _png_b64(mode: str, color: tuple) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new(mode, (2, 2), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_image_to_3d_job_type_registered() -> None:
    assert JobType.IMAGE_TO_3D.value == "image-to-3d"
    assert JOB_TYPE_TO_MODEL["image-to-3d"] == "trellis2"
    assert JOB_TYPE_PARAMS["image-to-3d"] is ImageTo3DParams
    params = ImageTo3DParams()
    assert params.resolution == 1024
    assert params.steps == 12
    assert params.seed == 42
    assert params.texture_size == 2048
    assert params.decimation == 500000
    assert params.include_stl is False
    assert params.image is None
    assert not hasattr(params, "prompt")


def test_schema_has_no_source_or_config_path() -> None:
    fields = ImageTo3DParams.model_fields
    for name in ("source", "config", "weights_path", "model_path", "prompt", "text"):
        assert name not in fields


def test_pipeline_type_and_hf_id() -> None:
    assert pipeline_type_for_resolution(1024) == "1024_cascade"
    assert HF_ID == "microsoft/TRELLIS.2-4B"


def test_missing_image_fails_before_pipeline(tmp_path: Path) -> None:
    adapter = Trellis2Adapter()
    with pytest.raises(InferenceError, match="No image"):
        adapter.infer({}, tmp_path, threading.Event())


def test_rgba_alpha_is_preserved() -> None:
    loaded = load_job_image({"image": _png_b64("RGBA", (10, 20, 30, 128))})
    assert loaded.mode == "RGBA"
    assert loaded.getpixel((0, 0))[3] == 128


def test_rgb_stays_rgb_for_birefnet_preprocess() -> None:
    loaded = load_job_image({"image": _png_b64("RGB", (10, 20, 30))})
    assert loaded.mode == "RGB"


def test_image_present_without_load_names_unloaded_pipeline(tmp_path: Path) -> None:
    adapter = Trellis2Adapter()
    with pytest.raises(InferenceError, match="pipeline is not loaded"):
        adapter.infer({"image": _png_b64("RGBA", (255, 0, 0, 200))}, tmp_path, threading.Event())


def test_estimate_time_grows_with_resolution() -> None:
    adapter = Trellis2Adapter()
    t512 = adapter.estimate_time({"resolution": 512, "steps": 12})
    t1024 = adapter.estimate_time({"resolution": 1024, "steps": 12})
    t1536 = adapter.estimate_time({"resolution": 1536, "steps": 12})
    assert t1536 > t1024 > t512 > 0


def test_weights_path_from_toml(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    snapshot = tmp_path / "TRELLIS.2-4B"
    snapshot.mkdir()
    (snapshot / "pipeline.json").write_text("{}")
    config.write_text(f'[trellis2]\nweights_path = "{snapshot}"\n')
    assert resolve_weights_path(config_path=config, snapshot_dir=snapshot) == str(snapshot)


@pytest.mark.parametrize("field,value", [
    ("resolution", 2048), ("steps", 0), ("steps", 51), ("seed", -1),
    ("texture_size", 8192), ("decimation", 1000001),
])
def test_schema_rejects_unbounded_work(field: str, value: int) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ImageTo3DParams.model_validate({field: value})


def test_worker_validates_params_before_model_execution(tmp_path: Path) -> None:
    from pydantic import ValidationError

    params = {"image": _png_b64("RGBA", (10, 20, 30, 128)), "steps": 5000}
    with pytest.raises(ValidationError):
        Trellis2Adapter().infer(params, tmp_path, threading.Event())
