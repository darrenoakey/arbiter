"""Qwen-Image-2.1 adapter unit tests (no GPU, no framework imports)."""

from __future__ import annotations

import threading

import pytest

from arbiter.adapters.base import InferenceError
from arbiter.adapters.qwen_image_21 import (
    QWEN_IMAGE_21_HF_ID,
    QwenImage21Adapter,
    snap_side,
)


def _loaded_adapter() -> QwenImage21Adapter:
    adapter = QwenImage21Adapter()
    adapter._pipe = object()  # type: ignore[assignment]  # never invoked on these paths
    return adapter


def test_prompt_is_required(tmp_path):
    adapter = _loaded_adapter()
    with pytest.raises(InferenceError, match="prompt is required"):
        adapter.infer({}, tmp_path, threading.Event())


def test_text_to_image_does_not_require_an_image(tmp_path):
    """Unlike the reference editor, the unified model has a T2I path: a bare
    prompt must get past validation (and then fail only on the stub pipe)."""
    adapter = _loaded_adapter()
    with pytest.raises(Exception) as excinfo:
        adapter.infer({"prompt": "a panda riding a bicycle"}, tmp_path, threading.Event())
    # The stub pipe is a plain object; the point is that we never raise the
    # input-image InferenceError on the T2I path.
    assert "requires an input image" not in str(excinfo.value)


def test_edit_mode_resolves_condition_image_before_inference(tmp_path):
    adapter = _loaded_adapter()
    with pytest.raises(InferenceError, match="bad input image|No image"):
        adapter.infer({"prompt": "enhance", "image": "not-base64!!"}, tmp_path, threading.Event())


def test_snap_side_clamps_and_aligns():
    assert snap_side(1024) == 1024
    assert snap_side(2752) == 2752
    assert snap_side(10000) == 2752
    assert snap_side(100) == 256
    assert snap_side(1000) == 992  # snapped down to /16


def test_adapter_identity():
    adapter = QwenImage21Adapter()
    assert adapter.model_id == "qwen-image-2.1"
    assert QWEN_IMAGE_21_HF_ID == "Qwen/Qwen-Image-2.1"


def test_estimate_time_scales_with_steps_and_edit_mode():
    adapter = QwenImage21Adapter()
    t2i = adapter.estimate_time({"prompt": "x", "steps": 40})
    edit = adapter.estimate_time({"prompt": "x", "steps": 40, "image_file": "/i.jpg"})
    assert edit > t2i > 0
    assert adapter.estimate_time({"prompt": "x", "steps": 20}) < t2i
