"""Qwen-Image-2.1 adapter unit tests (no GPU, no framework imports)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

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


def test_sglang_adapter_identity_does_not_import_sglang():
    from arbiter.adapters.qwen_image_21_sglang import (
        QWEN_IMAGE_21_SGLANG_MODEL_ID,
        QwenImage21SglangAdapter,
    )

    adapter = QwenImage21SglangAdapter()
    assert adapter.model_id == "qwen-image-2.1-sglang"
    assert QWEN_IMAGE_21_SGLANG_MODEL_ID == adapter.model_id
    assert adapter.estimate_time({"prompt": "x", "steps": 40, "image_file": "/i.png"}) > adapter.estimate_time(
        {"prompt": "x", "steps": 40}
    )


def test_sglang_load_uses_pinned_runtime_entrypoint(monkeypatch):
    """The benchmark adapter must lazy-load SGLang only inside an Arbiter worker."""
    import arbiter.adapters.qwen_image_21_sglang as sglang_adapter

    calls: dict[str, object] = {}

    class FakeGenerator:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls.update(kwargs)
            return cls()

    def fake_import(name: str):
        calls["module"] = name
        return SimpleNamespace(DiffGenerator=FakeGenerator)

    monkeypatch.delenv("ARBITER_MEMORY_GB", raising=False)
    monkeypatch.setattr(sglang_adapter.importlib, "import_module", fake_import)
    adapter = sglang_adapter.QwenImage21SglangAdapter()
    adapter.load()

    assert calls == {
        "module": "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator",
        "model_path": "Qwen/Qwen-Image-2.1",
        "num_gpus": 1,
        "performance_mode": "speed",
        "attention_backend": "torch_sdpa",
    }


def test_turbo_spark_model_json_pins_nvme_snapshot_and_venv():
    """Deploy merges this file into spark local/config.json. It must not
    auto-download into the full T9 hub cache."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "config/spark/qwen-image-2.1-turbo.model.json").read_text())
    assert config["memory_gb"] == 52
    assert config["model_path"] == "/home/darren/nvme-models/Qwen-Image-2.1-Turbo"
    assert "auto_download" not in config
    assert config["worker_cmd"] == [
        "/home/darren/src/arbiter/venvs/qwenimage-turbo/bin/python",
        "-m",
        "arbiter.worker_main",
        "qwen-image-2.1-turbo",
    ]
    assert config["adapter_params"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert config["max_instances"] == 1


def test_turbo_uses_checkpoint_schedule_and_kv_cache():
    """The turbo checkpoint's sample_sigmas grid is authoritative.

    A caller-supplied step count must not be forwarded as an overriding
    sigma grid, and the stock adapter must not grow a use_kv_cache kwarg
    that its older diffusers pin would reject.
    """
    from arbiter.adapters.qwen_image_21 import (
        QWEN_IMAGE_21_TURBO_HF_ID,
        QWEN_IMAGE_21_TURBO_PATH,
        TURBO_SCHEDULE_STEPS,
        QwenImage21TurboAdapter,
    )

    turbo = QwenImage21TurboAdapter()
    assert turbo.model_id == "qwen-image-2.1-turbo"
    assert turbo.hf_model_path == QWEN_IMAGE_21_TURBO_PATH
    assert QWEN_IMAGE_21_TURBO_HF_ID == "Qwen/Qwen-Image-2.1-Turbo"
    kwargs, steps, requested = turbo.sampling_kwargs(
        {"prompt": "a red panda in a yellow raincoat", "steps": 40, "seed": 7},
        None,
    )
    assert steps == TURBO_SCHEDULE_STEPS == 8
    assert requested == 40
    assert kwargs["num_inference_steps"] == 8
    assert kwargs["use_kv_cache"] is True
    assert "sigmas" not in kwargs
    assert kwargs["width"] == 1024 and kwargs["height"] == 1024
    assert turbo.estimate_time({"prompt": "x"}) < QwenImage21Adapter().estimate_time(
        {"prompt": "x", "steps": 40}
    )

    stock = QwenImage21Adapter()
    stock_kwargs, stock_steps, stock_requested = stock.sampling_kwargs(
        {"prompt": "a red panda", "steps": 40},
        None,
    )
    assert stock_steps == stock_requested == 40
    assert "use_kv_cache" not in stock_kwargs
    assert "sigmas" not in stock_kwargs


def test_heretic_adapter_identity():
    """Exception #3: same pipeline, same behavior, abliterated text encoder
    checkpoint on spark; distinct model id."""
    from arbiter.adapters.qwen_image_21 import (
        QWEN_IMAGE_21_HERETIC_MODEL_ID,
        QWEN_IMAGE_21_HERETIC_PATH,
        QwenImage21HereticAdapter,
    )

    adapter = QwenImage21HereticAdapter()
    assert adapter.model_id == "qwen-image-2.1-heretic"
    assert QWEN_IMAGE_21_HERETIC_MODEL_ID == "qwen-image-2.1-heretic"
    assert adapter.hf_model_path == QWEN_IMAGE_21_HERETIC_PATH
    assert QWEN_IMAGE_21_HERETIC_PATH == "/mnt/t9/models/qwen-image-2.1-heretic"
    assert adapter.model_id != QwenImage21Adapter.model_id
    # Identical inference behavior inherited from the stock adapter.
    assert type(adapter).estimate_time is QwenImage21Adapter.estimate_time
