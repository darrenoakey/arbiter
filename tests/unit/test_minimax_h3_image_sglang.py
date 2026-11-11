"""SGLang MXFP8 MiniMax-H3-Image adapter: snapshot verification against real on-disk
files with real Hugging Face download metadata, recipe flags, and job-type binding.
No GPU, torch, SGLang or model run; static unit tests only."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from arbiter.adapters import registry
from arbiter.adapters.base import InferenceError, LoadError
from arbiter.adapters.minimax_h3_image import MINIMAX_H3_IMAGE_REVISION
from arbiter.adapters.minimax_h3_image_sglang import (
    MODEL_ID,
    RECIPE,
    SHARED_WITH_BF16,
    MiniMaxH3ImageSglangAdapter,
    verify_native_snapshot,
)
from tests.unit.test_minimax_h3_image import _write, make_snapshot


def make_native(root: Path, bf16: Path, revision: str = MINIMAX_H3_IMAGE_REVISION) -> Path:
    entry = root / "sglang" / "MiniMax-H3-Image"
    _write(entry, "transformer/config.json", b'{"_class_name": "MiniMaxH3DiTModel"}', revision, lfs=False)
    _write(entry, "transformer/model-00001-of-00012.safetensors", b"dit" * 400, revision, lfs=True)
    _write(entry, "vae/config.json", b'{"_class_name": "MiniMaxH3ImageVAE"}', revision, lfs=False)
    _write(entry, "vae/diffusion_pytorch_model.safetensors", b"vae" * 300, revision, lfs=True)
    _write(entry, "pdd_config.json", b'{"steps": 6}', revision, lfs=False)
    _write(entry, "pdd_fused_heads.safetensors", b"pdd" * 200, revision, lfs=True)
    for rel in SHARED_WITH_BF16:
        source = bf16 / rel
        if source.is_dir():
            for child in source.rglob("*"):
                if child.is_file():
                    target = entry / child.relative_to(bf16)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(child, target)
        else:
            os.link(source, entry / rel)
    return entry


@pytest.fixture
def snapshots(tmp_path):
    bf16 = make_snapshot(tmp_path)
    return make_native(tmp_path, bf16), bf16


def test_registration_and_exact_recipe(snapshots):
    assert registry.get_adapter_class(MODEL_ID) is MiniMaxH3ImageSglangAdapter
    native, bf16 = snapshots
    kwargs = MiniMaxH3ImageSglangAdapter(native, bf16).generator_kwargs()
    assert kwargs["model_path"] == str(native)
    assert kwargs["component_paths"] == {"text_encoder": str(bf16 / "text_encoder")}
    assert kwargs["quantization"] == "mxfp8"
    assert kwargs["component_quantizations"] == {"text_encoder": "mxfp8"}
    assert kwargs["component_attention_backends"] == {"transformer": "fp8_fa_sm120"}
    assert kwargs["component_residency"] == {"all": "resident"}
    kwargs["component_quantizations"]["text_encoder"] = "fp8"
    assert RECIPE["component_quantizations"] == {"text_encoder": "mxfp8"}, "recipe must not be mutable via kwargs"


def test_verify_happy_path_counts_native_and_bf16_files(snapshots):
    native, bf16 = snapshots
    info = verify_native_snapshot(native, bf16)
    assert info["verified_native_files"] == 6
    assert info["verified_bf16_files"] > 0
    assert info["text_encoder_dir"] == str(bf16 / "text_encoder")


def test_verify_rejects_tampered_wrong_revision_copied_or_own_encoder(snapshots, tmp_path):
    native, bf16 = snapshots
    (native / "transformer" / "model-00001-of-00012.safetensors").write_bytes(b"tampered")
    with pytest.raises(LoadError, match="content hash"):
        verify_native_snapshot(native, bf16)

    other = make_native(tmp_path / "rev", bf16, revision="0" * 40)
    with pytest.raises(LoadError, match="revision"):
        verify_native_snapshot(other, bf16)

    copied = make_native(tmp_path / "copy", bf16)
    (copied / "LICENSE").unlink()
    (copied / "LICENSE").write_bytes((bf16 / "LICENSE").read_bytes())
    with pytest.raises(LoadError, match="not the verified BF16 snapshot file"):
        verify_native_snapshot(copied, bf16)

    own = make_native(tmp_path / "own", bf16)
    (own / "text_encoder").mkdir()
    with pytest.raises(LoadError, match="must not carry its own text_encoder"):
        verify_native_snapshot(own, bf16)


def test_load_fails_closed_without_cap_and_infer_refuses_other_job_types(snapshots, monkeypatch):
    native, bf16 = snapshots
    adapter = MiniMaxH3ImageSglangAdapter(native, bf16)
    monkeypatch.delenv("ARBITER_MEMORY_GB", raising=False)
    with pytest.raises(LoadError, match="memory cap"):
        adapter.load("cuda")
    with pytest.raises(InferenceError, match="serves only"):
        adapter.infer({"_job_type": "minimax-h3-image", "prompt": "x"}, native, threading.Event())


def test_result_metadata_is_quantized_not_bf16():
    precision = MiniMaxH3ImageSglangAdapter.result_precision
    assert precision["transformer"].startswith("mxfp8")
    assert precision["text_encoder"].startswith("mxfp8")
    assert precision["attention"].startswith("fp8_fa_sm120")
    assert "bfloat16" not in precision.values()
