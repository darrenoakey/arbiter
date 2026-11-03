"""Offline-only P-H3 adapter tests: no weights, GPU, torch or upstream code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from arbiter.adapters import registry
from arbiter.adapters.base import InferenceError, LoadError
from arbiter.adapters import p_h3_image as ph3
from arbiter.image_policy import is_disabled_still_image_model
from arbiter.schemas import JOB_TYPE_PARAMS, JOB_TYPE_TO_MODEL, PH3ImageParams

MODEL_ID = "p-h3-image-bf16"
JOB_TYPE = "p-h3-image"


def test_exact_registration_and_job_binding():
    assert issubclass(registry.get_adapter_class(MODEL_ID), ph3.PH3ImageAdapter)
    assert JOB_TYPE_TO_MODEL[JOB_TYPE] == MODEL_ID
    assert JOB_TYPE_PARAMS[JOB_TYPE] is PH3ImageParams
    for other in ("p-h3-image", "PrunaAI/P-H3-Image", "p-h3-image-w4a8", "p-h3-image-comfy"):
        with pytest.raises(KeyError):
            registry.get_adapter_class(other)


def test_policy_opens_only_bf16_comparison_id():
    assert not is_disabled_still_image_model(MODEL_ID)
    for other in ("p-h3-image", "p-h3-image-w4a8", "p-h3-image-comfy", "PrunaAI/P-H3-Image", "minimax-h3-image", "ph3-image"):
        assert is_disabled_still_image_model(other)


def test_schema_is_like_for_like_by_default_and_rejects_hidden_changes():
    p = PH3ImageParams(prompt="botanist")
    assert (p.width, p.height, p.seed, p.steps, p.prompt_upsampling) == (1024, 1024, 42, 8, False)
    for bad in ({"steps": 9}, {"negative_prompt": "x"}, {"image": "AAAA"}, {"sigmas": [1, 0]}, {"unknown": 1}):
        with pytest.raises(ValidationError):
            PH3ImageParams(prompt="x", **bad)
    PH3ImageParams(prompt="native", width=2912, height=2912, seed=42001)


def test_parser_defaults_native_arm_and_fail_closed_knobs():
    r = ph3.parse_request({"prompt": "botanist", "seed": 42001})
    assert (r.width, r.height, r.prompt_upsampling) == (1024, 1024, False)
    r = ph3.parse_request({"prompt": "botanist", "seed": 42001, "width": 2912, "height": 2912})
    assert (r.width, r.height) == (ph3.NATIVE_SIDE, ph3.NATIVE_SIDE)
    for bad in ({"steps": 16}, {"steps": True}, {"width": 1000}, {"height": 4097, "width": 4096}, {"guidance_scale": 1}, {"prompt_upsampling": "false"}, {"prompt_upsampling": True}):
        with pytest.raises(InferenceError):
            ph3.parse_request({"prompt": "x", **bad})


def _fixture_snapshot(root: Path, monkeypatch):
    rows = []
    for rel in ("modular_model_index.json", "LICENSE", "LICENSE-APACHE", "NOTICE"):
        rows.append({"type": "file", "path": rel})
    rows.append({"type": "file", "path": "transformer/config.json"})
    for directory in ph3._REQUIRED_DIRS:
        if directory != "transformer":
            rows.append({"type": "file", "path": f"{directory}/fixture.bin"})
    for row in rows:
        target = root / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if row["path"] == "modular_model_index.json":
            body = json.dumps({"_class_name": ph3.EXPECTED_PIPELINE_CLASS,
                               "_diffusers_version": ph3.DIFFUSERS_DECLARED_VERSION,
                               "sample_sigmas": list(ph3.SAMPLE_SIGMAS)}).encode()
        elif row["path"] == "transformer/config.json":
            body = json.dumps({"hidden_size": ph3.EXPECTED_DIM, "num_layers": ph3.EXPECTED_LAYERS}).encode()
        else:
            body = (row["path"] + "\n").encode()
        target.write_bytes(body)
        row.update({"size": len(body), "oid": hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()})
    monkeypatch.setattr(ph3, "_SNAPSHOT", rows)
    return root


def test_verification_real_files_and_git_hashes(tmp_path, monkeypatch):
    root = _fixture_snapshot(tmp_path / "snapshot", monkeypatch)
    info = ph3.verify_ph3_snapshot(root)
    assert info["revision"] == ph3.PH3_REVISION
    assert info["verified_files"] == 14
    # Real content mutation is detected without downloading or importing the model.
    (root / "transformer/config.json").write_text('{"hidden_size":5376,"num_layers":50} ')
    with pytest.raises(LoadError, match="size mismatch"):
        ph3.verify_ph3_snapshot(root)


def test_verification_rejects_missing_symlink_and_wrong_contract(tmp_path, monkeypatch):
    root = _fixture_snapshot(tmp_path / "snapshot", monkeypatch)
    (root / "modular_model_index.json").unlink()
    with pytest.raises(LoadError, match="missing"):
        ph3.verify_ph3_snapshot(root)
    root = _fixture_snapshot(tmp_path / "snapshot2", monkeypatch)
    (root / "tokenizer/fixture.bin").unlink()
    (root / "tokenizer/fixture.bin").symlink_to(root / "LICENSE")
    with pytest.raises(LoadError, match="real file"):
        ph3.verify_ph3_snapshot(root)


def test_official_snapshot_pins_lfs_weights_and_pipeline_shape():
    files = ph3.snapshot_files()
    assert files["transformer/diffusion_pytorch_model-00001-of-00007.safetensors"]["lfs"]["oid"]
    assert len([x for x in files if x.startswith("transformer/") and x.endswith(".safetensors")]) == 7
    assert ph3.PH3_REVISION == "7f98acef503cf2bcfba821a78cb1cf1a9928b2df"
    assert ph3.DIFFUSERS_DECLARED_VERSION == "0.42.0.dev0"
    assert ph3.SAMPLE_SIGMAS[-1] == 0.0
