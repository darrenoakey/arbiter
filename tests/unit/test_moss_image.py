"""MOSS-Image-SFT (OpenMOSS, ~5B DiT + Qwen3.5 text encoder) adapter:
registration, policy, schema and fail-closed snapshot verification. No GPU,
no torch: every check here runs before the adapter imports a framework.
Verification runs against a real on-disk snapshot layout with real content
hashes, mirroring tests/unit/test_flux3_klein.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from arbiter.adapters import registry
from arbiter.adapters.base import InferenceError, LoadError
from arbiter.adapters.moss_image import (
    DEFAULT_WEIGHTS_DIR,
    MOSS_REVISION,
    MossImageSftAdapter,
    parse_request,
    verify_moss_snapshot,
)
from arbiter.image_policy import is_disabled_still_image_model
from arbiter.schemas import JOB_TYPE_PARAMS, JOB_TYPE_TO_MODEL, MossImageParams

MODEL_ID = "moss-image-sft"
JOB_TYPE = "moss-image"


# --- real snapshot fixture ----------------------------------------------------


def _write(entry: Path, relpath: str, body: bytes, revision: str, lfs: bool) -> str:
    target = entry / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    if lfs:
        etag = hashlib.sha256(body).hexdigest()
    else:
        etag = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
    meta = entry / ".cache" / "huggingface" / "download" / f"{relpath}.metadata"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(f"{revision}\n{etag}\n1792098587.5\n")
    return hashlib.sha256(body).hexdigest()


def make_snapshot(root: Path, revision: str = MOSS_REVISION) -> Path:
    """The real repo's load-bearing layout (assets/ excluded: never read)."""
    entry = root / "MOSS-Image-SFT"
    _write(entry, "model_index.json", json.dumps({"_class_name": "MOSSImagePipeline"}).encode(), revision, lfs=False)
    _write(entry, "pipeline_moss_image.py", b"# MOSS image remote code\n", revision, lfs=False)
    _write(entry, "scheduler/scheduler_config.json", b'{"shift": 5.0}', revision, lfs=False)
    _write(entry, "scheduler/pipeline_moss_image.py", b"# MOSSImageScheduler\n", revision, lfs=False)
    transformer_cfg = {"dim": 3072, "num_layers": 42, "num_heads": 24, "ffn_dim": 8192}
    _write(entry, "transformer/config.json", json.dumps(transformer_cfg).encode(), revision, lfs=False)
    _write(entry, "transformer/pipeline_moss_image.py", b"# MOSSImageTransformer\n", revision, lfs=False)
    _write(entry, "transformer/diffusion_pytorch_model-00001-of-00003.safetensors", b"dit" * 1000, revision, lfs=True)
    _write(entry, "text_encoder/config.json", b'{"model_type": "qwen3_5"}', revision, lfs=False)
    _write(entry, "text_encoder/model-00001-of-00004.safetensors", b"te" * 500, revision, lfs=True)
    _write(entry, "tokenizer/tokenizer.json", b"{}", revision, lfs=False)
    _write(entry, "vae/config.json", b'{"_class_name": "AutoencoderKLFlux2"}', revision, lfs=False)
    _write(entry, "vae/diffusion_pytorch_model.safetensors", b"vae" * 100, revision, lfs=True)
    return entry


# --- registration / schemas / policy -----------------------------------------


def test_moss_image_sft_is_an_exact_registered_model_and_job_type():
    adapter_cls = registry.get_adapter_class(MODEL_ID)
    assert issubclass(adapter_cls, MossImageSftAdapter)
    assert adapter_cls.model_id == MODEL_ID
    assert JOB_TYPE_TO_MODEL[JOB_TYPE] == MODEL_ID
    assert JOB_TYPE_PARAMS[JOB_TYPE] is MossImageParams
    for alias in ("moss-image-base", "moss-image-sft-lora", "OpenMOSS-Team/MOSS-Image-SFT"):
        assert alias not in JOB_TYPE_TO_MODEL
        with pytest.raises(KeyError):
            registry.get_adapter_class(alias)


def test_image_policy_opens_only_the_exact_id():
    assert not is_disabled_still_image_model(MODEL_ID)
    assert not is_disabled_still_image_model("MOSS.Image.SFT")
    for denied in (
        "moss-image",
        "MOSS-IMAGE",
        "moss-image-base",
        "moss-image-sft-lora",
        "moss-image-sft-int8",
        "mossimage",
        "MOSSImageSFT",
        "OpenMOSS-Team/MOSS-Image-SFT",
        "OpenMOSS-Team/MOSS-Image-Base",
        "moss-image-edit",
    ):
        assert is_disabled_still_image_model(denied), denied


# --- params schema ------------------------------------------------------------


def test_schema_card_defaults_and_rejections():
    params = MossImageParams(prompt="a red fox")
    assert params.steps == 50
    assert params.guidance_scale == 3.0
    assert (params.width, params.height) == (1024, 1024)
    assert params.uncond_skip_layers is None
    assert params.negative_prompt == ""
    # Unknown keys fail closed (a benchmark row cannot smuggle a condition).
    with pytest.raises(ValidationError):
        MossImageParams(prompt="x", cfg_scale=3.0)
    with pytest.raises(ValidationError):
        MossImageParams(prompt="x", image="data:image/png;base64,AAAA")
    with pytest.raises(ValidationError):
        MossImageParams(prompt="x", true_cfg_scale=1.0)
    # Sides are multiples of 16 bounded by 2048 (the guide's 2K/4M-pixel top
    # end; 2048x2048 is exactly 4M and is allowed).
    with pytest.raises(ValidationError):
        MossImageParams(prompt="x", width=1000)
    with pytest.raises(ValidationError):
        MossImageParams(prompt="x", width=2064)
    MossImageParams(prompt="x", width=2048, height=2048)  # exactly 4M: fine
    MossImageParams(prompt="x", width=2048, height=1024)  # 2M pixels: fine
    with pytest.raises(ValidationError):
        MossImageParams(prompt="x", uncond_skip_layers=[42])
    with pytest.raises(ValidationError):
        MossImageParams(prompt="x", uncond_skip_layers="22")


def test_parse_request_defaults_and_card_sampling():
    request = parse_request({"prompt": "a red fox", "_job_type": JOB_TYPE})
    assert request.steps == 50
    assert request.guidance_scale == 3.0
    assert request.uncond_skip_layers == (22,)
    assert request.seed == 42
    request = parse_request(
        {
            "prompt": "a red fox",
            "negative_prompt": "worst quality",
            "seed": 7,
            "width": 1280,
            "height": 960,
            "steps": 60,
            "guidance_scale": 4.5,
            "uncond_skip_layers": [22, 24],
        }
    )
    assert (request.width, request.height) == (1280, 960)
    assert request.uncond_skip_layers == (22, 24)
    # [] explicitly disables skip-layer guidance.
    assert parse_request({"prompt": "x", "uncond_skip_layers": []}).uncond_skip_layers == ()
    # Conditions this comparison adapter must never accept silently.
    for bad in (
        {"image": "data:image/png;base64,AAAA"},
        {"image_file": "/mnt/arbiter-store/inbox/x.png"},
        {"cfg_scale": 3.0},
        {"true_cfg_scale": 1.0},
        {"slg_scale": 3.0},
        {"num_images_per_prompt": 2},
    ):
        with pytest.raises(InferenceError):
            parse_request({"prompt": "x", **bad})
    with pytest.raises(InferenceError):
        parse_request({"prompt": "x", "width": 1024, "height": 4096})
    with pytest.raises(InferenceError):
        parse_request({"prompt": "x", "width": 1000})
    with pytest.raises(InferenceError):
        parse_request({"prompt": "x", "uncond_skip_layers": [99]})
    with pytest.raises(InferenceError):
        parse_request({"prompt": "x", "uncond_skip_layers": 22})


# --- fail-closed snapshot verification ----------------------------------------


def test_verify_snapshot_happy_path(tmp_path):
    entry = make_snapshot(tmp_path)
    provenance = verify_moss_snapshot(entry)
    assert provenance["hf_repo"] == "OpenMOSS-Team/MOSS-Image-SFT"
    assert provenance["revision"] == MOSS_REVISION
    assert provenance["verified_files"] >= 12


def test_verify_snapshot_rejects_wrong_revision_and_tampering(tmp_path):
    entry = make_snapshot(tmp_path, revision="0" * 40)
    with pytest.raises(LoadError, match="revision"):
        verify_moss_snapshot(entry)
    entry = make_snapshot(tmp_path)
    (entry / "pipeline_moss_image.py").write_bytes(b"# tampered remote code\n")
    with pytest.raises(LoadError, match="hash"):
        verify_moss_snapshot(entry)
    entry = make_snapshot(tmp_path)
    (entry / "text_encoder/model-00001-of-00004.safetensors").write_bytes(b"tampered" * 100)
    with pytest.raises(LoadError, match="hash"):
        verify_moss_snapshot(entry)


def test_verify_snapshot_rejects_missing_paths_and_symlinks(tmp_path):
    entry = make_snapshot(tmp_path)
    (entry / "vae").rename(tmp_path / "vae-moved")
    with pytest.raises(LoadError, match="missing required path"):
        verify_moss_snapshot(entry)
    outside = tmp_path / "outside"
    make_snapshot(outside)
    link = tmp_path / "link"
    link.symlink_to(outside / "MOSS-Image-SFT", target_is_directory=True)
    with pytest.raises(LoadError, match="real directory"):
        verify_moss_snapshot(link)


def test_default_weights_dir_is_the_spark_nvme_snapshot():
    assert str(DEFAULT_WEIGHTS_DIR) == "/home/darren/nvme-models/moss-image/MOSS-Image-SFT"
