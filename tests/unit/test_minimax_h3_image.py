"""MiniMax-H3-Image (2026-11-10 still image) adapter: fail-closed validation,
licence gate, lane/canvas protocol and snapshot verification. No GPU, no
torch, no model run: every check runs before the adapter imports a framework.
Snapshot verification uses a real on-disk layout with real content hashes,
mirroring tests/unit/test_moss_image.py. These are unit/static tests only —
they are not model runs and prove no GPU route."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from arbiter.adapters import registry
from arbiter.adapters.base import InferenceError, LoadError
from arbiter.adapters.minimax_h3_image import (
    DEFAULT_MIN_GENERATION_AREA,
    DIFFUSERS_COMMIT,
    EXPECTED_SAMPLE_SIGMAS,
    EXPECTED_STEPS,
    MINIMAX_H3_IMAGE_REVISION,
    MiniMaxH3ImageAdapter,
    expected_canvas_size,
    parse_request,
    require_license_gate,
    verify_h3_image_snapshot,
)

LICENSED = {"license_confirmed": True, "license_confirmation_ref": "legal-determination-2026-11-10"}


def test_exact_registration_policy_and_schema():
    from arbiter.image_policy import is_disabled_still_image_model
    from arbiter.schemas import JOB_TYPE_PARAMS, JOB_TYPE_TO_MODEL, MiniMaxH3ImageParams

    assert issubclass(registry.get_adapter_class("minimax-h3-image"), MiniMaxH3ImageAdapter)
    assert JOB_TYPE_TO_MODEL["minimax-h3-image"] == "minimax-h3-image"
    assert JOB_TYPE_PARAMS["minimax-h3-image"] is MiniMaxH3ImageParams
    assert not is_disabled_still_image_model("minimax-h3-image")
    assert is_disabled_still_image_model("minimax-h3-image-bf16")
    p = MiniMaxH3ImageParams(prompt="botanist", **LICENSED)
    assert (p.lane, p.width, p.height, p.guidance_scale) == ("controlled-1024", None, None, 1)
    with pytest.raises(ValidationError):
        MiniMaxH3ImageParams(prompt="botanist")
    with pytest.raises(ValidationError):
        MiniMaxH3ImageParams(prompt="botanist", steps=6, **LICENSED)


def test_staged_lifecycle_does_not_use_futile_unified_memory_offload():
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(MiniMaxH3ImageAdapter._encode_and_swap))
    calls = [n.func.attr for n in ast.walk(ast.parse(source))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert calls.index("unload_components") < calls.index("load_components")
    assert "MiniMaxH3ImageTextEncoderStep" in source
    assert "state = PipelineState()" in source
    assert "remaining" in source
    assert "state.set(\"prompt_embeds\", state.get(\"prompt_embeds\").cpu())" not in source
    adapter_source = inspect.getsource(MiniMaxH3ImageAdapter)
    assert "enable_model_cpu_offload" not in adapter_source
    assert "enable_sequential_cpu_offload" not in adapter_source


def _write(entry: Path, relpath: str, body: bytes, revision: str, lfs: bool) -> None:
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


def make_snapshot(root: Path, revision: str = MINIMAX_H3_IMAGE_REVISION) -> Path:
    entry = root / "MiniMax-H3-Image"
    _write(entry, "LICENSE", b"MiniMax-H3-Image licence fixture\n", revision, lfs=False)
    _write(entry, "model_index.json", json.dumps({"_class_name": "MiniMaxH3ImagePipeline"}).encode(), revision, lfs=False)
    modular = {
        "_class_name": "MiniMaxH3ImageModularPipeline",
        "sample_sigmas": list(EXPECTED_SAMPLE_SIGMAS),
        "transformer": ["diffusers", "MiniMaxH3ImageTransformer2DModel", {"subfolder": "transformer_diffusers"}],
        "vae": ["diffusers", "AutoencoderKLMiniMaxH3Image", {"subfolder": "vae_diffusers"}],
    }
    _write(entry, "modular_model_index.json", json.dumps(modular).encode(), revision, lfs=False)
    _write(entry, "scheduler/scheduler_config.json", b'{"_class_name": "MiniMaxH3Scheduler"}', revision, lfs=False)
    _write(entry, "text_encoder/config.json", b'{"model_type": "qwen3_vl"}', revision, lfs=False)
    _write(entry, "text_encoder/model-00001-of-00014.safetensors", b"te" * 500, revision, lfs=True)
    _write(entry, "tokenizer/tokenizer.json", b"{}", revision, lfs=False)
    _write(entry, "processor/preprocessor_config.json", b"{}", revision, lfs=False)
    _write(entry, "transformer_diffusers/config.json", b'{"_class_name": "MiniMaxH3ImageTransformer2DModel"}', revision, lfs=False)
    _write(entry, "transformer_diffusers/diffusion_pytorch_model-00001-of-00012.safetensors", b"dit" * 1000, revision, lfs=True)
    _write(entry, "vae_diffusers/config.json", b'{"_class_name": "AutoencoderKLMiniMaxH3Image"}', revision, lfs=False)
    _write(entry, "vae_diffusers/diffusion_pytorch_model.safetensors", b"vae" * 100, revision, lfs=True)
    return entry


# --- lanes, canvas trap, sampling protocol ------------------------------------


def test_canvas_trap_and_lanes():
    # The runtime default would render native-1024 requests at 2048 and
    # LANCZOS-resize back; the controlled lane disables it with 1024.
    assert expected_canvas_size(1024, 1024, DEFAULT_MIN_GENERATION_AREA) == (2048, 2048)
    assert expected_canvas_size(1024, 1024, 1024) == (1024, 1024)
    assert expected_canvas_size(4096, 4096, DEFAULT_MIN_GENERATION_AREA) == (4096, 4096)
    controlled = parse_request({"prompt": "a red fox", **LICENSED})
    assert (controlled.lane, controlled.width, controlled.height) == ("controlled-1024", 1024, 1024)
    assert controlled.min_generation_area == 1024
    recommended = parse_request({"prompt": "a red fox", "lane": "recommended-4096", **LICENSED})
    assert (recommended.width, recommended.height, recommended.min_generation_area) == (4096, 4096, 4194304)
    assert EXPECTED_STEPS == 6
    # Any other size or caller control of the trap mislabels the row: fail.
    for bad in (
        {"width": 2048},
        {"lane": "recommended-4096", "width": 1024},
        {"lane": "native-2048"},
        {"min_generation_area": 1024},
        {"steps": 6},
        {"num_inference_steps": 6},
        {"sigmas": [1.0, 0.0]},
    ):
        with pytest.raises(InferenceError):
            parse_request({"prompt": "x", **LICENSED, **bad})


def test_distilled_protocol_rejections():
    for bad in (
        {"negative_prompt": "blurry"},
        {"guidance_scale": 3.0},
        {"cfg_scale": 1.0},
        {"true_cfg_scale": 1.0},
        {"image": "data:image/png;base64,AAAA"},
        {"prompt_enhancement": True},
        {"revised_prompt": "enhanced"},
        {"api_key": "secret"},
        {"cloud": True},
        {"hosted_service": True},
    ):
        with pytest.raises(InferenceError):
            parse_request({"prompt": "x", **LICENSED, **bad})
    # Guidance omitted or exactly 1 is the only accepted form.
    assert parse_request({"prompt": "x", "guidance_scale": 1, **LICENSED}).seed == 42
    with pytest.raises(InferenceError):
        parse_request({"prompt": "x", "seed": -1, **LICENSED})
    with pytest.raises(InferenceError):
        parse_request({"prompt": "  ", **LICENSED})


# --- licence gate ---------------------------------------------------------------


def test_license_gate_fails_closed():
    with pytest.raises(InferenceError, match="licence gate"):
        parse_request({"prompt": "x"})
    with pytest.raises(InferenceError, match="licence gate"):
        parse_request({"prompt": "x", "license_confirmed": True})
    with pytest.raises(InferenceError, match="licence gate"):
        require_license_gate({"license_confirmed": "true", "license_confirmation_ref": "determination-1"})
    with pytest.raises(InferenceError, match="licence gate"):
        require_license_gate({"license_confirmed": True, "license_confirmation_ref": "short"})
    assert require_license_gate(LICENSED) == "legal-determination-2026-11-10"


# --- snapshot verification --------------------------------------------------------


def test_verify_snapshot_happy_path(tmp_path):
    provenance = verify_h3_image_snapshot(make_snapshot(tmp_path))
    assert provenance["hf_repo"] == "MiniMaxAI/MiniMax-H3-Image"
    assert provenance["revision"] == MINIMAX_H3_IMAGE_REVISION
    assert provenance["verified_files"] >= 12


def test_verify_snapshot_rejects_old_or_quantized_or_tampered(tmp_path):
    entry = make_snapshot(tmp_path)
    (entry / "modular_model_index.json").unlink()
    with pytest.raises(LoadError):
        verify_h3_image_snapshot(entry)

    entry = make_snapshot(tmp_path)
    modular = json.loads((entry / "modular_model_index.json").read_text())
    modular["sample_sigmas"] = [1.0, 0.5, 0.0]
    (entry / "modular_model_index.json").write_text(json.dumps(modular))
    with pytest.raises(LoadError, match="sample_sigmas"):
        verify_h3_image_snapshot(entry)

    entry = make_snapshot(tmp_path)
    (entry / "transformer_diffusers/config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "nvfp4"}})
    )
    with pytest.raises(LoadError, match="NVFP4"):
        verify_h3_image_snapshot(entry)

    entry = make_snapshot(tmp_path, revision="0" * 40)
    with pytest.raises(LoadError, match="revision"):
        verify_h3_image_snapshot(entry)

    entry = make_snapshot(tmp_path)
    (entry / "vae_diffusers/diffusion_pytorch_model.safetensors").write_bytes(b"tampered" * 100)
    with pytest.raises(LoadError, match="hash"):
        verify_h3_image_snapshot(entry)


def test_verify_snapshot_rejects_symlink_and_missing(tmp_path):
    entry = make_snapshot(tmp_path)
    (entry / "vae_diffusers").rename(tmp_path / "vae-moved")
    with pytest.raises(LoadError, match="missing required path"):
        verify_h3_image_snapshot(entry)
    outside = make_snapshot(tmp_path / "outside-root")
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(LoadError, match="real directory"):
        verify_h3_image_snapshot(link)


def test_adapter_identity_and_pins():
    assert MiniMaxH3ImageAdapter.model_id == "minimax-h3-image"
    assert MINIMAX_H3_IMAGE_REVISION == "492fd8c4eca666f1085f57e30738c1e0d86b3439"
    assert DIFFUSERS_COMMIT == "f598e6e6a99528858b962c1399c45b59a6c25484"
    adapter = MiniMaxH3ImageAdapter()
    assert adapter.estimate_time({"lane": "recommended-4096"}) > adapter.estimate_time({})
