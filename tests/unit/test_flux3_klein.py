"""FLUX 3 Image Klein benchmark adapters: registration, policy, schema and
fail-closed weight verification. No GPU, no torch: every check here runs
before the adapter imports a framework. Weight verification runs against a
real on-disk snapshot layout with real content hashes."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from arbiter.adapters import registry
from arbiter.adapters.base import InferenceError, LoadError
from arbiter.adapters.flux3_klein import (
    DEFAULT_STEPS,
    FLUX3_ENTRY,
    FLUX3_REVISION,
    VARIANTS,
    Flux3KleinAdapter,
    parse_request,
    resolve_weights_root,
    source_provenance,
    verify_snapshot_file,
    verify_weights,
)
from arbiter.image_policy import is_disabled_still_image_model
from arbiter.schemas import JOB_TYPE_PARAMS, JOB_TYPE_TO_MODEL, Flux3KleinParams

EXACT_IDS = ("flux3-klein-bf16", "flux3-klein-fp8", "flux3-klein-nvfp4")
ROOT = Path(__file__).resolve().parents[2]


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


def _dit(entry: Path, prefix: str, quantization: str | None, revision: str, counts: dict) -> None:
    weights = f"weights-{quantization}".encode() * 64
    config = {
        "format_version": 1,
        "model": FLUX3_ENTRY,
        "quantization": quantization,
        "dit": {"flow_map": True},
        "sampler": {"steps": 16, "guidance": None, "alpha": 6.93, "size": [1024, 1024]},
    }
    config_bytes = json.dumps(config).encode()
    weights_sha = _write(entry, f"{prefix}model.safetensors", weights, revision, lfs=True)
    config_sha = _write(entry, f"{prefix}config.json", config_bytes, revision, lfs=False)
    manifest = {
        "kind": "f3_dit_export",
        "model": FLUX3_ENTRY,
        "quantization": quantization,
        "dtype": "bfloat16" if quantization is None else quantization,
        "sha256": {"config.json": config_sha, "model.safetensors": weights_sha},
        **counts,
    }
    _write(entry, f"{prefix}manifest.json", json.dumps(manifest).encode(), revision, lfs=False)


def make_snapshot(root: Path, revision: str = FLUX3_REVISION) -> Path:
    entry = root / FLUX3_ENTRY
    _dit(entry, "", None, revision, {})
    _dit(entry, "variants/fp8r/", "fp8r", revision, {"fp8_matrices": 234})
    _dit(entry, "variants/nvfp4/", "nvfp4", revision, {"nvfp4_matrices": 198})
    _write(entry, "ae.safetensors", b"autoencoder" * 100, revision, lfs=True)
    _write(entry, "text_encoder/config.json", b'{"model_type": "qwen3_vl"}', revision, lfs=False)
    _write(entry, "text_encoder/model-00001-of-00002.safetensors", b"te" * 500, revision, lfs=True)
    return entry


# --- registration / schemas / policy -----------------------------------------


def test_each_precision_is_an_exact_registered_model_and_job_type():
    for model_id in EXACT_IDS:
        adapter_cls = registry.get_adapter_class(model_id)
        assert issubclass(adapter_cls, Flux3KleinAdapter)
        assert adapter_cls.model_id == model_id == adapter_cls.variant.model_id
        assert JOB_TYPE_TO_MODEL[model_id] == model_id
        assert JOB_TYPE_PARAMS[model_id] is Flux3KleinParams
    assert {VARIANTS[m].quantization for m in EXACT_IDS} == {None, "fp8r", "nvfp4"}
    for alias in ("flux3", "flux3-klein", "flux-3-image-klein", "flux3-klein-base", "flux3-klein-int8"):
        assert alias not in JOB_TYPE_TO_MODEL
        with pytest.raises(KeyError):
            registry.get_adapter_class(alias)


def test_image_policy_opens_only_the_three_exact_ids():
    for model_id in EXACT_IDS:
        assert not is_disabled_still_image_model(model_id)
    for denied in (
        "flux3",
        "flux3-klein",
        "flux3-klein-bf16-lora",
        "flux3-klein-fp8-lora",
        "flux3-klein-base",
        "black-forest-labs/flux-3-image-klein",
        "flux-3-image-klein",
        "flux3_klein_nvfp4_v2",
        "flux2",
        "flux-schnell",
    ):
        assert is_disabled_still_image_model(denied), denied


def test_params_schema_defaults_and_rejects_unsupported_conditions():
    params = Flux3KleinParams(prompt="a red fox")
    assert (params.steps, params.width, params.height, params.seed) == (16, 1024, 1024, 42)
    for bad in (
        {"prompt": "x", "guidance": 3.5},
        {"prompt": "x", "pup": True},
        {"prompt": "x", "image_file": "/tmp/a.png"},
        {"prompt": "x", "width": 1000},
        {"prompt": "x", "steps": 0},
        {"prompt": ""},
    ):
        with pytest.raises(ValidationError):
            Flux3KleinParams(**bad)


def test_parse_request_is_fail_closed():
    request = parse_request({"prompt": "a lighthouse", "seed": 7})
    assert (request.steps, request.width, request.height, request.seed) == (DEFAULT_STEPS, 1024, 1024, 7)
    for params, match in (
        ({}, "prompt is required"),
        ({"prompt": "  "}, "prompt is required"),
        ({"prompt": "x", "guidance": 3.5}, "guidance"),
        ({"prompt": "x", "negative_prompt": "blur"}, "negative_prompt"),
        ({"prompt": "x", "pup": True}, "pup"),
        ({"prompt": "x", "image_file": "/tmp/a.png"}, "image_file"),
        ({"prompt": "x", "width": 1000}, "multiples of 16"),
        ({"prompt": "x", "steps": "16"}, "steps must be an integer"),
        ({"prompt": "x", "seed": True}, "seed must be an integer"),
        ({"prompt": "x", "seed": -1}, "seed must be between"),
    ):
        with pytest.raises(InferenceError, match=match):
            parse_request(params)


def test_infer_rejects_another_job_type_before_touching_the_gpu(tmp_path):
    adapter = registry.get_adapter_class("flux3-klein-fp8")()
    with pytest.raises(InferenceError, match="serves only"):
        adapter.infer({"prompt": "x", "_job_type": "flux3-klein-bf16"}, tmp_path, threading.Event())
    with pytest.raises(InferenceError, match="not loaded"):
        adapter.infer({"prompt": "x", "_job_type": "flux3-klein-fp8"}, tmp_path, threading.Event())


def test_load_refuses_without_the_arbiter_cuda_cap(monkeypatch):
    monkeypatch.delenv("ARBITER_MEMORY_GB", raising=False)
    adapter = registry.get_adapter_class("flux3-klein-bf16")()
    with pytest.raises(LoadError, match="ARBITER_MEMORY_GB"):
        adapter.load("cuda")
    with pytest.raises(LoadError, match="requires cuda"):
        adapter.load("cpu")


def test_reference_package_must_be_the_pinned_install():
    # The gate venv does not carry flux3; an absent or editable install must
    # never load. (The spark venvs/flux3 records the pinned git commit.)
    with pytest.raises(LoadError, match="not installed|direct_url|expected commit"):
        source_provenance()


# --- weights root and snapshot verification ----------------------------------


def test_weights_root_must_be_absolute_canonical_real_directory(tmp_path):
    assert resolve_weights_root({"FLUX3_WEIGHTS_ROOT": str(tmp_path)}) == tmp_path
    link = tmp_path.parent / f"{tmp_path.name}-link"
    link.symlink_to(tmp_path)
    for value in ("relative/flux3", f"{tmp_path}/../{tmp_path.name}", str(link), str(tmp_path / "missing"), ""):
        with pytest.raises(LoadError):
            resolve_weights_root({"FLUX3_WEIGHTS_ROOT": value})


def test_verify_weights_accepts_each_exact_variant(tmp_path):
    entry = make_snapshot(tmp_path)
    for model_id, expected in (("flux3-klein-bf16", None), ("flux3-klein-fp8", 234), ("flux3-klein-nvfp4", 198)):
        info = verify_weights(entry, VARIANTS[model_id])
        assert info["revision"] == FLUX3_REVISION
        assert info["quantization"] == VARIANTS[model_id].quantization
        assert info["expected_quantized_linears"] == expected
        assert info["verified_files"] == 6


def test_verify_weights_rejects_a_mislabelled_variant(tmp_path):
    entry = make_snapshot(tmp_path)
    # Swap the fp8 DiT into the nvfp4 slot: the manifest quantization no
    # longer matches, so nvfp4 must refuse rather than load FP8 weights.
    nvfp4 = entry / "variants" / "nvfp4"
    for name in ("manifest.json", "config.json", "model.safetensors"):
        (nvfp4 / name).write_bytes((entry / "variants" / "fp8r" / name).read_bytes())
    with pytest.raises(LoadError):
        verify_weights(entry, VARIANTS["flux3-klein-nvfp4"])


def test_verify_weights_rejects_tampered_or_wrong_revision_files(tmp_path):
    entry = make_snapshot(tmp_path)
    (entry / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(LoadError, match="content hash"):
        verify_weights(entry, VARIANTS["flux3-klein-bf16"])

    other = make_snapshot(tmp_path / "other", revision="0" * 40)
    with pytest.raises(LoadError, match="revision"):
        verify_weights(other, VARIANTS["flux3-klein-fp8"])

    fresh = make_snapshot(tmp_path / "fresh")
    (fresh / ".cache/huggingface/download/ae.safetensors.metadata").unlink()
    with pytest.raises(LoadError, match="metadata missing"):
        verify_snapshot_file(fresh, "ae.safetensors", FLUX3_REVISION)
    (fresh / "text_encoder/config.json").unlink()
    (fresh / "text_encoder/config.json").symlink_to(fresh / "ae.safetensors")
    with pytest.raises(LoadError):
        verify_weights(fresh, VARIANTS["flux3-klein-bf16"])


# --- spark config ---------------------------------------------------------------


@pytest.mark.parametrize(("model_id", "memory_gb"), [("flux3-klein-bf16", 36), ("flux3-klein-fp8", 32), ("flux3-klein-nvfp4", 32)])
def test_spark_model_json_pins_exact_worker_and_weights_root(model_id, memory_gb):
    config = json.loads((ROOT / "config" / "spark" / f"{model_id}.model.json").read_text())
    assert config["memory_gb"] == memory_gb
    assert config["worker_cmd"] == [
        "/home/darren/src/arbiter/venvs/flux3/bin/python",
        "-m",
        "arbiter.worker_main",
        model_id,
    ]
    assert config["adapter_params"] == {
        "FLUX3_WEIGHTS_ROOT": "/home/darren/nvme-models/flux3",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    assert "auto_download" not in config
    assert config["max_concurrent"] == 1 and config["max_instances"] == 1
    assert config["placements"] == ["spark"]
