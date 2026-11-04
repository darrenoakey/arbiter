"""Offline tests for the supported Comfy P-H3 adapter.

These use tiny local files and lightweight local execution doubles only for
our adapter's own runner boundary; no ComfyUI, torch, weights, network or GPU
is imported/executed. They are not evidence of a real P-H3 render.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from arbiter.adapters import registry
from arbiter.adapters.base import InferenceError, LoadError
from arbiter.adapters import p_h3_image_comfy as ph3
from arbiter.image_policy import is_disabled_still_image_model
from arbiter.schemas import JOB_TYPE_PARAMS, JOB_TYPE_TO_MODEL, PH3ImageComfyParams


def test_exact_registration_policy_and_schema():
    assert issubclass(registry.get_adapter_class("p-h3-image-comfy-w4a8"), ph3.PH3ImageComfyAdapter)
    assert JOB_TYPE_TO_MODEL["p-h3-image-comfy"] == "p-h3-image-comfy-w4a8"
    assert JOB_TYPE_PARAMS["p-h3-image-comfy"] is PH3ImageComfyParams
    assert not is_disabled_still_image_model("p-h3-image-comfy-w4a8")
    for alias in ("p-h3-image-comfy", "p-h3-image-comfy-bf16", "p-h3-image-comfy-w4a8-upsampler", "minimax-h3-image"):
        assert is_disabled_still_image_model(alias)
    p = PH3ImageComfyParams(prompt="botanist")
    assert (p.width, p.height, p.steps, p.prompt_upsampling) == (1024, 1024, 8, False)
    with pytest.raises(ValidationError):
        PH3ImageComfyParams(prompt="x", steps=9)
    with pytest.raises(ValidationError):
        PH3ImageComfyParams(prompt="x", negative_prompt="x")


def test_parse_request_like_for_like_native_and_fail_closed():
    r = ph3.parse_request({"prompt": "botanist", "seed": 42001})
    assert (r.width, r.height, r.seed) == (1024, 1024, 42001)
    assert ph3.parse_request({"prompt": "x", "width": 2912, "height": 2912}).width == 2912
    for bad in ({"steps": 9}, {"prompt_upsampling": True}, {"guidance_scale": 1}, {"image": "AAAA"}, {"width": 1000}, {"height": 4097}, {"seed": True}):
        with pytest.raises(InferenceError):
            ph3.parse_request({"prompt": "x", **bad})


def test_graph_uses_official_ph3_components_and_fixed_seed_grid(tmp_path):
    g = ph3.build_graph(ph3.PH3Request("a portrait", 42001, 1024, 1024), tmp_path)
    assert g["1"]["inputs"]["unet_name"] == "p-h3-image_w4a8.safetensors"
    assert g["2"]["inputs"] == {"clip_name": "p-h3-image_text_encoder_w4a8.safetensors", "type": "minimax", "device": "default"}
    assert g["3"]["inputs"]["vae_name"] == "p-h3-image_vae_int8_convrot.safetensors"
    assert g["4"]["inputs"]["shift_video"] == g["4"]["inputs"]["shift_audio"] == 1.0
    assert g["5"]["class_type"] == "PH3ImageConditioning"
    assert g["6"]["inputs"]["noise_seed"] == 42001
    assert g["7"]["inputs"]["sigmas"] == ph3.SIGMA_STRING
    assert g["10"]["inputs"]["latent_image"] == ["5", 1]
    assert g["12"]["inputs"]["images"] == ["11", 0]
    assert "upsampler" not in json.dumps(g).lower()


def _weights(root: Path, monkeypatch):
    for name in ph3.WEIGHTS:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        body = name.encode()
        p.write_bytes(body)
        monkeypatch.setitem(ph3.WEIGHTS[name], "size", len(body))
        monkeypatch.setitem(ph3.WEIGHTS[name], "lfs", hashlib.sha256(body).hexdigest())
    return root


def test_verify_real_weights_and_reject_tampering(tmp_path, monkeypatch):
    root = _weights(tmp_path / "models", monkeypatch)
    info = ph3.verify_weights(root)
    assert info["revision"] == ph3.PH3_REVISION
    assert info["weight_bytes"] == sum(len(n.encode()) for n in ph3.WEIGHTS)
    assert info["precision"] == {"transformer": "w4a8", "text_encoder": "w4a8", "vae": "int8_convrot"}
    path = root / "vae/p-h3-image_vae_int8_convrot.safetensors"
    path.write_bytes(b"wrong")
    with pytest.raises(LoadError, match="size mismatch"):
        ph3.verify_weights(root)


def test_verify_source_revision_and_hashes(tmp_path, monkeypatch):
    root = tmp_path / "ComfyUI"
    (root / ".git").mkdir(parents=True)
    (root / ".git/HEAD").write_text(ph3.COMFY_REVISION + "\n")
    for name in ph3.EXPECTED_COMFY_GIT_BLOBS:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        body = name.encode()
        p.write_bytes(body)
        monkeypatch.setitem(ph3.EXPECTED_COMFY_GIT_BLOBS, name, hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest())
    info = ph3.verify_comfy_source(root)
    assert info["revision"] == ph3.COMFY_REVISION
    (root / "comfy/sd.py").write_bytes(b"tampered")
    with pytest.raises(LoadError, match="source changed"):
        ph3.verify_comfy_source(root)
    (root / ".git/HEAD").write_text("0" * 40)
    with pytest.raises(LoadError, match="requires ComfyUI"):
        ph3.verify_comfy_source(root)


class _Cuda:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def synchronize(): pass
    @staticmethod
    def reset_peak_memory_stats(): pass
    @staticmethod
    def max_memory_allocated(): return 12_345
    @staticmethod
    def max_memory_reserved(): return 23_456


class _Executor:
    def __init__(self, server):
        self.success = True
        self.status_messages = []
        self.server = server

    def execute(self, graph, prompt_id, extra, outputs):
        self.graph = graph
        prefix = graph["12"]["inputs"]["filename_prefix"]
        output_dir = Path(extra["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{prefix}_00001_.png").write_bytes(b"test png")


def test_runtime_paths_and_builtin_initialization(tmp_path, monkeypatch):
    root = tmp_path / "ComfyUI"
    (root / ".git").mkdir(parents=True)
    (root / ".git/HEAD").write_text(ph3.COMFY_REVISION + "\n")
    for name in ph3.EXPECTED_COMFY_GIT_BLOBS:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        body = name.encode()
        p.write_bytes(body)
        monkeypatch.setitem(ph3.EXPECTED_COMFY_GIT_BLOBS, name, hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest())
    paths = SimpleNamespace(add_model_folder_path=lambda *args: None)
    nodes = SimpleNamespace(init_extra_nodes=lambda **kwargs: None)
    class Asyncio:
        @staticmethod
        def run(value):
            assert value is None
    monkeypatch.setitem(sys.modules, "folder_paths", paths)
    monkeypatch.setitem(sys.modules, "nodes", nodes)
    monkeypatch.setitem(sys.modules, "asyncio", Asyncio)
    monkeypatch.setitem(sys.modules, "execution", SimpleNamespace(PromptExecutor=_Executor))
    monkeypatch.setattr(ph3, "verify_weights", lambda p: {})
    monkeypatch.setenv("ARBITER_MEMORY_GB", "28")
    adapter = ph3.PH3ImageComfyAdapter(models_dir=tmp_path / "models", runtime_dir=root)
    adapter.load()
    adapter._executor.output_dir = tmp_path / "comfy-output"
    assert adapter._provenance["runtime"]["revision"] == ph3.COMFY_REVISION


def test_output_path_is_set_per_job(tmp_path, monkeypatch):
    output_calls = []
    paths = SimpleNamespace(set_output_directory=output_calls.append)
    monkeypatch.setitem(sys.modules, "folder_paths", paths)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_Cuda))
    adapter = ph3.PH3ImageComfyAdapter(models_dir=tmp_path, executor_factory=_Executor)
    monkeypatch.setattr(ph3, "verify_weights", lambda p: {"repo": ph3.PH3_REPO, "revision": ph3.PH3_REVISION})
    monkeypatch.setattr(ph3, "verify_comfy_source", lambda p: {"revision": ph3.COMFY_REVISION})
    monkeypatch.setenv("ARBITER_MEMORY_GB", "28")
    adapter.load()
    adapter._executor.output_dir = tmp_path
    result = adapter.infer({"_job_type": "p-h3-image-comfy", "prompt": "botanist", "seed": 42001}, tmp_path, threading.Event())
    assert (tmp_path / "result.png").read_bytes() == b"test png"
    assert result["comparison"]["like_for_like_house_1mp"] is True
    assert result["prompt_upsampling"] is False
    assert result["timings"]["render_index"] == 1
    assert result["memory"]["peak_reserved_bytes"] == 23_456
    assert result["provenance"]["weights"]["revision"] == ph3.PH3_REVISION
    assert output_calls == []  # custom boundary does not mutate global Comfy paths
    adapter.unload()


def test_runner_failure_does_not_claim_image(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_Cuda))
    adapter = ph3.PH3ImageComfyAdapter(models_dir=tmp_path, executor_factory=_Executor)
    monkeypatch.setattr(ph3, "verify_weights", lambda p: {})
    monkeypatch.setattr(ph3, "verify_comfy_source", lambda p: {"revision": ph3.COMFY_REVISION})
    monkeypatch.setenv("ARBITER_MEMORY_GB", "28")
    adapter.load()
    adapter._executor.success = False
    adapter._executor.status_messages = ["local error"]
    with pytest.raises(InferenceError, match="local error"):
        adapter.infer({"_job_type": "p-h3-image-comfy", "prompt": "x"}, tmp_path, threading.Event())
    assert not (tmp_path / "result.png").exists()
