"""Tests for the FastH3 adapter's output boundary and trained sigma ladder."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import arbiter.adapters  # noqa: F401 - imports the complete built-in registry
from arbiter.adapters.minimax_fast_h3 import MinimaxFastH3Adapter
from arbiter.adapters.minimax_h3_local import MinimaxH3LocalAdapter
from arbiter.adapters.registry import list_registered


def test_fasth3_sigma_ladder_is_four_trained_jumps() -> None:
    from arbiter.adapters import minimax_fast_h3 as fast

    video = fast.fasth3_video_sigmas()
    audio = fast.fasth3_audio_sigmas()
    assert video == [0.999, 0.749, 0.500, 0.250, 0.0]
    assert audio == video
    assert video[-1] == 0.0
    assert all(earlier > later for earlier, later in zip(video, video[1:]))
    assert fast.fasth3_steps({"num_inference_steps": 50}) == 4
    assert fast.fasth3_steps({}) == 4


def test_fasth3_source_uses_nvfp4_and_trained_ladder() -> None:
    from arbiter.adapters import minimax_fast_h3 as fast

    src = inspect.getsource(fast)
    assert "NVFP4WeightOnlyConfig" in inspect.getsource(
        __import__("arbiter.adapters.minimax_h3_local", fromlist=["_nvfp4_config"])
    )
    assert "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2" in src
    assert "_install_trained_sigma_ladder" in src
    assert "Int8WeightOnlyConfig" not in src
    assert "enable_group_offload" not in src
    assert fast.FASTH3_STEPS == 4


def test_fasth3_and_h3_adapters_are_both_registered() -> None:
    registered = list_registered()
    assert "minimax-h3" in registered
    assert "minimax-h3-local" in registered
    assert "minimax-fast-h3" in registered
    assert MinimaxFastH3Adapter.model_id == "minimax-fast-h3"
    assert MinimaxH3LocalAdapter.model_id == "minimax-h3-local"
    assert MinimaxFastH3Adapter.model_id != MinimaxH3LocalAdapter.model_id


def test_fasth3_deploy_config_is_discoverable() -> None:
    root = Path(__file__).parents[3]
    config = json.loads((root / "config/spark/minimax-fast-h3.model.json").read_text())
    assert config["max_concurrent"] == 1
    assert config["max_instances"] == 1
    assert config["memory_gb"] >= 35
    assert config["load_ms"] >= 60_000
    assert config["worker_cmd"][3] == "minimax-fast-h3"
    assert config["worker_cmd"][0].endswith("venvs/minimax-h3/bin/python")


class _SavableModel:
    """Stand-in for a quantized HF model: save_pretrained writes files."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.saved_to = []

    def save_pretrained(self, dest):
        if self.fail:
            raise OSError("disk full")
        path = Path(dest)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_bytes(b"weights")
        self.saved_to.append(path)


def test_fasth3_snapshot_ready_requires_transformer_and_shared_text_encoder(
    tmp_path, monkeypatch
) -> None:
    from arbiter.adapters import minimax_fast_h3 as fast
    from arbiter.adapters import minimax_h3_local as h3

    transformer_root = tmp_path / "fast"
    text_root = tmp_path / "h3"
    monkeypatch.setattr(fast, "FASTH3_SNAPSHOT_DIR", transformer_root)
    monkeypatch.setattr(h3, "H3_SNAPSHOT_DIR", text_root)
    monkeypatch.setattr(fast, "H3_SNAPSHOT_DIR", text_root)
    assert not fast._fasth3_snapshot_ready()

    (transformer_root / "transformer").mkdir(parents=True)
    assert not fast._fasth3_snapshot_ready()

    (text_root / "text_encoder").mkdir(parents=True)
    assert fast._fasth3_snapshot_ready()


def test_fasth3_snapshot_writes_transformer_only(tmp_path, monkeypatch) -> None:
    from arbiter.adapters import minimax_fast_h3 as fast

    monkeypatch.setattr(fast, "FASTH3_SNAPSHOT_DIR", tmp_path)
    adapter = MinimaxFastH3Adapter()
    transformer, text_encoder = _SavableModel(), _SavableModel()

    adapter._write_nvfp4_snapshot(transformer, text_encoder)
    assert (tmp_path / "transformer" / "model.safetensors").read_bytes() == b"weights"
    assert not (tmp_path / "text_encoder").exists()
    assert not (tmp_path / "transformer.tmp").exists()

    fresh = _SavableModel()
    adapter._write_nvfp4_snapshot(fresh, fresh)
    assert fresh.saved_to == []


def test_fasth3_snapshot_failure_does_not_raise(tmp_path, monkeypatch) -> None:
    from arbiter.adapters import minimax_fast_h3 as fast

    monkeypatch.setattr(fast, "FASTH3_SNAPSHOT_DIR", tmp_path)
    adapter = MinimaxFastH3Adapter()
    adapter._write_nvfp4_snapshot(_SavableModel(fail=True), _SavableModel())
    assert not (tmp_path / "transformer").exists()
