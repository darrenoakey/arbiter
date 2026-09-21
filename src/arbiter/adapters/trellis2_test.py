"""Co-located TRELLIS.2 adapter validation (no GPU, no model load)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arbiter.adapters.base import InferenceError
from arbiter.adapters.registry import list_registered
from arbiter.adapters.trellis2 import (
    DEFAULT_SNAPSHOT,
    HF_ID,
    Trellis2Adapter,
    clamp_decimation,
    clamp_resolution,
    clamp_steps,
    clamp_texture_size,
    pipeline_type_for_resolution,
    resolve_weights_path,
)


def test_trellis2_is_registered() -> None:
    assert "trellis2" in list_registered()
    assert Trellis2Adapter.model_id == "trellis2"


def test_pipeline_types_match_upstream_app() -> None:
    assert pipeline_type_for_resolution(512) == "512"
    assert pipeline_type_for_resolution(1024) == "1024_cascade"
    assert pipeline_type_for_resolution(1536) == "1536_cascade"


def test_bounds_match_upstream_app_defaults() -> None:
    assert clamp_steps(None) == 12
    assert clamp_steps(0) == 1
    assert clamp_steps(99) == 50
    assert clamp_texture_size(None) == 2048
    assert clamp_texture_size(3000) == 2048
    assert clamp_texture_size(4096) == 4096
    assert clamp_decimation(None) == 500_000
    assert clamp_decimation(10) == 100_000
    assert clamp_resolution(1024) == 1024


def test_default_snapshot_is_the_4b_directory() -> None:
    assert DEFAULT_SNAPSHOT == Path("/home/darren/local/models/trellis2/TRELLIS.2-4B")
    assert HF_ID == "microsoft/TRELLIS.2-4B"


def test_weights_require_existing_local_snapshot(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    snapshot = tmp_path / "TRELLIS.2-4B"
    with pytest.raises(InferenceError, match="local snapshot missing"):
        resolve_weights_path(config_path=config, snapshot_dir=snapshot)
    snapshot.mkdir()
    (snapshot / "pipeline.json").write_text("{}")
    assert resolve_weights_path(config_path=config, snapshot_dir=snapshot) == str(snapshot)
    config.write_text('[trellis2]\nweights_path = "/missing/trellis2"\n')
    with pytest.raises(InferenceError, match="configured snapshot is invalid"):
        resolve_weights_path(config_path=config, snapshot_dir=snapshot)
    config.write_text(f'[trellis2]\nweights_path = "{snapshot}"\n')
    assert resolve_weights_path(config_path=config, snapshot_dir=snapshot) == str(snapshot)


def test_source_selects_sdpa_without_attention_env() -> None:
    module = Path(__file__).with_name("trellis2.py").read_text()
    assert "set_backend" in module
    assert '"sdpa"' in module
    assert "extension_webp=False" in module
    assert "os.environ" not in module
    assert "ATTN_BACKEND=" not in module


def test_deploy_config_is_serialized_single_worker() -> None:
    root = Path(__file__).parents[3]
    config = json.loads((root / "config/spark/trellis2.model.json").read_text())
    assert config["max_concurrent"] == 1
    assert config["max_instances"] == 1
    assert config["memory_gb"] == 48
    assert config["worker_cmd"] == [
        "/home/darren/src/arbiter/venvs/trellis2/bin/python",
        "-m",
        "arbiter.worker_main",
        "trellis2",
    ]
    assert config["adapter_params"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    module = Path(__file__).with_name("trellis2.py").read_text()
    assert '"png"' in module
    assert "texture_encoding" in module
