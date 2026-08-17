"""Tests for the local MiniMax H3 adapter's output boundary."""

from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

import numpy as np

import arbiter.adapters  # noqa: F401 - imports the complete built-in registry
from arbiter.adapters.minimax_h3_local import MinimaxH3LocalAdapter
from arbiter.adapters.registry import list_registered


def test_extract_video_frames_unwraps_single_video_batch() -> None:
    """Diffusers' single-video batch becomes the frame sequence to encode."""
    videos = np.zeros((1, 2, 16, 24, 3), dtype=np.uint8)

    frames = MinimaxH3LocalAdapter._extract_video_frames(videos)

    assert len(frames) == 2
    assert frames[0].shape == (16, 24, 3)


def test_encode_mp4_writes_decodable_video(tmp_path) -> None:
    """The raw-frame pipe closes cleanly and emits the requested geometry."""
    output = tmp_path / "clip.mp4"
    frames = [np.full((16, 24, 3), value, dtype=np.uint8) for value in (0, 255)]

    width, height = MinimaxH3LocalAdapter._encode_mp4(frames, str(output), fps=2)

    assert (width, height) == (24, 16)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "csv=p=0",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "24,16,2"


def test_adapter_source_uses_nvfp4_not_int8() -> None:
    """H3 must load through NVFP4WeightOnlyConfig — never the old int8 path."""
    from arbiter.adapters import minimax_h3_local as h3

    src = inspect.getsource(h3)
    assert "NVFP4WeightOnlyConfig" in src
    assert "_nvfp4_config" in src
    assert "Int8WeightOnlyConfig" not in src
    assert "_int8_config" not in src
    # Group-offload was the int8 streaming path; NVFP4 stays on-device.
    assert "enable_group_offload" not in src


def test_local_and_cloud_adapters_are_both_registered() -> None:
    """Cloud restore must not erase the local GPU model id the worker looks up."""
    registered = list_registered()
    assert "minimax-h3" in registered
    assert "minimax-h3-local" in registered
    assert MinimaxH3LocalAdapter.model_id == "minimax-h3-local"


def test_local_deploy_config_is_discoverable() -> None:
    root = Path(__file__).parents[3]
    config = json.loads((root / "config/spark/minimax-h3-local.model.json").read_text())
    assert config["max_concurrent"] == 1
    assert config["max_instances"] == 1
    assert config["memory_gb"] >= 35
    assert config["load_ms"] >= 60_000
