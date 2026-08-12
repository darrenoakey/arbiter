"""Tests for the local MiniMax H3 adapter's output boundary."""

from __future__ import annotations

import subprocess

import numpy as np

from arbiter.adapters.minimax_h3 import MinimaxH3Adapter


def test_extract_video_frames_unwraps_single_video_batch() -> None:
    """Diffusers' single-video batch becomes the frame sequence to encode."""
    videos = np.zeros((1, 2, 16, 24, 3), dtype=np.uint8)

    frames = MinimaxH3Adapter._extract_video_frames(videos)

    assert len(frames) == 2
    assert frames[0].shape == (16, 24, 3)


def test_encode_mp4_writes_decodable_video(tmp_path) -> None:
    """The raw-frame pipe closes cleanly and emits the requested geometry."""
    output = tmp_path / "clip.mp4"
    frames = [np.full((16, 24, 3), value, dtype=np.uint8) for value in (0, 255)]

    width, height = MinimaxH3Adapter._encode_mp4(frames, str(output), fps=2)

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
    import inspect

    from arbiter.adapters import minimax_h3 as h3

    src = inspect.getsource(h3)
    assert "NVFP4WeightOnlyConfig" in src
    assert "_nvfp4_config" in src
    assert "Int8WeightOnlyConfig" not in src
    assert "_int8_config" not in src
    # Group-offload was the int8 streaming path; NVFP4 stays on-device.
    assert "enable_group_offload" not in src


def test_nvfp4_config_wraps_torchao_weight_only_config() -> None:
    """Load-time quant must be NVFP4 weight-only when torchao is installed."""
    import pytest

    pytest.importorskip("torchao")
    pytest.importorskip("transformers")
    from arbiter.adapters import minimax_h3 as h3

    # Import path is the contract the adapter documents; fail closed if torchao
    # moves NVFP4WeightOnlyConfig out of the prototype workflow without us.
    cfg = h3._nvfp4_config(["proj_out", "lm_head"], transformers_flavour=True)
    assert cfg.modules_to_not_convert == ["proj_out", "lm_head"]
    quant = cfg.quant_type
    assert quant.__class__.__name__ == "NVFP4WeightOnlyConfig"
    assert getattr(quant, "use_dynamic_per_tensor_scale", None) is True


def test_nvfp4_config_uses_diffusers_wrapper_by_default() -> None:
    """Transformer loads go through diffusers.TorchAoConfig when installed."""
    import pytest

    pytest.importorskip("torchao")
    pytest.importorskip("diffusers")
    from arbiter.adapters import minimax_h3 as h3

    cfg = h3._nvfp4_config(h3.H3_TRANSFORMER_FP_MODULES)
    assert cfg.modules_to_not_convert == list(h3.H3_TRANSFORMER_FP_MODULES)
    assert cfg.quant_type.__class__.__name__ == "NVFP4WeightOnlyConfig"
