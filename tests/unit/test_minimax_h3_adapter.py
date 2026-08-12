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
