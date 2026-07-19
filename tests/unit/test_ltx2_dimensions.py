"""Spatial and media-timing contracts for the split LTX-2 dev pipeline."""

import json
import subprocess

import numpy as np
import pytest

from arbiter.adapters.base import InferenceError
from arbiter.adapters.ltx2_denoise2 import _crop_frames_to_target, _mux_audio_slice


def test_model_only_spatial_padding_is_cropped_to_exact_1080p() -> None:
    frames = np.arange(2 * 1088 * 1920 * 3, dtype=np.uint8).reshape(2, 1088, 1920, 3)

    cropped = _crop_frames_to_target(frames, 1920, 1080)

    assert cropped.shape == (2, 1080, 1920, 3)
    assert np.array_equal(cropped[:, 0], frames[:, 4])
    assert np.array_equal(cropped[:, -1], frames[:, 1083])


def test_model_crop_rejects_target_larger_than_decoded_frames() -> None:
    frames = np.zeros((1, 1080, 1920, 3), dtype=np.uint8)

    with pytest.raises(InferenceError, match="exceeds decoded model frame"):
        _crop_frames_to_target(frames, 1920, 1088)


@pytest.mark.parametrize("source_audio_end", [134.304625, 135.12])
def test_audio_mux_preserves_all_video_frames_at_native_rate(
    tmp_path, source_audio_end: float
) -> None:
    """Short audio is padded and ordinary audio remains a valid AAC stream."""
    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.wav"
    output_path = tmp_path / "result.mp4"

    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=size=16x16:rate=25",
            "-frames:v",
            "105",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            str(source_audio_end),
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ],
        check=True,
    )

    _mux_audio_slice(
        video_path=str(video_path),
        audio_path=str(audio_path),
        start_time=130.12,
        video_duration=105 / 25,
        output_path=str(output_path),
    )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,codec_name,duration,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = next(stream for stream in streams if stream["codec_type"] == "audio")

    assert video["nb_read_frames"] == "105"
    assert video["r_frame_rate"] == "25/1"
    assert video["duration"] == "4.200000"
    assert audio["codec_name"] == "aac"
    assert audio["duration"] == "4.200000"
    decoded_audio = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(output_path),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    assert any(decoded_audio.stdout)
