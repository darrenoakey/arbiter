"""Param-validation tests for the vocal-stem adapter and the ltx25-denoise1
a2v_guidance_scale lever.

Every check here runs BEFORE the adapter's loaded-pipeline/ready gate, so
they exercise the real validation code on any machine — the failure mode
under test is a malformed job being rejected with InferenceError, not the
model running. Real separation/normalization runs on spark via
tests/integration/test_voice_jobs.py::TestVocalStem.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from arbiter.adapters.base import InferenceError
from arbiter.adapters.ltx25_denoise1 import LTX25Denoise1Adapter
from arbiter.adapters.vocal_stem import VocalStemAdapter

OUT = Path("/tmp/vocal-stem-unit-out")


def _vocal_stem(params: dict) -> None:
    VocalStemAdapter().infer(params, OUT, threading.Event())


def _denoise1(params: dict) -> None:
    LTX25Denoise1Adapter().infer(params, OUT, threading.Event())


class TestVocalStemParamValidation:
    def test_missing_audio_file_rejected(self):
        with pytest.raises(InferenceError, match="audio_file is required"):
            _vocal_stem({})

    def test_nonexistent_audio_file_rejected(self, tmp_path: Path):
        with pytest.raises(InferenceError, match="does not exist"):
            _vocal_stem({"audio_file": str(tmp_path / "nope.wav")})

    def test_unsupported_separator_model_rejected(self, tmp_path: Path):
        wav = tmp_path / "in.wav"
        wav.write_bytes(b"RIFF")  # existence is checked, content is not
        with pytest.raises(InferenceError, match="only 'htdemucs'"):
            _vocal_stem({"audio_file": str(wav), "model": "mdx"})

    def test_nonfinite_target_lufs_rejected(self, tmp_path: Path):
        wav = tmp_path / "in.wav"
        wav.write_bytes(b"RIFF")
        with pytest.raises(InferenceError, match="target_lufs must be finite"):
            _vocal_stem({"audio_file": str(wav), "target_lufs": float("inf")})

    def test_valid_params_pass_validation_then_fail_not_loaded(
        self, tmp_path: Path
    ):
        # Params validate clean; the job then stops at the unloaded gate.
        wav = tmp_path / "in.wav"
        wav.write_bytes(b"RIFF")
        with pytest.raises(InferenceError, match="not loaded"):
            _vocal_stem({"audio_file": str(wav)})


class TestLTX25Denoise1GuidanceScaleValidation:
    def test_default_scale_accepted_past_validation(self, tmp_path: Path):
        encoded = tmp_path / "encoded.pt"
        audio = tmp_path / "in.mp3"
        encoded.write_bytes(b"x")
        audio.write_bytes(b"x")
        # 3.0 default passes validation; the unloaded pipeline gate fires next.
        with pytest.raises(InferenceError, match="not loaded"):
            _denoise1({"encoded_file": str(encoded), "audio_file": str(audio)})

    def test_scale_below_one_rejected(self, tmp_path: Path):
        encoded = tmp_path / "encoded.pt"
        audio = tmp_path / "in.mp3"
        encoded.write_bytes(b"x")
        audio.write_bytes(b"x")
        with pytest.raises(InferenceError, match="a2v_guidance_scale must be >= 1.0"):
            _denoise1(
                {
                    "encoded_file": str(encoded),
                    "audio_file": str(audio),
                    "a2v_guidance_scale": 0.5,
                }
            )

    def test_non_numeric_scale_rejected(self, tmp_path: Path):
        encoded = tmp_path / "encoded.pt"
        audio = tmp_path / "in.mp3"
        encoded.write_bytes(b"x")
        audio.write_bytes(b"x")
        with pytest.raises(InferenceError, match="must be a number"):
            _denoise1(
                {
                    "encoded_file": str(encoded),
                    "audio_file": str(audio),
                    "a2v_guidance_scale": "high",
                }
            )
