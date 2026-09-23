"""Opt-in end_image_strength contract for ltx25-denoise1.

The flag rewrites only end-anchor strength. It must not import a pipeline
module or change frame 0.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
from pydantic import ValidationError

from arbiter.adapters.base import InferenceError
from arbiter.adapters.ltx25_denoise1 import (
    apply_end_image_strength,
    maybe_apply_end_image_strength,
    parse_end_image_strength,
)
from arbiter.schemas import LTX25Denoise1Params


class _End:
    def __init__(self, frame_idx: int, strength: float):
        self.frame_idx = frame_idx
        self.strength = strength


class _Image(NamedTuple):
    path: str
    frame_idx: int
    strength: float


class TestEndImageStrength:
    def test_schema_default_is_absent(self):
        params = LTX25Denoise1Params(encoded_file="/e.pt", audio_file="/x.mp3")
        assert params.end_image_strength is None

    def test_schema_accepts_quarter_strength(self):
        params = LTX25Denoise1Params(
            encoded_file="/e.pt",
            audio_file="/x.mp3",
            end_image_strength=0.25,
        )
        assert params.end_image_strength == 0.25

    @pytest.mark.parametrize("value", [True, "0.25", 0, 1.5, -0.1])
    def test_schema_rejects_bad_values(self, value):
        with pytest.raises(ValidationError, match="end_image_strength"):
            LTX25Denoise1Params(
                encoded_file="/e.pt",
                audio_file="/x.mp3",
                end_image_strength=value,
            )

    def test_parser_missing_is_none(self):
        assert parse_end_image_strength({}) is None

    def test_disabled_does_not_touch_bundle(self):
        end = _End(128, 1.0)
        payload = {"stage_1_conditionings": [end], "images": []}
        maybe_apply_end_image_strength(payload, None)
        assert end.strength == 1.0

    def test_sets_end_strength_and_keeps_frame_zero(self):
        start = _End(0, 1.0)
        end = _End(128, 1.0)
        payload = {
            "stage_1_conditionings": [start, end],
            "images": [
                _Image("/start.png", 0, 1.0),
                _Image("/end.png", 128, 1.0),
            ],
            "seed": 2089827940,
        }
        apply_end_image_strength(payload, 0.25)
        assert start.strength == 1.0
        assert end.strength == 0.25
        assert payload["images"][0].strength == 1.0
        assert payload["images"][1].strength == 0.25
        assert payload["seed"] == 2089827940

    def test_missing_end_conditioning_fails_closed(self):
        with pytest.raises(InferenceError, match="no stage-1 end"):
            apply_end_image_strength({
                "stage_1_conditionings": [_End(0, 1.0)],
                "images": [_Image("/end.png", 128, 1.0)],
            }, 0.25)
