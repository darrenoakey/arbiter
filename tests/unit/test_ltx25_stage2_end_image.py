"""Opt-in stage2_drop_end_image contract for ltx25-denoise1.

The flag only filters the images list that the spark runner uses for stage 2.
It must not rewrite stage-1 conditionings or import a pipeline module.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
from pydantic import ValidationError

from arbiter.adapters.base import InferenceError
from arbiter.adapters.ltx25_denoise1 import (
    drop_stage2_end_image,
    maybe_drop_stage2_end_image,
    parse_stage2_drop_end_image,
)
from arbiter.schemas import LTX25Denoise1Params


class TestStage2DropEndImage:
    def test_schema_default_is_false(self):
        params = LTX25Denoise1Params(encoded_file="/e.pt", audio_file="/x.mp3")
        assert params.stage2_drop_end_image is False

    def test_schema_rejects_non_bool(self):
        with pytest.raises(ValidationError, match="stage2_drop_end_image"):
            LTX25Denoise1Params(
                encoded_file="/e.pt",
                audio_file="/x.mp3",
                stage2_drop_end_image=1,
            )

    def test_parser_missing_is_false(self):
        assert parse_stage2_drop_end_image({}) is False

    def test_parser_rejects_int(self):
        with pytest.raises(InferenceError, match="boolean"):
            parse_stage2_drop_end_image({"stage2_drop_end_image": 1})

    def test_disabled_does_not_touch_images(self):
        images = [{"path": "/start.png", "frame_idx": 0, "strength": 1.0}]
        payload = {"images": images, "stage_1_conditionings": ["keep"]}
        maybe_drop_stage2_end_image(payload, False)
        assert payload["images"] is images
        assert payload["stage_1_conditionings"] == ["keep"]

    def test_drops_end_image_and_keeps_stage1(self):
        stage1 = ["start-cond", "end-cond"]
        payload = {
            "images": [
                {"path": "/start.png", "frame_idx": 0, "strength": 1.0},
                {"path": "/end.png", "frame_idx": 128, "strength": 1.0},
            ],
            "stage_1_conditionings": stage1,
            "seed": 2089827940,
        }
        drop_stage2_end_image(payload)
        assert payload["images"] == [
            {"path": "/start.png", "frame_idx": 0, "strength": 1.0}
        ]
        assert payload["stage_1_conditionings"] is stage1
        assert payload["seed"] == 2089827940

    def test_namedtuple_uses_frame_idx_not_strength(self):
        class ImageConditioningInput(NamedTuple):
            path: str
            frame_idx: int
            strength: float
            crf: int | None = None

        stage1 = ["keep"]
        payload = {
            "images": [
                ImageConditioningInput("/start.png", 0, 1.0),
                ImageConditioningInput("/end.png", 128, 1.0),
            ],
            "stage_1_conditionings": stage1,
        }
        drop_stage2_end_image(payload)
        assert payload["images"] == [ImageConditioningInput("/start.png", 0, 1.0)]
        assert payload["stage_1_conditionings"] is stage1

    def test_fractional_frame_fails_closed(self):
        with pytest.raises(InferenceError, match="integer frame_idx"):
            drop_stage2_end_image({
                "images": [{"path": "/start.png", "frame_idx": 1.5}],
                "stage_1_conditionings": [],
            })

    def test_missing_end_image_fails_closed(self):
        with pytest.raises(InferenceError, match="no end image"):
            drop_stage2_end_image({
                "images": [{"path": "/start.png", "frame_idx": 0}],
                "stage_1_conditionings": [],
            })
