"""Opt-in generated_keyframes contract for ltx25-denoise1.

Schema and adapter checks reject non-integers before any pipeline load.
The slot check uses the real LTX VideoGeneratedKeyframeSlots class and the
official position helper from the local LTX 2.5 mirror. It does not load a
checkpoint or touch CUDA.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from arbiter.adapters.base import InferenceError
from arbiter.adapters.ltx25_denoise1 import (
    GENERATED_KEYFRAMES_MAX,
    _OFFICIAL_KEYFRAME_POSITION_LINE,
    LTX25Denoise1Adapter,
    apply_generated_keyframes,
    interior_keyframe_positions,
    maybe_apply_generated_keyframes,
    parse_generated_keyframes,
)
from arbiter.schemas import LTX25Denoise1Params

OUT = Path("/tmp/ltx25-generated-keyframes-unit-out")
_CORE = (
    Path("/Users/darrenoakey/src/ltx25-spark-mirror/packages/ltx-core/src"),
    Path.home() / "src/ltx25-spark/packages/ltx-core/src",
)
_HELPERS = (
    Path(
        "/Users/darrenoakey/src/ltx25-spark-mirror/packages/ltx-pipelines/src"
        "/ltx_pipelines/utils/helpers.py"
    ),
    Path.home()
    / "src/ltx25-spark/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py",
)


def _denoise1(params: dict) -> None:
    LTX25Denoise1Adapter().infer(params, OUT, threading.Event())


def _existing_files(tmp_path: Path) -> dict:
    encoded = tmp_path / "encoded.pt"
    audio = tmp_path / "in.mp3"
    encoded.write_bytes(b"x")
    audio.write_bytes(b"x")
    return {"encoded_file": str(encoded), "audio_file": str(audio)}


def _require_mirror(candidates: tuple[Path, ...]) -> Path:
    mirror = next((path for path in candidates if path.is_dir()), None)
    if mirror is None:
        raise AssertionError(
            f"LTX 2.5 mirror is required; looked in {candidates}"
        )
    mirror_str = str(mirror)
    if mirror_str not in sys.path:
        sys.path.insert(0, mirror_str)
    return mirror


def _import_real_slot_types():
    _require_mirror(_CORE)
    import torch
    from ltx_core.conditioning.types.keyframe_slots import (
        VideoGeneratedKeyframeSlots,
    )

    assert not torch.cuda.is_initialized()
    return torch, VideoGeneratedKeyframeSlots


class TestGeneratedKeyframesSchema:
    def test_default_is_zero(self):
        params = LTX25Denoise1Params(encoded_file="/e.pt", audio_file="/x.mp3")
        assert params.generated_keyframes == 0

    def test_real_integer_accepted(self):
        params = LTX25Denoise1Params(
            encoded_file="/e.pt",
            audio_file="/x.mp3",
            generated_keyframes=3,
        )
        assert params.generated_keyframes == 3

    @pytest.mark.parametrize("value", [True, False, 1.5, "3", None, -1, 9])
    def test_non_integer_or_out_of_range_rejected(self, value):
        with pytest.raises(ValidationError, match="generated_keyframes"):
            LTX25Denoise1Params(
                encoded_file="/e.pt",
                audio_file="/x.mp3",
                generated_keyframes=value,
            )


class TestGeneratedKeyframesAdapterValidation:
    def test_missing_count_reaches_unloaded_gate(self, tmp_path: Path):
        with pytest.raises(InferenceError, match="not loaded"):
            _denoise1(_existing_files(tmp_path))

    def test_zero_reaches_unloaded_gate(self, tmp_path: Path):
        params = _existing_files(tmp_path)
        params["generated_keyframes"] = 0
        with pytest.raises(InferenceError, match="not loaded"):
            _denoise1(params)

    def test_positive_count_is_accepted_before_unloaded_gate(self, tmp_path: Path):
        params = _existing_files(tmp_path)
        params["generated_keyframes"] = 3
        with pytest.raises(InferenceError, match="not loaded"):
            _denoise1(params)

    @pytest.mark.parametrize("value", [True, False, 1.5, "3", None, -1, 9])
    def test_bad_count_rejected_before_pipeline(self, tmp_path: Path, value):
        params = _existing_files(tmp_path)
        params["generated_keyframes"] = value
        with pytest.raises(InferenceError, match="generated_keyframes"):
            _denoise1(params)

    def test_parser_missing_is_zero(self):
        assert parse_generated_keyframes({}) == 0
        assert GENERATED_KEYFRAMES_MAX == 8


class TestGeneratedKeyframesSlotContract:
    def test_zero_does_not_touch_payload_or_import_helper(self):
        sentinel = object()
        payload = {
            "stage_1_conditionings": [sentinel],
            "num_frames": 129,
            "seed": 2089827940,
            "images": ("boundary",),
        }
        before_modules = set(sys.modules)
        maybe_apply_generated_keyframes(payload, 0)
        assert payload["stage_1_conditionings"][0] is sentinel
        assert payload["num_frames"] == 129
        assert payload["seed"] == 2089827940
        assert "ltx_core.conditioning.types.keyframe_slots" not in (
            set(sys.modules) - before_modules
        )

    @pytest.mark.parametrize(
        ("count", "num_frames"),
        [(1, 121), (3, 129), (3, 121)],
    )
    def test_appends_official_slots_without_rewriting_existing_items(
        self, count, num_frames
    ):
        torch, slot_type = _import_real_slot_types()
        sentinel = object()
        images = ("start-boundary", "end-boundary")
        payload = {
            "stage_1_conditionings": [sentinel],
            "num_frames": num_frames,
            "seed": 2089827940,
            "images": images,
        }
        expected = tuple(
            torch.linspace(0, num_frames - 1, count + 2)
            .round()
            .to(torch.int64)
            .tolist()[1:-1]
        )
        assert tuple(interior_keyframe_positions(count, num_frames)) == expected

        apply_generated_keyframes(payload, count)

        conditionings = payload["stage_1_conditionings"]
        assert conditionings[0] is sentinel
        assert len(conditionings) == 2
        slot = conditionings[1]
        assert type(slot) is slot_type
        assert tuple(slot.pixel_frame_indices) == expected
        assert expected[0] > 0
        assert expected[-1] < num_frames - 1
        assert payload["seed"] == 2089827940
        assert payload["images"] == images
        assert payload["num_frames"] == num_frames

    def test_missing_conditionings_fail_closed(self):
        with pytest.raises(InferenceError, match="stage_1_conditionings"):
            apply_generated_keyframes({"num_frames": 129}, 3)

    def test_bad_frame_count_fails_closed(self):
        with pytest.raises(InferenceError, match="num_frames"):
            apply_generated_keyframes(
                {"stage_1_conditionings": [], "num_frames": True}, 1
            )

    def test_count_that_does_not_fit_fails_closed(self):
        with pytest.raises(InferenceError, match="generated_keyframes rejected"):
            apply_generated_keyframes(
                {"stage_1_conditionings": [], "num_frames": 4}, 3
            )

    def test_position_formula_matches_upstream_helper_source(self):
        helper = next((path for path in _HELPERS if path.is_file()), None)
        if helper is None:
            raise AssertionError(
                f"upstream helpers.py is required; looked in {_HELPERS}"
            )
        source = helper.read_text()
        assert _OFFICIAL_KEYFRAME_POSITION_LINE in source
        assert "def evenly_spaced_keyframe_positions" in source
