"""Opt-in stage1_guiding_keyframes contract for ltx25-denoise1.

Schema and adapter checks reject non-booleans before any pipeline load.
The tensor check uses the real LTX conditioning classes and real torch
tensors from the local LTX 2.5 mirror — not stand-ins. It does not load a
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
    LTX25Denoise1Adapter,
    apply_stage1_guiding_keyframes,
    maybe_apply_stage1_guiding_keyframes,
    parse_stage1_guiding_keyframes,
)
from arbiter.schemas import LTX25Denoise1Params

OUT = Path("/tmp/ltx25-stage1-guiding-unit-out")
_MIRROR_CANDIDATES = (
    Path("/Users/darrenoakey/src/ltx25-spark-mirror/packages/ltx-core/src"),
    Path.home() / "src/ltx25-spark/packages/ltx-core/src",
)


def _denoise1(params: dict) -> None:
    LTX25Denoise1Adapter().infer(params, OUT, threading.Event())


def _existing_files(tmp_path: Path) -> dict:
    encoded = tmp_path / "encoded.pt"
    audio = tmp_path / "in.mp3"
    encoded.write_bytes(b"x")
    audio.write_bytes(b"x")
    return {"encoded_file": str(encoded), "audio_file": str(audio)}


def _import_real_conditioning_types():
    mirror = next((path for path in _MIRROR_CANDIDATES if path.is_dir()), None)
    if mirror is None:
        raise AssertionError(
            "LTX 2.5 core mirror is required for the real conditioning-class "
            f"contract test; looked in {_MIRROR_CANDIDATES}"
        )
    mirror_str = str(mirror)
    if mirror_str not in sys.path:
        sys.path.insert(0, mirror_str)
    import torch
    from ltx_core.conditioning.types.keyframe_cond import (
        VideoConditionByKeyframeIndex,
    )
    from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex

    assert not torch.cuda.is_initialized()
    return torch, VideoConditionByLatentIndex, VideoConditionByKeyframeIndex


class TestStage1GuidingSchema:
    def test_default_is_false(self):
        params = LTX25Denoise1Params(encoded_file="/e.pt", audio_file="/x.mp3")
        assert params.stage1_guiding_keyframes is False

    def test_real_boolean_accepted(self):
        params = LTX25Denoise1Params(
            encoded_file="/e.pt",
            audio_file="/x.mp3",
            stage1_guiding_keyframes=True,
        )
        assert params.stage1_guiding_keyframes is True

    @pytest.mark.parametrize("value", [1, 0, "true", "false", "yes", None])
    def test_non_boolean_rejected(self, value):
        with pytest.raises(ValidationError, match="must be a JSON boolean"):
            LTX25Denoise1Params(
                encoded_file="/e.pt",
                audio_file="/x.mp3",
                stage1_guiding_keyframes=value,
            )


class TestStage1GuidingAdapterValidation:
    def test_missing_flag_reaches_unloaded_gate(self, tmp_path: Path):
        with pytest.raises(InferenceError, match="not loaded"):
            _denoise1(_existing_files(tmp_path))

    def test_explicit_false_reaches_unloaded_gate(self, tmp_path: Path):
        params = _existing_files(tmp_path)
        params["stage1_guiding_keyframes"] = False
        with pytest.raises(InferenceError, match="not loaded"):
            _denoise1(params)

    def test_explicit_true_is_accepted_before_unloaded_gate(self, tmp_path: Path):
        params = _existing_files(tmp_path)
        params["stage1_guiding_keyframes"] = True
        with pytest.raises(InferenceError, match="not loaded"):
            _denoise1(params)

    @pytest.mark.parametrize("value", [1, 0, "true", "false", None])
    def test_non_boolean_rejected_before_pipeline(self, tmp_path: Path, value):
        params = _existing_files(tmp_path)
        params["stage1_guiding_keyframes"] = value
        with pytest.raises(InferenceError, match="must be a boolean"):
            _denoise1(params)

    def test_parser_missing_is_false(self):
        assert parse_stage1_guiding_keyframes({}) is False


class TestStage1GuidingTensorContract:
    def test_disabled_path_does_not_touch_payload_or_import(self):
        sentinel = object()
        payload = {
            "stage_1_conditionings": [sentinel],
            "seed": 2089827940,
            "images": ("boundary",),
        }
        before_modules = set(sys.modules)
        maybe_apply_stage1_guiding_keyframes(payload, False)
        assert payload["stage_1_conditionings"][0] is sentinel
        assert payload["seed"] == 2089827940
        assert payload["images"] == ("boundary",)
        assert "ltx_core.conditioning.types.latent_cond" not in (
            set(sys.modules) - before_modules
        )

    def test_converts_only_frame0_and_keeps_tensor_identity(self):
        torch, latent_type, keyframe_type = _import_real_conditioning_types()
        frame0 = torch.arange(2 * 4 * 1 * 3 * 5, dtype=torch.float32).reshape(
            2, 4, 1, 3, 5
        )
        end = torch.zeros(2, 4, 1, 3, 5)
        frame0_item = latent_type(latent=frame0, strength=0.8, latent_idx=0)
        end_item = keyframe_type(keyframes=end, frame_idx=128, strength=0.5)
        audio = torch.ones(1, 8, 3)
        images = ("start-boundary", "end-boundary")
        payload = {
            "stage_1_conditionings": [frame0_item, end_item],
            "seed": 2089827940,
            "images": images,
            "encoded_audio_latent": audio,
            "v_context_p": torch.zeros(1, 2),
        }
        untouched = {
            key: payload[key]
            for key in ("seed", "images", "encoded_audio_latent", "v_context_p")
        }

        apply_stage1_guiding_keyframes(payload)

        converted = payload["stage_1_conditionings"][0]
        assert type(converted) is keyframe_type
        assert converted.keyframes is frame0
        assert converted.keyframes.data_ptr() == frame0.data_ptr()
        assert converted.frame_idx == 0
        assert converted.strength == 0.8
        assert payload["stage_1_conditionings"][1] is end_item
        assert end_item.keyframes is end
        for key, value in untouched.items():
            assert payload[key] is value
        assert not torch.cuda.is_initialized()

    def test_missing_conditionings_rejected_without_rewriting_other_keys(self):
        seed = object()
        payload = {"seed": seed}
        with pytest.raises(InferenceError, match="stage_1_conditionings"):
            apply_stage1_guiding_keyframes(payload)
        assert payload["seed"] is seed
        assert "stage_1_conditionings" not in payload

    def test_nonzero_latent_index_rejected_and_list_unchanged(self):
        torch, latent_type, _keyframe_type = _import_real_conditioning_types()
        latent = torch.zeros(1, 4, 1, 2, 2)
        item = latent_type(latent=latent, strength=1.0, latent_idx=3)
        conditionings = [item]
        payload = {"stage_1_conditionings": conditionings, "images": ("keep",)}
        with pytest.raises(InferenceError, match="latent_idx"):
            apply_stage1_guiding_keyframes(payload)
        assert payload["stage_1_conditionings"] is conditionings
        assert conditionings[0] is item
        assert payload["images"] == ("keep",)

    def test_rank4_latent_rejected(self):
        torch, latent_type, _keyframe_type = _import_real_conditioning_types()
        item = latent_type(latent=torch.zeros(1, 4, 2, 2), strength=1.0, latent_idx=0)
        conditionings = [item]
        with pytest.raises(InferenceError, match="rank-5"):
            apply_stage1_guiding_keyframes({"stage_1_conditionings": conditionings})
        assert conditionings[0] is item

    def test_non_tensor_latent_rejected(self):
        _torch, latent_type, _keyframe_type = _import_real_conditioning_types()
        item = latent_type(latent=[1, 2, 3], strength=1.0, latent_idx=0)
        with pytest.raises(InferenceError, match="torch.Tensor"):
            apply_stage1_guiding_keyframes({"stage_1_conditionings": [item]})

    def test_unknown_conditioning_type_rejected(self):
        _import_real_conditioning_types()

        class NotAConditioning:
            pass

        with pytest.raises(InferenceError, match="expected VideoConditionByLatentIndex"):
            apply_stage1_guiding_keyframes(
                {"stage_1_conditionings": [NotAConditioning()]}
            )

    def test_two_frame0_items_rejected_without_partial_rewrite(self):
        torch, latent_type, _keyframe_type = _import_real_conditioning_types()
        first = latent_type(
            latent=torch.zeros(1, 2, 1, 2, 2), strength=1.0, latent_idx=0
        )
        second = latent_type(
            latent=torch.ones(1, 2, 1, 2, 2), strength=0.4, latent_idx=0
        )
        conditionings = [first, second]
        with pytest.raises(InferenceError, match="exactly one"):
            apply_stage1_guiding_keyframes({"stage_1_conditionings": conditionings})
        assert conditionings == [first, second]

    def test_no_frame0_latent_index_rejected(self):
        torch, _latent_type, keyframe_type = _import_real_conditioning_types()
        end = keyframe_type(
            keyframes=torch.zeros(1, 2, 1, 2, 2), frame_idx=8, strength=0.5
        )
        conditionings = [end]
        with pytest.raises(InferenceError, match="found 0"):
            apply_stage1_guiding_keyframes({"stage_1_conditionings": conditionings})
        assert conditionings[0] is end
