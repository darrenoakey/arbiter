"""Real CPU tensor coverage for the LTX 2.5 temporal attention-ramp core."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from arbiter.adapters.base import InferenceError
from arbiter.adapters.ltx25_temporal_attention import (
    CAPABILITY_VERSION,
    END_IMAGE_ATTENTION_STRENGTHS,
    endpoint_guide_log_bias,
    normalize_end_image_attention_strengths,
    preserve_encoded_temporal_attention_ramp,
    resolve_denoise_temporal_attention_ramp,
    sample_video_query_strengths,
    validate_ramp_images,
)
from arbiter.schemas import LTX25Denoise1Params, LTX25EncodeParams

_CORE = Path(
    "/Users/darrenoakey/src/ltx25-spark-mirror/packages/ltx-core/src"
)
_PROVENANCE = {
    "capability_version": CAPABILITY_VERSION,
    "release_id": "test-release",
}


def _real_ltx():
    if not _CORE.is_dir():
        raise AssertionError(f"LTX 2.5 core mirror is required at {_CORE}")
    if str(_CORE) not in sys.path:
        sys.path.insert(0, str(_CORE))
    import torch
    from ltx_core.components.patchifiers import (
        VideoLatentPatchifier,
        get_pixel_coords,
    )
    from ltx_core.conditioning.types.keyframe_cond import (
        VideoConditionByKeyframeIndex,
    )
    from ltx_core.conditioning.types.latent_cond import (
        VideoConditionByLatentIndex,
    )
    from ltx_core.model.transformer.attention import PytorchAttention
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    assert not torch.cuda.is_initialized()
    return (
        torch,
        VideoLatentPatchifier,
        get_pixel_coords,
        VideoConditionByKeyframeIndex,
        VideoConditionByLatentIndex,
        PytorchAttention,
        SpatioTemporalScaleFactors,
        VideoLatentShape,
    )


def _stage_positions(height: int, width: int):
    (
        torch,
        patchifier_type,
        get_pixel_coords,
        _endpoint_type,
        _start_type,
        _attention_type,
        scale_type,
        shape_type,
    ) = _real_ltx()
    shape = shape_type(batch=1, channels=128, frames=16, height=height, width=width)
    bounds = patchifier_type(1).get_patch_grid_bounds(shape)
    positions = get_pixel_coords(bounds, scale_type.default(), causal_fix=True).float()
    positions[:, 0] /= 25.0
    assert not torch.cuda.is_initialized()
    return positions, shape.token_count()


def _conditioning_bundle(num_frames: int = 100):
    (
        torch,
        _patchifier_type,
        _get_pixel_coords,
        endpoint_type,
        start_type,
        _attention_type,
        _scale_type,
        _shape_type,
    ) = _real_ltx()
    latent = torch.zeros(1, 4, 1, 2, 2)
    start = start_type(latent=latent, strength=0.8, latent_idx=0)
    endpoint = endpoint_type(
        keyframes=latent,
        frame_idx=num_frames - 1,
        strength=0.37,
    )
    return {
        "num_frames": num_frames,
        "images": [
            {"path": "/start.png", "frame_idx": 0, "strength": 0.8},
            {
                "path": "/end.png",
                "frame_idx": num_frames - 1,
                "strength": 0.37,
            },
        ],
        "stage_1_conditionings": [start, endpoint],
        "ltx25_temporal_attention_capability": {"version": CAPABILITY_VERSION},
    }, start, endpoint


class TestRampContract:
    def test_pair_expands_to_positive_whole_clip_ramp(self):
        strengths = normalize_end_image_attention_strengths([0.01, 1], 100)
        assert strengths is not None
        assert len(strengths) == 100
        assert strengths[0] == pytest.approx(0.01)
        assert strengths[-1] == 1.0
        assert all(left < right for left, right in zip(strengths, strengths[1:]))

    def test_expanded_list_is_preserved_exactly(self):
        expected = [index / 99 for index in range(100)]
        assert normalize_end_image_attention_strengths(expected, 100) == tuple(
            expected
        )

    def test_single_frame_has_no_distinct_endpoint(self):
        with pytest.raises(InferenceError, match="at least 2 frames"):
            normalize_end_image_attention_strengths([1.0], 1)

    @pytest.mark.parametrize(
        "value",
        [True, "0.2", [0.1], [0.1, 0.2, 0.3], [False, 1.0], [math.nan, 1], [0, math.inf], [-0.1, 1], [0, 1.1]],
    )
    def test_invalid_shapes_and_values_fail_closed(self, value):
        with pytest.raises(InferenceError, match=END_IMAGE_ATTENTION_STRENGTHS):
            normalize_end_image_attention_strengths(value, 100)

    @pytest.mark.parametrize("model", [LTX25EncodeParams, LTX25Denoise1Params])
    def test_schema_rejects_boolean_entries(self, model):
        common = {"audio_file": "/audio.wav"}
        if model is LTX25EncodeParams:
            common.update(audio_duration=4.0, num_frames=100)
        else:
            common.update(encoded_file="/encoded.pt")
        with pytest.raises(ValidationError, match=END_IMAGE_ATTENTION_STRENGTHS):
            model(**common, end_image_attention_strengths=[False, 1.0])

    def test_encode_schema_preserves_single_endpoint_guide(self):
        params = LTX25EncodeParams(
            audio_file="/audio.wav",
            audio_duration=4.0,
            num_frames=97,
            images=[
                {"path": "/start.png", "frame_idx": 0},
                {"path": "/end.png", "frame_idx": 96, "strength": 0.37},
            ],
            end_image_attention_strengths=[1 / 97, 1.0],
        )
        dumped = params.model_dump()
        assert dumped["images"] == [
            {"path": "/start.png", "frame_idx": 0, "strength": 1.0, "crf": None},
            {"path": "/end.png", "frame_idx": 96, "strength": 0.37, "crf": None},
        ]


class TestEndpointIdentityAndBundle:
    def test_normalized_schedule_survives_encode_bundle_preservation(self):
        bundle, _start, _endpoint = _conditioning_bundle()
        normalized = normalize_end_image_attention_strengths([0.01, 1.0], 100)
        assert normalized is not None
        preserved = preserve_encoded_temporal_attention_ramp(
            bundle, list(normalized), 100, _PROVENANCE
        )
        assert preserved == normalized
        assert bundle[END_IMAGE_ATTENTION_STRENGTHS] == list(normalized)

    def test_preserves_start_and_endpoint_denoise_strength(self):
        bundle, start, endpoint = _conditioning_bundle()
        strengths = preserve_encoded_temporal_attention_ramp(
            bundle, [0.01, 1.0], 100, _PROVENANCE
        )
        assert strengths is not None
        assert bundle["stage_1_conditionings"][0] is start
        assert bundle["stage_1_conditionings"][1] is endpoint
        assert start.strength == 0.8
        assert endpoint.strength == 0.37
        assert bundle[END_IMAGE_ATTENTION_STRENGTHS] == list(strengths)

    def test_repeated_scheduled_guides_are_rejected(self):
        images = [
            {"frame_idx": 0},
            {"frame_idx": 50},
            {"frame_idx": 99},
        ]
        with pytest.raises(InferenceError, match="scheduled end guides"):
            validate_ramp_images(images, 100)

    def test_ambiguous_endpoint_is_rejected(self):
        images = [{"frame_idx": 0}, {"frame_idx": 99}, {"frame_idx": 99}]
        with pytest.raises(InferenceError, match="exactly one endpoint"):
            validate_ramp_images(images, 100)

    def test_request_must_match_encoded_schedule(self):
        bundle, _start, _endpoint = _conditioning_bundle()
        preserve_encoded_temporal_attention_ramp(
            bundle, [0.01, 1.0], 100, _PROVENANCE
        )
        with pytest.raises(InferenceError, match="does not match"):
            resolve_denoise_temporal_attention_ramp(
                bundle,
                {END_IMAGE_ATTENTION_STRENGTHS: [0.02, 1.0]},
                _PROVENANCE,
            )

    def test_omission_keeps_historical_path_and_mixed_runtime_fails_loudly(self):
        bundle, _start, _endpoint = _conditioning_bundle()
        assert resolve_denoise_temporal_attention_ramp(bundle, {}) is None
        preserve_encoded_temporal_attention_ramp(
            bundle, [0.01, 1.0], 100, _PROVENANCE
        )
        with pytest.raises(InferenceError, match="runtime identity"):
            resolve_denoise_temporal_attention_ramp(
                bundle, {}, {"release_id": "different"}
            )


class TestLatentTimeSamplingAndAttention:
    @pytest.mark.parametrize("height,width", [(17, 30), (34, 60)])
    def test_real_stage_layout_repeats_each_time_across_spatial_queries(
        self, height, width
    ):
        positions, token_count = _stage_positions(height, width)
        strengths = tuple((index + 1) / 100 for index in range(100))
        sampled = sample_video_query_strengths(strengths, positions, token_count)
        grid = sampled.reshape(1, 16, height * width)
        assert torch_all_equal_per_row(grid)
        assert grid[0, 0, 0].item() == pytest.approx(0.01)
        assert grid[0, -1, 0].item() == pytest.approx(1.0)

    def test_real_attention_matches_additive_log_reference_and_zero_masks(self):
        (
            torch,
            _patchifier_type,
            _get_pixel_coords,
            _endpoint_type,
            _start_type,
            attention_type,
            _scale_type,
            _shape_type,
        ) = _real_ltx()
        torch.manual_seed(240924)
        video_tokens = 4
        endpoint_tokens = 2
        total = video_tokens + endpoint_tokens
        q = torch.randn(1, total, 8)
        k = torch.randn(1, total, 8)
        v = torch.randn(1, total, 8)
        query_strengths = torch.tensor([[0.0, 0.25, 0.5, 1.0]])
        block = endpoint_guide_log_bias(query_strengths, endpoint_tokens)
        bias = torch.zeros(1, total, total)
        bias[:, :video_tokens, video_tokens:] = block

        actual = attention_type()(q, k, v, heads=1, mask=bias)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        expected = torch.softmax(scores + bias, dim=-1) @ v
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
        assert torch.all(bias[:, 0, video_tokens:] == torch.finfo(torch.float32).min)

        reference = attention_type()(q, k, v, heads=1, mask=torch.zeros_like(bias))
        assert not torch.allclose(actual[:, 0], reference[:, 0])
        assert torch.allclose(actual[:, 3], reference[:, 3], atol=1e-6, rtol=1e-6)
        assert torch.all(bias[:, video_tokens:, :video_tokens] == 0)

    def test_stage2_bias_is_a_compact_endpoint_block_view(self):
        torch, *_rest = _real_ltx()
        query_strengths = torch.linspace(0.01, 1.0, 32_640).reshape(1, -1)
        block = endpoint_guide_log_bias(query_strengths, 2_040)
        assert block.shape == (1, 32_640, 2_040)
        assert block.stride(-1) == 0
        assert block.untyped_storage().nbytes() == query_strengths.numel() * 4
        assert block[0, 0, 0].item() == pytest.approx(math.log(0.01))
        assert block[0, -1, -1].item() == pytest.approx(0.0)


def torch_all_equal_per_row(grid) -> bool:
    return bool((grid == grid[:, :, :1]).all().item())
