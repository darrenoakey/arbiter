"""CPU numerical verification for the patched LTX runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_runtime_paths(runtime: Path) -> None:
    sys.path[:0] = [
        str(runtime),
        str(runtime / "packages" / "ltx-core" / "src"),
        str(runtime / "packages" / "ltx-pipelines" / "src"),
    ]


def build_conditioned_state(height: int, width: int):
    import torch
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.conditioning.types.attention_strength_wrapper import ConditioningItemAttentionStrengthWrapper
    from ltx_core.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
    from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape

    frames = 13
    num_pixel_frames = 97
    shape = VideoLatentShape(batch=2, channels=8, frames=frames, height=height, width=width)
    tools = VideoLatentTools(VideoLatentPatchifier(1), shape, fps=25.0)
    state = tools.create_initial_state("cpu", torch.float32)
    start_latent = torch.full((2, 8, 1, height, width), 0.25)
    endpoint_latent = torch.full((2, 8, 1, height, width), 0.75)
    state = VideoConditionByLatentIndex(start_latent, strength=0.8, latent_idx=0).apply_to(state, tools)
    start_key_end = state.latent.shape[1]
    frame_strengths = torch.linspace(0.0, 1.0, num_pixel_frames)
    endpoint = VideoConditionByKeyframeIndex(
        endpoint_latent,
        frame_idx=num_pixel_frames - 1,
        strength=0.37,
    )
    state = ConditioningItemAttentionStrengthWrapper(
        endpoint,
        video_query_attention_strengths=frame_strengths,
        query_chunk_size=7,
    ).apply_to(state, tools)
    return state, tools.target_shape.token_count(), start_key_end


def verify_conditioning_shapes() -> None:
    import torch
    from ltx_core.model.transformer.adaln import AdaLayerNormSingle
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.rope import LTXRopeType
    from ltx_core.model.transformer.transformer_args import TransformerArgsPreprocessor

    for height, width in ((2, 3), (4, 6)):
        state, video_tokens, start_key_end = build_conditioned_state(height, width)
        bias = state.temporal_attention_bias
        assert bias is not None
        assert bias.key_start == start_key_end
        assert bias.key_end == state.latent.shape[1]
        assert bias.query_log_weights.shape == (2, video_tokens)
        weights = torch.exp(bias.query_log_weights).reshape(2, 13, height * width)
        assert torch.all(weights == weights[:, :, :1])
        assert torch.all(weights[:, 0] == 0)
        assert torch.allclose(weights[:, -1], torch.ones_like(weights[:, -1]))
        assert torch.all(weights[:, 1:-1] > 0)
        assert torch.all(weights[:, 1:-1] < 1)
        assert torch.all(state.denoise_mask[:, video_tokens:start_key_end] == 0.2)
        assert torch.all(state.denoise_mask[:, start_key_end:] == 0.63)
        modality = Modality(
            latent=state.latent,
            timesteps=state.denoise_mask,
            positions=state.positions,
            context=torch.zeros(2, 1, 8),
            sigma=torch.ones(2),
            attention_mask=state.attention_mask,
            keyframes_mask=state.keyframes_mask,
            temporal_attention_bias=state.temporal_attention_bias,
        )
        preprocessor = TransformerArgsPreprocessor(
            patchify_proj=torch.nn.Linear(8, 8),
            adaln=AdaLayerNormSingle(8),
            inner_dim=8,
            max_pos=[20, 20, 20],
            num_attention_heads=2,
            use_middle_indices_grid=True,
            timestep_scale_multiplier=1000,
            double_precision_rope=False,
            positional_embedding_theta=10000.0,
            rope_type=LTXRopeType.SPLIT,
        )
        transformer_args = preprocessor.prepare(modality)
        assert transformer_args.temporal_attention_bias is bias


def dense_reference(q, k, v, heads, base_mask, query_weights, key_start, key_end):
    import torch

    batch, query_count, width = q.shape
    head_width = width // heads
    qh, kh, vh = (item.view(batch, -1, heads, head_width).transpose(1, 2) for item in (q, k, v))
    full = base_mask.expand(batch, 1, query_count, k.shape[1]).clone()
    target = full[:, :, : query_weights.shape[1], key_start:key_end]
    finfo = torch.finfo(q.dtype)
    positive = query_weights > finfo.min
    target.copy_(torch.where(positive[:, None, :, None], target + query_weights[:, None, :, None], finfo.min))
    expected = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, attn_mask=full)
    return expected.transpose(1, 2).reshape(batch, query_count, width), full


def verify_attention(dtype_name: str) -> None:
    import torch
    from ltx_core.model.transformer.attention import (
        Attention,
        AttentionOps,
        PytorchAttention,
        temporal_attention,
    )
    from ltx_core.types import TemporalAttentionBias

    dtype = getattr(torch, dtype_name)
    torch.manual_seed(240924)
    batch, heads, query_count, width = 2, 2, 11, 8
    key_start, key_end = 9, 11
    q = torch.randn(batch, query_count, width, dtype=dtype)
    k = torch.randn(batch, query_count, width, dtype=dtype)
    v = torch.randn(batch, query_count, width, dtype=dtype)
    assert q.device.type == k.device.type == v.device.type == "cpu"
    base_mask = torch.zeros(1, 1, query_count, query_count, dtype=dtype)
    base_mask[..., 7:, 2] = -0.4
    strengths = torch.tensor([0.0, 0.25, 0.5, 1.0, 0.7, 0.1, 1.0], dtype=dtype)
    finfo = torch.finfo(dtype)
    log_weights = torch.full_like(strengths, finfo.min)
    positive = strengths > 0
    log_weights[positive] = torch.log(strengths[positive])
    compact = TemporalAttentionBias(log_weights.unsqueeze(0), key_start, key_end, query_chunk_size=3)
    attention = PytorchAttention()
    actual = temporal_attention(attention, q, k, v, heads, base_mask, compact)
    expected, full = dense_reference(q, k, v, heads, base_mask, log_weights.expand(batch, -1), key_start, key_end)
    tolerance = 2e-2 if dtype is torch.bfloat16 else 1e-6
    assert torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)
    assert torch.all(full[:, :, 0, key_start:key_end] == finfo.min)
    assert torch.all(full[:, :, 3, key_start:key_end] == 0)
    assert torch.all(full[:, :, :7, 7:key_start] == 0)
    assert torch.all(full[:, :, 7:, 2] == -0.4)

    omitted = attention(q, k, v, heads, base_mask)
    omitted_again = attention(q, k, v, heads, base_mask)
    assert torch.equal(omitted, omitted_again)

    cpu_attention = PytorchAttention()
    component = Attention(
        query_dim=width,
        heads=heads,
        dim_head=width // heads,
        ops=AttentionOps(
            attention_function=cpu_attention,
            masked_attention_function=cpu_attention,
        ),
    )
    x = torch.randn(batch, query_count, width, dtype=dtype)
    component = component.to(dtype=dtype)
    actual_component = component(x, mask=base_mask, temporal_attention_bias=compact)
    component_q = component.to_q(x)
    component_k = component.to_k(x)
    component_v = component.to_v(x)
    component_q, component_k = component.preattention_function(
        component_q, component_k, component, base_mask, None, None
    )
    expected_component, _ = dense_reference(
        component_q,
        component_k,
        component_v,
        heads,
        base_mask,
        log_weights.expand(batch, -1),
        key_start,
        key_end,
    )
    expected_component = component.to_out(expected_component)
    assert torch.allclose(actual_component, expected_component, atol=tolerance, rtol=tolerance)


def verify_runner(runtime: Path) -> None:
    import torch
    import video_fast_gpu
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.conditioning.types.attention_strength_wrapper import ConditioningItemAttentionStrengthWrapper
    from ltx_core.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape

    capability = video_fast_gpu.temporal_attention_capability()
    assert capability == {
        "version": "ltx25-temporal-endpoint-v1",
        "both_stages": True,
        "asymmetric_video_query_to_endpoint_key": True,
        "query_chunk_size": 256,
    }
    assert Path(video_fast_gpu.__file__).resolve().is_relative_to(runtime.resolve())
    strengths = torch.linspace(0.0, 1.0, 97)
    for height, width in ((2, 3), (4, 6)):
        shape = VideoLatentShape(batch=1, channels=8, frames=13, height=height, width=width)
        tools = VideoLatentTools(VideoLatentPatchifier(1), shape, fps=25.0)
        endpoint = VideoConditionByKeyframeIndex(
            torch.zeros(1, 8, 1, height, width), frame_idx=96, strength=0.37
        )
        wrapped = video_fast_gpu._apply_endpoint_attention_strengths([endpoint], strengths, 97)
        assert len(wrapped) == 1
        assert isinstance(wrapped[0], ConditioningItemAttentionStrengthWrapper)
        state = wrapped[0].apply_to(tools.create_initial_state("cpu", torch.float32), tools)
        assert state.temporal_attention_bias is not None
        assert state.temporal_attention_bias.query_log_weights.shape == (1, shape.token_count())


def verify_cuda_graph_metadata_copy() -> None:
    import dataclasses

    import torch
    from ltx_core.model.transformer.cudagraph_capture import _clone_value, _copy_value, _shape_key
    from ltx_core.types import TemporalAttentionBias

    original = TemporalAttentionBias(torch.tensor([[0.0, -0.5, -1.0]]), 3, 4, 2)
    cloned = _clone_value(original)
    assert isinstance(cloned, TemporalAttentionBias)
    assert cloned.query_log_weights.data_ptr() != original.query_log_weights.data_ptr()
    replacement = TemporalAttentionBias(torch.tensor([[-2.0, -1.5, 0.0]]), 3, 4, 2)
    _copy_value(cloned, replacement)
    torch.testing.assert_close(cloned.query_log_weights, replacement.query_log_weights)
    assert (cloned.key_start, cloned.key_end, cloned.query_chunk_size) == (3, 4, 2)

    @dataclasses.dataclass
    class GraphArgs:
        temporal_attention_bias: TemporalAttentionBias | None

    masks = torch.zeros(2, dtype=torch.bool)
    omitted = _shape_key(GraphArgs(None), None, masks)
    enabled = _shape_key(GraphArgs(original), None, masks)
    assert omitted != enabled
    assert enabled != _shape_key(GraphArgs(None), None, masks)
    assert enabled != _shape_key(
        GraphArgs(TemporalAttentionBias(torch.zeros(1, 4), 3, 4, 2)), None, masks
    )
    assert enabled != _shape_key(
        GraphArgs(TemporalAttentionBias(torch.zeros(1, 3), 2, 4, 2)), None, masks
    )
    assert enabled != _shape_key(
        GraphArgs(TemporalAttentionBias(torch.zeros(1, 3), 3, 5, 2)), None, masks
    )
    assert enabled != _shape_key(
        GraphArgs(TemporalAttentionBias(torch.zeros(1, 3), 3, 4, 7)), None, masks
    )
    updated_weights = TemporalAttentionBias(torch.tensor([[1.0, 0.5, 0.0]]), 3, 4, 2)
    assert enabled == _shape_key(GraphArgs(updated_weights), None, masks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--skip-runner-import", action="store_true")
    args = parser.parse_args()
    add_runtime_paths(args.runtime)
    verify_conditioning_shapes()
    verify_attention("float32")
    verify_attention("bfloat16")
    verify_cuda_graph_metadata_copy()
    if not args.skip_runner_import:
        verify_runner(args.runtime)
    print("LTX25_TEMPORAL_ATTENTION_CPU_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
