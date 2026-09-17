"""SeedVR2 one-step restoration/upscaling for stills on spark CUDA.

Wraps the vendored official ByteDance-Seed/SeedVR inference code
(Apache-2.0, see vendor/LICENSE) for single-image use: a one-frame
"video" at 2x target resolution, matching photo-enhance's detail
pre-pass contract.

Two dependencies of the research code are not viable on Grace Blackwell
aarch64 (no flash-attn wheels, no apex), so they are shimmed before the
vendored modules are imported:

- flash_attn.flash_attn_varlen_func -> torch SDPA over per-sequence
  segments (cu_seqlens-batched). Attention math is identical; speed is
  lower than fused kernels, which is acceptable for a one-photo-at-a-time
  queue.
- apex.normalization FusedLayerNorm/FusedRMSNorm -> torch.nn.LayerNorm /
  torch.nn.RMSNorm (same semantics, unfused).
"""
