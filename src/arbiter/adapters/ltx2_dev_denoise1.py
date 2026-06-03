"""LTX-2 DEV denoise1 adapter — non-distilled stage-1 transformer.

Identical to ltx2-denoise1 (same video encoder + upsampler + stage-1 denoise
loop, same 30-step LTX2Scheduler schedule) — the ONLY difference is the stage-1
transformer CHECKPOINT: this loads the full non-distilled `dev` model instead of
the distilled one. At the same step count and render time, the dev model
produces dramatically sharper, more detailed output (the distillation was what
softened everything).

This runs in its own worker process (model_id "ltx2-dev-denoise1"), so
overriding the module-global CHECKPOINT here cannot affect the distilled
ltx2-denoise1 adapter — both remain available side by side.
"""
from __future__ import annotations

import sys
from pathlib import Path

from arbiter.adapters.base import LoadError
from arbiter.adapters.ltx2_denoise1 import LTX2_SPARK_DIR, LTX2Denoise1Adapter
from arbiter.adapters.registry import register

DEV_CHECKPOINT_NAME = "ltx-2.3-22b-dev.safetensors"


@register
class LTX2DevDenoise1Adapter(LTX2Denoise1Adapter):
    """Dev (non-distilled) variant of ltx2-denoise1."""

    model_id = "ltx2-dev-denoise1"

    def load(self, device: str = "cuda") -> None:
        spark_str = str(LTX2_SPARK_DIR)
        if spark_str not in sys.path:
            sys.path.insert(0, spark_str)
        # Point the pipeline at the dev (non-distilled) transformer BEFORE
        # FastPipeline() reads it. Set both the constants module and the
        # already-bound name in video_fast_gpu so it takes regardless of import
        # order within this worker process.
        import constants
        dev = constants.MODELS_DIR / DEV_CHECKPOINT_NAME
        if not Path(dev).exists():
            raise LoadError(f"dev checkpoint missing: {dev}")
        constants.CHECKPOINT = dev
        import video_fast_gpu
        video_fast_gpu.CHECKPOINT = dev
        super().load(device)
