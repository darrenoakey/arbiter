"""LTX-2 DEV denoise2 adapter — non-distilled stage-2 refine.

Identical to ltx2-denoise2 (same 11-step stage-2 refine + distilled LoRA + VAE
decode) — the ONLY difference is the transformer CHECKPOINT: the full
non-distilled `dev` model instead of the distilled one. Pairs with
ltx2-dev-denoise1 to render at the proven sharper dev quality.

Own worker process (model_id "ltx2-dev-denoise2"); overriding the module-global
CHECKPOINT here does not affect the distilled ltx2-denoise2 adapter.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from arbiter.adapters.base import LoadError
from arbiter.adapters.ltx2_denoise2 import LTX2_SPARK_DIR, LTX2Denoise2Adapter
from arbiter.adapters.registry import register

DEV_CHECKPOINT_NAME = "ltx-2.3-22b-dev.safetensors"


@register
class LTX2DevDenoise2Adapter(LTX2Denoise2Adapter):
    """Dev (non-distilled) variant of ltx2-denoise2."""

    model_id = "ltx2-dev-denoise2"

    def load(self, device: str = "cuda") -> None:
        spark_str = str(LTX2_SPARK_DIR)
        if spark_str not in sys.path:
            sys.path.insert(0, spark_str)
        constants = importlib.import_module("constants")

        dev = constants.MODELS_DIR / DEV_CHECKPOINT_NAME
        if not Path(dev).exists():
            raise LoadError(f"dev checkpoint missing: {dev}")
        setattr(constants, "CHECKPOINT", dev)
        video_fast_gpu = importlib.import_module("video_fast_gpu")

        setattr(video_fast_gpu, "CHECKPOINT", dev)
        LTX2Denoise2Adapter.load(self, device)
