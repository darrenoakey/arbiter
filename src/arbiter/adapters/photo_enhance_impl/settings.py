"""Settings for the spark-side photo-enhance worker.

Mirrors photo-enhance's config.Settings surface (only the fields the
vendored pipeline touches), with spark-specific defaults: the worker calls
arbiter on localhost, stages into the shared inbox, and reads SeedVR2
weights from the checkout on spark.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    arbiter_url: str = os.environ.get("ARBITER_BASE_URL", "http://127.0.0.1:8400")
    # staging: on spark the inbox mount IS local disk.
    inbox_local: str = os.environ.get("ARBITER_INBOX_PATH", "/mnt/arbiter-store/inbox")
    inbox_remote: str = os.environ.get("ARBITER_INBOX_SPARK_PATH", "/mnt/arbiter-store/inbox")
    output_dir: str = os.environ.get("ARBITER_OUTPUT_PATH", "/mnt/arbiter-store/output")
    cache_dir: str = os.environ.get("PHOTO_ENHANCE_CACHE", "/home/darren/local/photo-enhance-cache")
    # climb conversation model: served by arbiter on spark.
    vision_model: str = os.environ.get("PHOTO_ENHANCE_VISION_MODEL", "llm:qwen3-vl-8b-fp8")
    vision_url: str = ""  # unused; the chat goes through arbiter_url
    # SeedVR2 (official ByteDance-Seed/SeedVR checkout + 3B checkpoint).
    seedvr2_home: str = os.environ.get("SEEDVR_HOME", "/home/darren/src/seedvr")
    seedvr2_ckpt: str = os.environ.get("SEEDVR_CKPT", "/home/darren/src/seedvr/ckpts/seedvr2_ema_3b.pth")

    @property
    def seedvr2_lock(self) -> str:  # kept for interface parity; unused on spark
        return str(Path(self.cache_dir) / "seedvr2.lock")

    @property
    def mflux_python(self) -> str:  # unused on spark; SeedVR2 runs in-process on CUDA
        return ""
