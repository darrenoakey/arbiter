"""Unconditional owner policy disabling still-image generation in Arbiter."""

from __future__ import annotations

import re

STILL_IMAGE_DISABLED_MESSAGE = (
    "still-image generation is actively disabled in Arbiter; "
    "callers must use the Mac mini Codex image service"
)

_DISABLED_MARKERS = (
    "flux",
    "kontext",
    "z-image",
    "zimage",
    "stable-diffusion",
    "stablediffusion",
    "sdxl",
    "sd3",
    "sd-3",
    "pixart",
    "kandinsky",
    "aura-flow",
    "auraflow",
    "playground-v",
    "ideogram",
    "recraft",
    "hidream",
    "hunyuan-image",
    "qwen-image",
    "kolors",
    "omnigen",
    "dreamshaper",
    "realvis",
    "juggernaut",
    "image-generator",
)


class StillImageGenerationDisabled(RuntimeError):
    """Raised before any disabled still-image model can load or infer."""


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower())


def is_disabled_still_image_model(model_id: str) -> bool:
    """Return whether a model identifier belongs to a still-image generator."""
    normalized = _normalize(model_id)
    if not normalized:
        return False
    if normalized == "lora-train" or normalized.startswith("ltx2-"):
        return False
    return "lora" in normalized.split("-") or any(
        marker in normalized for marker in _DISABLED_MARKERS
    )


def require_still_image_disabled(model_id: str) -> None:
    """Fail closed for disabled adapters before importing ML frameworks."""
    if is_disabled_still_image_model(model_id):
        raise_still_image_disabled()


def raise_still_image_disabled() -> None:
    """Unconditionally stop a retained still-image adapter at its boundary."""
    raise StillImageGenerationDisabled(STILL_IMAGE_DISABLED_MESSAGE)
