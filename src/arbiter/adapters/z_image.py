"""Z-Image-Turbo text/image generation adapter.

Tongyi-MAI/Z-Image-Turbo: 6B parameter distilled model with strong
prompt adherence and good img2img performance at low step counts.
Requires guidance_scale=0.0 and num_inference_steps=9 (8 NFEs).
"""
from __future__ import annotations

import base64
import io
import logging
import threading
from pathlib import Path

from arbiter.adapters.base import ModelAdapter, InferenceError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

ZIMAGE_HF_ID = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_STEPS = 9  # 9 steps = 8 NFEs (recommended for turbo)
GUIDANCE_SCALE = 0.0  # Must be 0 for turbo distilled models


@register
class ZImageTurboAdapter(ModelAdapter):
    model_id = "z-image-turbo"

    def __init__(self):
        self._pipe = None
        self._img2img_pipe = None
        self._device = "cuda"

    def load(self, device: str = "cuda") -> None:
        import torch
        from diffusers import ZImagePipeline

        log.info("Loading Z-Image-Turbo on %s ...", device)
        self._pipe = ZImagePipeline.from_pretrained(
            ZIMAGE_HF_ID,
            torch_dtype=torch.bfloat16,
        ).to(device)

        self._device = device
        log.info("Z-Image-Turbo ready.")

    def unload(self) -> None:
        log.info("Unloading Z-Image-Turbo.")
        del self._pipe
        self._pipe = None
        if self._img2img_pipe is not None:
            del self._img2img_pipe
            self._img2img_pipe = None
        self._cleanup_gpu()

    def _get_img2img_pipe(self):
        if self._img2img_pipe is not None:
            return self._img2img_pipe
        import torch
        from diffusers import ZImageImg2ImgPipeline

        log.info("Loading Z-Image-Turbo img2img pipeline ...")
        self._img2img_pipe = ZImageImg2ImgPipeline.from_pretrained(
            ZIMAGE_HF_ID,
            torch_dtype=torch.bfloat16,
        ).to(self._device)

        return self._img2img_pipe

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        import torch

        self._check_cancel(cancel_flag)

        prompt = params.get("prompt", "")
        steps = int(params.get("steps", DEFAULT_STEPS))
        seed = int(params.get("seed", 42))
        width = int(params.get("width", 1024))
        height = int(params.get("height", 1024))
        guidance = float(params.get("guidance_scale", GUIDANCE_SCALE))

        generator = torch.Generator(device=self._device).manual_seed(seed)

        has_input_image = "image" in params and params["image"]

        self._check_cancel(cancel_flag)

        if has_input_image:
            input_image = self._resolve_image(params)
            pipe = self._get_img2img_pipe()
            strength = float(params.get("strength", 0.85))
            result_image = pipe(
                prompt=prompt,
                image=input_image.resize((width, height)),
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            ).images[0]
        else:
            result_image = self._pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            ).images[0]

        self._check_cancel(cancel_flag)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "result.png"
        result_image.save(str(out_path), format="PNG")

        return {
            "format": "png",
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "file": "result.png",
        }

    def estimate_time(self, params: dict) -> float:
        steps = int(params.get("steps", DEFAULT_STEPS))
        return float(steps) * 400.0
