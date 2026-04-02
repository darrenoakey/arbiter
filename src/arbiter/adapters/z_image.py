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
        self._txt2img_backend = "native"
        self._img2img_backend = "native"

    def load(self, device: str = "cuda") -> None:
        import torch
        from diffusers import ZImagePipeline

        log.info("Loading Z-Image-Turbo on %s ...", device)
        self._pipe = ZImagePipeline.from_pretrained(
            ZIMAGE_HF_ID,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self._txt2img_backend = self._enable_fast_attention(self._pipe, pipeline_name="txt2img")

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
        self._img2img_backend = self._enable_fast_attention(self._img2img_pipe, pipeline_name="img2img")

        return self._img2img_pipe

    def _enable_fast_attention(self, pipe, pipeline_name: str) -> str:
        transformer = getattr(pipe, "transformer", None)
        if transformer is None or not hasattr(transformer, "set_attention_backend"):
            log.info("Z-Image-Turbo %s using default attention; transformer backend switch unsupported", pipeline_name)
            return "native"

        for backend in ("_native_cudnn", "native"):
            try:
                transformer.set_attention_backend(backend)
                log.info("Z-Image-Turbo %s enabled attention backend: %s", pipeline_name, backend)
                return backend
            except Exception as exc:
                log.info("Z-Image-Turbo %s attention backend %s unavailable: %s", pipeline_name, backend, exc)

        log.warning("Z-Image-Turbo %s could not enable a preferred attention backend; leaving default in place", pipeline_name)
        return "native"

    def _retry_with_native_attention(self, pipe, pipeline_name: str, exc: Exception):
        transformer = getattr(pipe, "transformer", None)
        if transformer is None or not hasattr(transformer, "set_attention_backend"):
            raise exc

        log.warning(
            "Z-Image-Turbo %s backend failed (%s); retrying with native backend",
            pipeline_name,
            exc,
        )
        transformer.set_attention_backend("native")
        if pipeline_name == "txt2img":
            self._txt2img_backend = "native"
        else:
            self._img2img_backend = "native"

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
            try:
                result_image = pipe(
                    prompt=prompt,
                    image=input_image.resize((width, height)),
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                ).images[0]
            except Exception as exc:
                self._retry_with_native_attention(pipe, "img2img", exc)
                result_image = pipe(
                    prompt=prompt,
                    image=input_image.resize((width, height)),
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                ).images[0]
        else:
            try:
                result_image = self._pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                ).images[0]
            except Exception as exc:
                self._retry_with_native_attention(self._pipe, "txt2img", exc)
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
