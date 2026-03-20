"""FLUX.1-schnell text/image generation adapter."""
from __future__ import annotations

import base64
import io
import logging
import threading
from pathlib import Path

from arbiter.adapters.base import ModelAdapter, InferenceError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

FLUX_HF_ID = "black-forest-labs/FLUX.1-schnell"
DEFAULT_STEPS = 4

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "3:4": (768, 1024),
    "4:3": (1024, 768),
    "9:16": (576, 1024),
    "16:9": (1024, 576),
    "3:2": (1024, 682),
    "2:3": (682, 1024),
}


@register
class FluxSchnellAdapter(ModelAdapter):
    model_id = "flux-schnell"

    def __init__(self):
        self._pipe = None
        self._img2img_pipe = None

    def load(self, device: str = "cuda") -> None:
        import torch
        from diffusers import DiffusionPipeline

        log.info("Loading FLUX.1-schnell on %s ...", device)
        self._pipe = DiffusionPipeline.from_pretrained(
            FLUX_HF_ID,
            torch_dtype=torch.bfloat16,
        ).to(device)

        try:
            self._pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        self._device = device
        log.info("FLUX.1-schnell ready.")

    def unload(self) -> None:
        log.info("Unloading FLUX.1-schnell.")
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
        from diffusers import FluxImg2ImgPipeline

        log.info("Loading FLUX.1-schnell img2img pipeline ...")
        self._img2img_pipe = FluxImg2ImgPipeline.from_pretrained(
            FLUX_HF_ID,
            torch_dtype=torch.bfloat16,
        ).to(self._device)

        try:
            self._img2img_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        return self._img2img_pipe

    def _resolve_dimensions(self, params: dict) -> tuple[int, int]:
        aspect_ratio = params.get("aspect_ratio")
        if aspect_ratio and aspect_ratio in ASPECT_RATIOS:
            return ASPECT_RATIOS[aspect_ratio]
        width = int(params.get("width", 1024))
        height = int(params.get("height", 1024))
        return width, height

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        import torch
        from PIL import Image

        self._check_cancel(cancel_flag)

        prompt = params.get("prompt", "")
        steps = int(params.get("steps", DEFAULT_STEPS))
        seed = int(params.get("seed", 42))
        width, height = self._resolve_dimensions(params)
        transparent = str(params.get("transparent", "false")).lower() == "true"

        generator = torch.Generator(device=self._device).manual_seed(seed)

        has_input_image = "image" in params and params["image"]

        self._check_cancel(cancel_flag)

        if has_input_image:
            # Image-to-image
            image_b64 = params["image"]
            if image_b64.startswith("data:"):
                _, image_b64 = image_b64.split(",", 1)
            try:
                input_image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
            except Exception as e:
                raise InferenceError(f"Failed to decode input image: {e}")

            pipe = self._get_img2img_pipe()
            strength = float(params.get("strength", 0.75))
            result_image = pipe(
                prompt=prompt,
                image=input_image.resize((width, height)),
                strength=strength,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]
        else:
            # Text-to-image
            result_image = self._pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]

        self._check_cancel(cancel_flag)

        if transparent:
            # Defer to BiRefNet if available, otherwise skip
            try:
                from arbiter.adapters.birefnet import BiRefNetAdapter
                # Inline background removal is not supported here;
                # the caller should chain a birefnet job instead.
                log.warning("transparent=true requested but should be handled as a separate birefnet job")
            except ImportError:
                pass

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
        return float(steps) * 500.0
