"""FLUX.2-klein-9B unified generation + editing adapter.

black-forest-labs/FLUX.2-klein-9B (FLUX non-commercial license, gated —
access accepted for this account): step-distilled (4 steps, guidance 1.0)
9B transformer with an integrated ~8B Qwen3 text encoder. A SINGLE
Flux2KleinPipeline handles both text-to-image and image editing
(image-to-image / reference editing) — pass ``image=`` to edit. The
smaller ungated klein-4B is the secondary choice if 9B access is ever revoked.

Legacy disabled adapter that ran in the isolated ``venvs/flux2`` environment
(torch 2.12 / git diffusers / transformers 5.x) via an old ``worker_cmd`` entry
config, so the bleeding-edge stack never touches the shared arbiter
.venv. ``diffusers`` is imported lazily inside load()/_get_pipe so the
main-venv adapter-registry smoke test (deploy-to-spark.sh) still passes.
"""

from __future__ import annotations

import logging
import importlib
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from PIL import Image

from arbiter.adapters.base import ModelAdapter
from arbiter.adapters.registry import register
from arbiter.image_policy import raise_still_image_disabled

log = logging.getLogger(__name__)

FLUX2_HF_ID = "black-forest-labs/FLUX.2-klein-9B"
DEFAULT_STEPS = 4  # klein is step-distilled
GUIDANCE_SCALE = 1.0  # klein recommended guidance


class _PipelineResult(Protocol):
    images: list["Image.Image"]


class _ImagePipeline(Protocol):
    def __call__(self, **kwargs: object) -> _PipelineResult: ...
    def enable_xformers_memory_efficient_attention(self) -> None: ...


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
class Flux2KleinAdapter(ModelAdapter):
    model_id = "flux2"

    def __init__(self):
        self._pipe: _ImagePipeline | None = None
        self._device = "cuda"

    def load(self, device: str = "cuda") -> None:
        raise_still_image_disabled()
        import torch

        Flux2KleinPipeline = importlib.import_module("diffusers").Flux2KleinPipeline

        log.info("Loading %s on %s ...", FLUX2_HF_ID, device)
        self._pipe = cast(
            _ImagePipeline,
            Flux2KleinPipeline.from_pretrained(
                FLUX2_HF_ID,
                torch_dtype=torch.bfloat16,
            ),
        )
        # GB10 mmap->cuda workaround: clone each tensor off mmap-backed
        # storage before moving to device. Same fix flux/z-image rely on;
        # NOT enable_model_cpu_offload (GB10 has 128GB unified VRAM and we
        # want the whole pipeline resident). See base._pipe_to_cuda_cloned.
        self._pipe_to_cuda_cloned(self._pipe, device)

        try:
            self._pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        self._device = device
        log.info("FLUX.2-klein-9B ready.")

    def unload(self) -> None:
        log.info("Unloading FLUX.2-klein-9B.")
        del self._pipe
        self._pipe = None
        self._cleanup_gpu()

    def _resolve_dimensions(self, params: dict) -> tuple[int, int]:
        aspect_ratio = params.get("aspect_ratio")
        if aspect_ratio and aspect_ratio in ASPECT_RATIOS:
            return ASPECT_RATIOS[aspect_ratio]
        width = int(params.get("width", 1024))
        height = int(params.get("height", 1024))
        return width, height

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        raise_still_image_disabled()
        import torch

        self._check_cancel(cancel_flag)

        prompt = params.get("prompt", "")
        steps = int(params.get("steps", DEFAULT_STEPS))
        seed = int(params.get("seed", 42))
        width, height = self._resolve_dimensions(params)
        guidance = float(params.get("guidance_scale", GUIDANCE_SCALE))

        generator = torch.Generator(device=self._device).manual_seed(seed)
        has_input_image = bool(params.get("image") or params.get("image_file"))

        self._check_cancel(cancel_flag)

        pipe = self._pipe
        if pipe is None:
            raise RuntimeError("flux2 not loaded")
        if has_input_image:
            # Unified editing path — same pipeline, pass image=.
            input_image = self._resolve_image(params)
            result_image = pipe(
                prompt=prompt,
                image=input_image,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            ).images[0]
        else:
            result_image = pipe(
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
        # fsync data + dir so bytes are durable on the CIFS share before we
        # return — arbiter's queue-priority eviction may SIGKILL the worker
        # ~1s after we return (cache=strict drops unsynced writes). Same
        # durability guard as z_image.py.
        with open(out_path, "wb") as out_fh:
            result_image.save(out_fh, format="PNG")
            out_fh.flush()
            os.fsync(out_fh.fileno())
        try:
            dir_fd = os.open(str(output_dir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

        return {
            "format": "png",
            "width": result_image.width,
            "height": result_image.height,
            "steps": steps,
            "seed": seed,
            "file": "result.png",
        }

    def estimate_time(self, params: dict) -> float:
        steps = int(params.get("steps", DEFAULT_STEPS))
        return float(steps) * 1500.0
