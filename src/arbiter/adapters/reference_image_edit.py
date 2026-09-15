"""Reference-conditioned local image editor (FLUX.2-klein-9B) — reference renders only.

This is the ONE sanctioned exception to Arbiter's still-image policy (owner
decision, 2026-09-16). Its purpose is to render *reference targets* for other
pipelines — e.g. "what would a professional DSLR shot of this exact photo look
like" — that a per-segment photometric grading loop then steers the ORIGINAL
pixels toward. The render itself is never the deliverable: FLUX redraws faces.

Scope, enforced here and in the Go policy (`referenceImageEditModel`):

* an input image is REQUIRED — there is no text-to-image path;
* it is never wired into ``generate_image``, ``daz-agent-sdk`` or the Mac mini
  Codex IGS route, and must never be used as a fallback when IGS is down;
* the legacy ``image-generate`` / ``image-edit`` job types and every other
  still-image adapter stay disabled.

Runs in the isolated ``venvs/flux2`` environment (torch 2.12 / diffusers 0.39)
via ``worker_cmd`` so the bleeding-edge stack never touches the shared .venv.
``diffusers`` is imported lazily inside ``load()`` so the main-venv registry
smoke test still passes.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from PIL import Image

from arbiter.adapters.base import InferenceError, ModelAdapter
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

REFERENCE_EDIT_HF_ID = "black-forest-labs/FLUX.2-klein-9B"
DEFAULT_STEPS = 4  # klein is step-distilled
DEFAULT_GUIDANCE = 1.0  # klein recommended guidance
MAX_STEPS = 50
MAX_SIDE = 1536  # klein is trained around 1MP; larger inputs are downscaled


class _PipelineResult(Protocol):
    images: list["Image.Image"]


class _EditPipeline(Protocol):
    def __call__(self, **kwargs: object) -> _PipelineResult: ...


def fit_within(width: int, height: int, max_side: int = MAX_SIDE) -> tuple[int, int]:
    """Scale (width, height) to fit max_side and snap to multiples of 16."""
    scale = min(1.0, max_side / max(width, height))
    w = max(16, int(round(width * scale / 16)) * 16)
    h = max(16, int(round(height * scale / 16)) * 16)
    return w, h


@register
class ReferenceImageEditAdapter(ModelAdapter):
    model_id = "reference-image-edit"

    def __init__(self):
        self._pipe: _EditPipeline | None = None
        self._device = "cuda"

    def load(self, device: str = "cuda") -> None:
        import torch

        Flux2KleinPipeline = importlib.import_module("diffusers").Flux2KleinPipeline

        log.info("Loading %s on %s ...", REFERENCE_EDIT_HF_ID, device)
        self._pipe = cast(
            _EditPipeline,
            Flux2KleinPipeline.from_pretrained(
                REFERENCE_EDIT_HF_ID,
                torch_dtype=torch.bfloat16,
            ),
        )
        # GB10 mmap->cuda workaround: clone each tensor off mmap-backed
        # storage before moving to device. See base._pipe_to_cuda_cloned.
        self._pipe_to_cuda_cloned(self._pipe, device)
        self._device = device
        log.info("%s ready.", REFERENCE_EDIT_HF_ID)

    def unload(self) -> None:
        log.info("Unloading %s.", REFERENCE_EDIT_HF_ID)
        self._pipe = None
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        import torch

        self._check_cancel(cancel_flag)
        if self._pipe is None:
            raise InferenceError("reference-image-edit pipeline is not loaded")

        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            raise InferenceError("prompt is required")
        if not (params.get("image") or params.get("image_file")):
            raise InferenceError(
                "reference-image-edit requires an input image; it is not a text-to-image generator"
            )
        input_image = self._resolve_image(params)

        steps = max(1, min(MAX_STEPS, int(params.get("steps", DEFAULT_STEPS))))
        guidance = float(params.get("guidance_scale", DEFAULT_GUIDANCE))
        seed = int(params.get("seed", 42))
        width, height = fit_within(input_image.width, input_image.height)
        if (width, height) != input_image.size:
            input_image = input_image.resize((width, height))

        generator = torch.Generator(device=self._device).manual_seed(seed)
        self._check_cancel(cancel_flag)
        result_image = self._pipe(
            prompt=prompt,
            image=input_image,
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
        # return — queue-priority eviction may SIGKILL the worker ~1s after
        # we return (cache=strict drops unsynced writes).
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
            "guidance_scale": guidance,
            "model": REFERENCE_EDIT_HF_ID,
            "file": "result.png",
        }

    def estimate_time(self, params: dict) -> float:
        steps = int(params.get("steps", DEFAULT_STEPS))
        return 3000.0 + 1500.0 * steps
