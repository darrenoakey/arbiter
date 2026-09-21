"""Qwen-Image-2.1 — unified text-to-image and reference-image editing adapter.

Sanctioned still-image exception #2 (owner decision, 2026-09-21): a local
``Qwen/Qwen-Image-2.1`` adapter (7B single-stream DiT + Qwen3-VL text encoder,
33 GB bf16) exposed through the ``qwen-image`` job type. Unlike the
reference-only FLUX editor (exception #1), this model legitimately generates
images from text AND edits/conditions on input images in one unified pipeline:
``prompt`` alone is text-to-image; ``prompt`` + ``image`` is editing,
enhancement, or multi-reference composition. Scope, enforced in
``image_policy.py`` and the Go ``model_policy.go``:

* the adapter serves ONLY the ``qwen-image`` job type and the
  ``qwen-image-2.1`` model id — no legacy ``image-generate``/``image-edit``
  path, and no other model id may claim the job type;
* it is not wired into ``generate_image`` / ``daz-agent-sdk`` / the Mac mini
  Codex IGS route;
* every ``qwen-image-*`` alias except this exact id (e.g. a LoRA variant)
  stays denied.

Sanctioned still-image exception #3 (owner decision, 2026-09-22):
``qwen-image-2.1-heretic`` / job type ``qwen-image-heretic`` — the identical
pipeline with the stock Qwen3-VL text encoder swapped for the community
abliterated (``heretic``) text encoder (``pottokao/Qwen-Image-2.1-Text-
Encoder-Heretic``). Same DiT, same venv, same params; only the text encoder
differs. The DiT weights are the stock ``Qwen/Qwen-Image-2.1`` snapshot
symlinked under ``/mnt/t9/models/qwen-image-2.1-heretic`` on spark.

Runs in the isolated ``venvs/qwenimage`` environment (torch 2.12 cu130,
transformers 5.17, diffusers pinned from git because ``QwenImage21Pipeline``
is not in a released diffusers yet) via ``worker_cmd``. ``diffusers`` and
``torch`` are imported lazily inside ``load()`` so the main-venv registry
smoke test still passes. See requirements/qwen-image.txt for recreation.

Sampling notes (from the model card):
* the model is meant to be sampled WITHOUT classifier-free guidance —
  ``true_cfg_scale`` stays 1.0 unless a negative prompt justifies raising it;
* transparent RGBA output is prompt-driven (no separate flag);
* supported aspect ratios cap the long side at 2752 px.
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

from arbiter.adapters.base import HeapTrimGuard, InferenceError, ModelAdapter
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

QWEN_IMAGE_21_HF_ID = "Qwen/Qwen-Image-2.1"
QWEN_IMAGE_21_HERETIC_MODEL_ID = "qwen-image-2.1-heretic"
# Local merged pipeline dir on spark: the stock Qwen-Image-2.1 snapshot with
# text_encoder/ replaced by pottokao/Qwen-Image-2.1-Text-Encoder-Heretic.
QWEN_IMAGE_21_HERETIC_PATH = "/mnt/t9/models/qwen-image-2.1-heretic"
DEFAULT_STEPS = 40  # card default
MAX_STEPS = 60
MAX_SIDE = 2752  # largest supported aspect-ratio side (16:9 / 9:16)
MIN_SIDE = 256


class _PipelineResult(Protocol):
    images: list["Image.Image"]


class _Image21Pipeline(Protocol):
    def __call__(self, **kwargs: object) -> _PipelineResult: ...


def snap_side(value: int) -> int:
    """Clamp a pixel side to the supported range and snap to /16."""
    value = max(MIN_SIDE, min(MAX_SIDE, int(value)))
    value = max(MIN_SIDE, (value // 16) * 16)
    return value


@register
class QwenImage21Adapter(ModelAdapter):
    model_id = "qwen-image-2.1"
    #: HF repo id (stock) or local merged pipeline dir (heretic subclass).
    hf_model_path: str = QWEN_IMAGE_21_HF_ID

    def __init__(self):
        self._pipe: _Image21Pipeline | None = None
        self._device = "cuda"

    def load(self, device: str = "cuda") -> None:
        import torch

        diffusers = importlib.import_module("diffusers")
        QwenImage21Pipeline = diffusers.QwenImage21Pipeline

        log.info("Loading %s on %s ...", self.hf_model_path, device)
        with HeapTrimGuard():
            self._pipe = cast(
                _Image21Pipeline,
                QwenImage21Pipeline.from_pretrained(
                    self.hf_model_path,
                    torch_dtype=torch.bfloat16,
                ),
            )
            # GB10 mmap->cuda workaround: clone each tensor off mmap-backed
            # storage before moving to device. See base._pipe_to_cuda_cloned.
            self._pipe_to_cuda_cloned(self._pipe, device)
        self._device = device
        log.info("%s ready.", QWEN_IMAGE_21_HF_ID)

    def unload(self) -> None:
        log.info("Unloading %s.", self.hf_model_path)
        self._pipe = None
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        import torch

        self._check_cancel(cancel_flag)
        if self._pipe is None:
            raise InferenceError("qwen-image-2.1 pipeline is not loaded")

        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            raise InferenceError("prompt is required")

        has_image = bool(params.get("image") or params.get("image_file"))
        input_image = self._resolve_image(params) if has_image else None

        steps = max(1, min(MAX_STEPS, int(params.get("steps", DEFAULT_STEPS))))
        true_cfg = float(params.get("true_cfg_scale", 1.0))
        if true_cfg <= 1.0 and params.get("negative_prompt"):
            # negative_prompt only has effect alongside true_cfg_scale > 1
            true_cfg = max(true_cfg, 2.0)
        negative_prompt = params.get("negative_prompt") or None
        seed = int(params.get("seed", 42))
        output_resolution = max(
            MIN_SIDE, min(MAX_SIDE, int(params.get("output_resolution", 1024)))
        )

        width = params.get("width")
        height = params.get("height")
        if width is not None and height is not None:
            width, height = snap_side(int(width)), snap_side(int(height))
        elif input_image is None:
            # Plain text-to-image without explicit dimensions: square at the
            # requested output resolution.
            width = height = snap_side(output_resolution)
        else:
            # Editing/conditioning: let the pipeline derive height/width from
            # the condition image's aspect ratio at the target resolution.
            width = height = None

        call_kwargs: dict = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "true_cfg_scale": true_cfg,
            "generator": torch.Generator(device=self._device).manual_seed(seed),
        }
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt
        if input_image is not None:
            call_kwargs["image"] = input_image
            call_kwargs["output_resolution"] = output_resolution
        if width is not None and height is not None:
            call_kwargs["width"] = width
            call_kwargs["height"] = height

        self._check_cancel(cancel_flag)
        result_image = self._pipe(**call_kwargs).images[0]
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
            "mode": "edit" if input_image is not None else "t2i",
            "width": result_image.width,
            "height": result_image.height,
            "steps": steps,
            "seed": seed,
            "true_cfg_scale": true_cfg,
            "model": self.hf_model_path,
            "file": "result.png",
        }

    def estimate_time(self, params: dict) -> float:
        steps = int(params.get("steps", DEFAULT_STEPS))
        edit = 1 if (params.get("image") or params.get("image_file")) else 0
        # 7B DiT at ~1-4 MP; conditioning images add text-encoder vision tokens
        return 6000.0 + 350.0 * steps + 1500.0 * edit


@register
class QwenImage21HereticAdapter(QwenImage21Adapter):
    """Abliterated-text-encoder variant of the sanctioned Qwen-Image-2.1
    pipeline (exception #3, owner decision 2026-09-22).

    Identical DiT/VAE and sampling behavior to :class:`QwenImage21Adapter`;
    only the text encoder differs (community ``heretic`` abliteration of the
    Qwen3-VL-8B text encoder, which removes the prompt-level censorship baked
    into the stock text encoder). Serves ONLY the ``qwen-image-heretic`` job
    type under the ``qwen-image-2.1-heretic`` model id; every other
    ``qwen-image-*`` id stays denied.
    """

    model_id = QWEN_IMAGE_21_HERETIC_MODEL_ID
    hf_model_path = QWEN_IMAGE_21_HERETIC_PATH
