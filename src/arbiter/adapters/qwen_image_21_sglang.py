"""Qwen-Image-2.1 through SGLang-Diffusion, inside the arbiter worker.

Sanctioned still-image exception #4 (owner decision, 2026-09-29): the same
``Qwen/Qwen-Image-2.1`` checkpoint the Diffusers adapter serves, loaded by
SGLang's resident BF16 pipeline (Torch SDPA, eager, no Cache-DiT, no
quantization). Exposed only through the ``qwen-image-sglang`` job type and
the ``qwen-image-2.1-sglang`` model id.

SGLang's local generator spawns a scheduler process. That child does not
inherit ``torch.cuda.set_per_process_memory_fraction``, so this venv's
``sitecustomize.py`` must apply the cap from ``ARBITER_MEMORY_GB`` before
any CUDA allocation. ``load()`` refuses to start the child if that file is
missing. See ``requirements/qwen-image-sglang.txt``.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
from pathlib import Path

from arbiter.adapters.base import InferenceError, ModelAdapter
from arbiter.adapters.qwen_image_21 import (
    DEFAULT_STEPS,
    MAX_SIDE,
    MAX_STEPS,
    MIN_SIDE,
    snap_side,
)
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

QWEN_IMAGE_21_SGLANG_MODEL_ID = "qwen-image-2.1-sglang"
QWEN_IMAGE_21_SGLANG_HF_ID = "Qwen/Qwen-Image-2.1"


def _child_cuda_cap_installed() -> bool:
    """True when this interpreter's sitecustomize applies the arbiter CUDA cap."""
    site = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "sitecustomize.py"
    )
    try:
        text = site.read_text(encoding="utf-8")
    except OSError:
        return False
    return "set_per_process_memory_fraction" in text and "ARBITER_MEMORY_GB" in text


@register
class QwenImage21SglangAdapter(ModelAdapter):
    """Resident SGLang BF16 generator for Qwen-Image-2.1."""

    model_id = QWEN_IMAGE_21_SGLANG_MODEL_ID

    def __init__(self) -> None:
        self._gen: object | None = None

    def load(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise InferenceError("qwen-image-2.1-sglang requires cuda")
        if os.environ.get("ARBITER_MEMORY_GB") and not _child_cuda_cap_installed():
            raise InferenceError(
                "sglang venv is missing the CUDA memory cap sitecustomize; "
                "refusing to spawn an uncapped scheduler"
            )
        # The public convenience import is ``sglang.multimodal_gen``. Import
        # the pinned entrypoint directly: it keeps the worker's dependency
        # boundary explicit and is the path present in SGLang 7bcfcf5aa.
        entrypoint = importlib.import_module(
            "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator"
        )
        diff_generator = entrypoint.DiffGenerator

        log.info("Loading %s via SGLang on %s", QWEN_IMAGE_21_SGLANG_HF_ID, device)
        self._gen = diff_generator.from_pretrained(
            model_path=QWEN_IMAGE_21_SGLANG_HF_ID,
            num_gpus=1,
            performance_mode="speed",
            attention_backend="torch_sdpa",
        )
        log.info("%s ready.", QWEN_IMAGE_21_SGLANG_MODEL_ID)

    def unload(self) -> None:
        log.info("Unloading %s.", QWEN_IMAGE_21_SGLANG_MODEL_ID)
        gen = self._gen
        self._gen = None
        if gen is not None:
            shutdown = getattr(gen, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    log.exception("SGLang generator shutdown failed")
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        self._check_cancel(cancel_flag)
        if self._gen is None:
            raise InferenceError("qwen-image-2.1-sglang pipeline is not loaded")

        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            raise InferenceError("prompt is required")

        steps = max(1, min(MAX_STEPS, int(params.get("steps", DEFAULT_STEPS))))
        true_cfg = float(params.get("true_cfg_scale", 1.0))
        if true_cfg <= 1.0 and params.get("negative_prompt"):
            true_cfg = max(true_cfg, 2.0)
        negative_prompt = params.get("negative_prompt") or None
        seed = int(params.get("seed", 42))
        output_resolution = max(
            MIN_SIDE, min(MAX_SIDE, int(params.get("output_resolution", 1024)))
        )
        image_path = self._condition_path(params, output_dir)
        width = params.get("width")
        height = params.get("height")
        if width is not None and height is not None:
            width, height = snap_side(int(width)), snap_side(int(height))
        elif image_path is None:
            width = height = snap_side(output_resolution)
        else:
            width = height = None

        call: dict = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": 1.0 if true_cfg <= 1.0 else true_cfg,
            "true_cfg_scale": true_cfg,
            "seed": seed,
            "save_output": True,
            "output_path": str(output_dir),
            "output_file_name": "result.png",
        }
        if negative_prompt:
            call["negative_prompt"] = negative_prompt
        if image_path is not None:
            call["image_path"] = image_path
        if width is not None and height is not None:
            call["width"] = width
            call["height"] = height

        self._check_cancel(cancel_flag)
        result = self._gen.generate(sampling_params_kwargs=call)  # type: ignore[attr-defined]
        self._check_cancel(cancel_flag)
        out_path = self._durable_png(result, output_dir)
        from PIL import Image

        with Image.open(out_path) as saved:
            saved.load()
            pixel_w, pixel_h = saved.size
        return {
            "format": "png",
            "mode": "edit" if image_path is not None else "t2i",
            "width": pixel_w,
            "height": pixel_h,
            "steps": steps,
            "seed": seed,
            "true_cfg_scale": true_cfg,
            "model": QWEN_IMAGE_21_SGLANG_HF_ID,
            "runtime": "sglang",
            "file": "result.png",
        }

    def _condition_path(self, params: dict, output_dir: Path) -> str | None:
        image_file = params.get("image_file")
        if image_file:
            return str(image_file)
        if not params.get("image"):
            return None
        image = self._resolve_image(params)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "condition.png"
        image.save(path)
        return str(path)

    def _durable_png(self, result: object, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "result.png"
        written = getattr(result, "output_file_path", None)
        if written and Path(written).is_file() and Path(written) != out_path:
            Path(written).replace(out_path)
        if not out_path.is_file():
            sample = getattr(result, "samples", None)
            frames = getattr(result, "frames", None)
            image = sample if sample is not None else (frames[0] if frames else None)
            if image is None or not hasattr(image, "save"):
                raise InferenceError("sglang returned no image")
            image.save(out_path, format="PNG")
        with open(out_path, "rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        try:
            dir_fd = os.open(str(output_dir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
        if out_path.stat().st_size < 1000:
            raise InferenceError("sglang wrote an empty image")
        return out_path

    def estimate_time(self, params: dict) -> float:
        steps = int(params.get("steps", DEFAULT_STEPS))
        edit = 1 if (params.get("image") or params.get("image_file")) else 0
        # Published DGX Spark median is ~35s T2I / ~42s edit at 1024, 40 steps.
        return 4000.0 + 200.0 * steps + 1200.0 * edit
