"""Ming-Image-0.1-Design text-to-image adapter.

Owner comparison request, 2026-10-11. Design-oriented generator from
``inclusionAI/Ming-Image-0.1-Design``. Not an IGS fallback. Inference code
lives at ``/home/darren/src/Ming-Image``; weights at the NVMe snapshot.
Torch is imported only inside ``load``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from arbiter.adapters.base import InferenceError, ModelAdapter
from arbiter.adapters.registry import register

MING_SRC = Path("/home/darren/src/Ming-Image")
MING_WEIGHTS = Path("/home/darren/nvme-models/Ming-Image-0.1-Design")
DEFAULT_STEPS = 12
DEFAULT_CFG = 1.0


@register
class MingImageDesignAdapter(ModelAdapter):
    """Serves only the ``ming-image-0.1-design`` model id."""

    model_id = "ming-image-0.1-design"

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._profile = None

    def load(self, device: str = "cuda") -> None:
        if str(MING_SRC) not in sys.path:
            sys.path.insert(0, str(MING_SRC))
        from infer import load_model_and_processor
        from inference_profile import load_checkpoint_capabilities

        if not (MING_WEIGHTS / "transformer").is_dir():
            raise InferenceError(f"Ming weights missing at {MING_WEIGHTS}")
        args = SimpleNamespace(
            processor=None,
            dtype="bfloat16",
            attn_implementation="eager",
            device=device if ":" in device else f"{device}:0",
            device_map="balanced",
            num_gpus=1,
        )
        self._profile = load_checkpoint_capabilities(MING_WEIGHTS)
        self._model, self._processor = load_model_and_processor(MING_WEIGHTS, args)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._profile = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        from infer import run_generation

        self._check_cancel(cancel_flag)
        if self._model is None or self._processor is None or self._profile is None:
            raise InferenceError("ming-image-0.1-design is not loaded")
        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            raise InferenceError("prompt is required")
        steps = int(params.get("steps", DEFAULT_STEPS))
        cfg = float(params.get("cfg_scale", DEFAULT_CFG))
        seed = int(params.get("seed", 42))
        width = int(params.get("width") or 1024)
        height = int(params.get("height") or 1024)
        sampling = self._profile.resolve_sampling_parameters(steps=steps, cfg=cfg)
        started = time.perf_counter()
        self._check_cancel(cancel_flag)
        images = run_generation(
            self._model,
            self._processor,
            self._profile,
            task="text-to-image",
            prompt=prompt,
            input_image=None,
            resolution=1024,
            sampling=sampling,
            seed=seed,
            num_layers=1,
            dtype=_dtype(),
            width=width,
            height=height,
        )
        elapsed = time.perf_counter() - started
        if not images:
            raise InferenceError("Ming returned no images")
        image = images[0]
        self._write_png(output_dir / "result.png", image)
        return {
            "format": "png",
            "mode": "t2i",
            "width": image.width,
            "height": image.height,
            "steps": sampling.steps,
            "cfg_scale": sampling.cfg,
            "seed": seed,
            "inference_seconds": round(elapsed, 2),
            "model": "inclusionAI/Ming-Image-0.1-Design",
            "file": "result.png",
        }

    def estimate_time(self, params: dict) -> float:
        steps = int(params.get("steps", DEFAULT_STEPS))
        return 12000.0 + 2500.0 * steps

    @staticmethod
    def _write_png(path: Path, image) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            image.save(handle, format="PNG")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            directory = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass


def _dtype():
    import torch

    return torch.bfloat16
