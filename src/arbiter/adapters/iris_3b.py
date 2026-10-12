"""Iris-3B text-to-image adapter.

Owner comparison request, 2026-10-11. Pixel-space generator from
``speridlabs/iris-3b``. Not an IGS fallback. The worker imports the local
checkout at ``/home/darren/src/iris-3b`` and the NVMe snapshots; torch is
imported only inside ``load``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from arbiter.adapters.base import InferenceError, ModelAdapter
from arbiter.adapters.registry import register

IRIS_SRC = Path("/home/darren/src/iris-3b/src")
IRIS_WEIGHTS = Path("/home/darren/nvme-models/iris-3b")
IRIS_TEXT_ENCODER = Path("/home/darren/nvme-models/Qwen3-VL-4B-Instruct")
DEFAULT_STEPS = 100
DEFAULT_CFG = 3.0


@register
class Iris3BAdapter(ModelAdapter):
    """Serves only the ``iris-3b`` model id."""

    model_id = "iris-3b"

    def __init__(self) -> None:
        self._model = None
        self._text = None
        self._cfg = None
        self._device = "cuda"

    def load(self, device: str = "cuda") -> None:
        if str(IRIS_SRC) not in sys.path:
            sys.path.insert(0, str(IRIS_SRC))
        import torch

        import iris3b.text  # noqa: F401  (registers the Qwen3-VL encoder)
        from iris3b.config import inference_config
        from iris3b.models.dit import IrisDiT
        from iris3b.registry import TEXT_ENCODERS
        from iris3b.sampling import load_for_inference

        if not (IRIS_WEIGHTS / "model.safetensors").is_file():
            raise InferenceError(f"Iris weights missing at {IRIS_WEIGHTS}")
        if not (IRIS_TEXT_ENCODER / "config.json").is_file():
            raise InferenceError(f"Iris text encoder missing at {IRIS_TEXT_ENCODER}")

        raw, weights = load_for_inference(IRIS_WEIGHTS)
        cfg = inference_config(
            raw,
            [f"text_encoder.pretrained={IRIS_TEXT_ENCODER}"],
        )
        with torch.device("meta"):
            model = IrisDiT(cfg.model)
        model.load_state_dict(weights, strict=True, assign=True)
        model = model.eval().to(device=device, dtype=torch.float32)
        text_encoder = TEXT_ENCODERS.build(
            cfg.text_encoder.name, cfg.text_encoder, device=device
        )
        self._model = model
        self._text = text_encoder
        self._cfg = cfg
        self._device = device

    def unload(self) -> None:
        self._model = None
        self._text = None
        self._cfg = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        import torch
        from torchvision.transforms.functional import to_pil_image

        from iris3b.sampling import generate

        self._check_cancel(cancel_flag)
        if self._model is None or self._text is None or self._cfg is None:
            raise InferenceError("iris-3b is not loaded")
        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            raise InferenceError("prompt is required")
        steps = int(params.get("steps", DEFAULT_STEPS))
        cfg_scale = float(params.get("cfg_scale", DEFAULT_CFG))
        seed = int(params.get("seed", 42))
        width = int(params.get("width") or 1024)
        height = int(params.get("height") or 1024)
        negative = params.get("negative_prompt") or self._cfg.sample.negative_prompt
        device = torch.device(self._device)
        generator = torch.Generator(device=device).manual_seed(seed)
        started = time.perf_counter()
        self._check_cancel(cancel_flag)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            images = generate(
                self._model,
                self._text,
                [prompt],
                height=height,
                width=width,
                steps=steps,
                order=self._cfg.sample.order,
                cfg_scale=cfg_scale,
                cfg_interval=tuple(self._cfg.sample.cfg_interval),
                shift=self._cfg.flow.shift,
                negative_prompt=negative,
                generator=generator,
                device=device,
                num_train_timesteps=self._cfg.flow.num_train_timesteps,
                prediction=self._cfg.flow.prediction,
            )
        elapsed = time.perf_counter() - started
        image = images[0].detach().float().cpu().clamp(-1, 1)
        image = (image + 1) / 2
        pil = to_pil_image(image)
        self._write_png(output_dir / "result.png", pil)
        return {
            "format": "png",
            "mode": "t2i",
            "width": pil.width,
            "height": pil.height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "inference_seconds": round(elapsed, 2),
            "model": "speridlabs/iris-3b",
            "file": "result.png",
        }

    def estimate_time(self, params: dict) -> float:
        steps = int(params.get("steps", DEFAULT_STEPS))
        return 8000.0 + 400.0 * steps

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
