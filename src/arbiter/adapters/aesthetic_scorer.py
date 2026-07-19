"""Aesthetic image scorer adapter — CLIP-based multi-dimensional aesthetic scoring."""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import torch

from arbiter.adapters.base import ModelAdapter, InferenceError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

_MODEL_REPOSITORY = "rsinema/aesthetic-scorer"
_BACKBONE_REPOSITORY = "openai/clip-vit-base-patch32"
_SCORE_NAMES = (
    "overall_aesthetic",
    "technical_quality",
    "composition",
    "lighting",
    "color_harmony",
    "depth_of_field",
    "content",
)


class _Processor(Protocol):
    def __call__(self, **kwargs: object) -> dict[str, "torch.Tensor"]: ...


class _Scorer(Protocol):
    def __call__(self, pixels: "torch.Tensor") -> "torch.Tensor": ...


@register
class AestheticScorerAdapter(ModelAdapter):
    model_id = "aesthetic-scorer"

    def __init__(self):
        self._processor: _Processor | None = None
        self._model: _Scorer | None = None
        self._device = "cpu"

    def load(self, device: str = "cuda") -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import CLIPImageProcessor, CLIPModel

        log.info("Loading aesthetic scorer on %s ...", device)

        # Load CLIP processor
        processor = CLIPImageProcessor.from_pretrained(_MODEL_REPOSITORY, use_fast=True)

        # Load CLIP vision backbone
        backbone = CLIPModel.from_pretrained(_BACKBONE_REPOSITORY).vision_model

        # Load custom scorer class from HF repo
        module_path = Path(
            hf_hub_download(_MODEL_REPOSITORY, "aesthetic_scorer.py", repo_type="model")
        )
        spec = importlib.util.spec_from_file_location(
            "hf_aesthetic_scorer", module_path
        )
        if spec is None or spec.loader is None:
            raise InferenceError(f"Unable to load module spec from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scorer_class = module.AestheticScorer

        # Build model and load trained weights
        model = scorer_class(backbone)
        state_path = Path(
            hf_hub_download(_MODEL_REPOSITORY, "model.pt", repo_type="model")
        )
        state_dict = torch.load(state_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self._processor = cast(_Processor, processor)
        self._model = cast(_Scorer, model.to(device))
        self._device = device
        log.info("Aesthetic scorer ready.")

    def unload(self) -> None:
        log.info("Unloading aesthetic scorer.")
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        import torch

        self._check_cancel(cancel_flag)

        image = self._resolve_image(params)

        self._check_cancel(cancel_flag)

        # Run inference
        start = time.perf_counter()
        processor = self._processor
        model = self._model
        if processor is None or model is None:
            raise InferenceError("aesthetic-scorer not loaded")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self._device)
        with torch.no_grad():
            outputs = model(pixel_values)

        scores = {}
        for name, value in zip(_SCORE_NAMES, outputs, strict=True):
            numeric_value = float(value.squeeze().item())
            if not math.isfinite(numeric_value):
                raise InferenceError(f"Non-finite score for {name}: {numeric_value}")
            scores[name] = numeric_value

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Write result to output dir
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "scores": scores,
            "elapsed_ms": round(elapsed_ms, 1),
            "model_repository": _MODEL_REPOSITORY,
            "score_names": list(_SCORE_NAMES),
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2))

        return result

    def estimate_time(self, params: dict) -> float:
        return 2000.0  # ~2s on GPU
