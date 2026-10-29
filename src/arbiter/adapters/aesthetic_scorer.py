"""Aesthetic image scorer adapter — CLIP-based multi-dimensional aesthetic scoring."""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import math
import threading
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import torch

from arbiter.adapters.base import ModelAdapter, InferenceError, CancelledException
from arbiter.adapters.registry import register

_BATCH_SIZE = 32

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
    def __call__(self, pixels: "torch.Tensor") -> tuple["torch.Tensor", ...]: ...


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
        self._check_cancel(cancel_flag)

        from arbiter.schemas import AestheticScoreParams

        AestheticScoreParams.model_validate(params)
        start = time.perf_counter()
        if params.get("images_file"):
            payload = {"items": self._score_archive(params, cancel_flag)}
        else:
            image = self._resolve_image(params)
            try:
                pixels = self._pixels(image)
                payload = {"scores": self._score_pixels(pixels)[0]}
            finally:
                image.close()
        self._check_cancel(cancel_flag)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Write result to output dir
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            **payload,
            "elapsed_ms": round(elapsed_ms, 1),
            "model_repository": _MODEL_REPOSITORY,
            "score_names": list(_SCORE_NAMES),
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2))

        return result

    def _pixels(self, image):
        if self._processor is None:
            raise InferenceError("aesthetic-scorer not loaded")
        return self._processor(images=image, return_tensors="pt")["pixel_values"]

    def _score_pixels(self, pixels) -> list[dict]:
        import torch

        if self._model is None:
            raise InferenceError("aesthetic-scorer not loaded")
        with torch.inference_mode():
            outputs = self._model(pixels.to(self._device))
        # Each of the seven heads is [batch, 1], not [7] per image.
        heads = dict(zip(_SCORE_NAMES, outputs, strict=True))
        if any(tuple(head.shape) != (pixels.shape[0], 1) for head in heads.values()):
            raise InferenceError("Unexpected aesthetic scorer head shape")
        rows = []
        for index in range(pixels.shape[0]):
            scores = {name: float(head[index].item()) for name, head in heads.items()}
            if not all(math.isfinite(value) for value in scores.values()):
                raise InferenceError("Non-finite aesthetic score")
            rows.append(scores)
        return rows

    @staticmethod
    def _archive_image(archive, member):
        from PIL import Image

        # Never extract: member names are labels, not filesystem paths.
        with archive.open(member) as stream, Image.open(stream) as image:
            return image.convert("RGB")

    def _score_archive(self, params, cancel_flag) -> list[dict]:
        raw = self._resolve_media(params, "images")
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as error:
            raise InferenceError(f"bad input images ZIP: {error}") from error
        items: list[dict] = []
        pending: list[tuple[int, torch.Tensor]] = []
        with archive:
            for member in archive.infolist():
                self._check_cancel(cancel_flag)
                if member.is_dir():
                    continue
                index = len(items)
                items.append({"name": member.filename})
                try:
                    image = self._archive_image(archive, member)
                    try:
                        pixels = self._pixels(image)
                    finally:
                        image.close()
                    pending.append((index, pixels))
                except CancelledException:
                    raise
                except Exception as error:
                    items[index]["error"] = f"bad input image: {error}"
                if len(pending) == _BATCH_SIZE:
                    self._flush_batch(items, pending, cancel_flag)
            self._flush_batch(items, pending, cancel_flag)
        if not items:
            raise InferenceError("bad input images ZIP: no file members")
        return items

    def _flush_batch(self, items, pending, cancel_flag):
        import torch

        self._check_cancel(cancel_flag)
        if not pending:
            return
        try:
            rows = self._score_pixels(torch.cat([pixels for _, pixels in pending]))
            self._check_cancel(cancel_flag)
            for (index, _), scores in zip(pending, rows, strict=True):
                items[index]["scores"] = scores
        except CancelledException:
            raise
        except Exception as error:
            for index, _ in pending:
                items[index]["error"] = str(error)
        finally:
            pending.clear()

    def estimate_time(self, params: dict) -> float:
        return 2000.0  # ~2s on GPU
