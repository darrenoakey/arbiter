"""Moondream 2 vision-language adapter."""
from __future__ import annotations

import base64
import io
import json
import logging
import threading
from pathlib import Path

from arbiter.adapters.base import ModelAdapter, InferenceError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

MODEL_HF_ID = "vikhyatk/moondream2"


@register
class MoondreamAdapter(ModelAdapter):
    model_id = "moondream"

    def __init__(self):
        self._model = None

    def load(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM

        log.info("Loading %s on %s with bfloat16 ...", MODEL_HF_ID, device)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_HF_ID,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map={"": device},
        )
        self._device = device
        log.info("Moondream2 ready.")

    def unload(self) -> None:
        log.info("Unloading Moondream2.")
        del self._model
        self._model = None
        self._cleanup_gpu()

    def _decode_image(self, params: dict):
        from PIL import Image

        image_b64 = params.get("image") or params.get("image_url", "")
        if image_b64.startswith("data:"):
            _, image_b64 = image_b64.split(",", 1)
        try:
            return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        except Exception as e:
            raise InferenceError(f"Failed to decode image: {e}")

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        self._check_cancel(cancel_flag)

        image = self._decode_image(params)
        task = params.get("task", "caption")

        self._check_cancel(cancel_flag)

        if task == "caption":
            result = self._caption(image, params)
        elif task == "query":
            result = self._query(image, params)
        elif task == "detect":
            result = self._detect(image, params)
        elif task == "point":
            result = self._point(image, params)
        else:
            raise InferenceError(f"Unknown task: {task}")

        self._check_cancel(cancel_flag)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "result.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))

        return {
            "format": "json",
            "file": "result.json",
            "task": task,
            **result,
        }

    def _sampling_kwargs(self, params: dict) -> dict:
        kw = {}
        if "temperature" in params:
            kw["temperature"] = float(params["temperature"])
        if "max_tokens" in params:
            kw["max_new_tokens"] = int(params["max_tokens"])
        if "top_p" in params:
            kw["top_p"] = float(params["top_p"])
        return kw

    def _caption(self, image, params: dict) -> dict:
        length = params.get("length", "normal")
        skw = self._sampling_kwargs(params)
        result = self._model.caption(image, length=length, **skw)
        return {"caption": result["caption"]}

    def _query(self, image, params: dict) -> dict:
        question = params.get("question", "")
        if not question:
            raise InferenceError("question is required for query task")
        reasoning = str(params.get("reasoning", "false")).lower() == "true"
        skw = self._sampling_kwargs(params)
        result = self._model.query(image=image, question=question, reasoning=reasoning, **skw)
        return {"answer": result["answer"]}

    def _detect(self, image, params: dict) -> dict:
        obj = params.get("object") or params.get("obj", "")
        if not obj:
            raise InferenceError("object is required for detect task")
        w, h = image.size
        result = self._model.detect(image, obj)
        objects = []
        for det in result.get("objects", []):
            objects.append({
                "bbox": [
                    round(det["x_min"] * w),
                    round(det["y_min"] * h),
                    round(det["x_max"] * w),
                    round(det["y_max"] * h),
                ],
                "confidence": det.get("confidence", 1.0),
            })
        return {"objects": objects}

    def _point(self, image, params: dict) -> dict:
        obj = params.get("object") or params.get("obj", "")
        if not obj:
            raise InferenceError("object is required for point task")
        w, h = image.size
        result = self._model.point(image, obj)
        points = [{"x": round(p["x"] * w), "y": round(p["y"] * h)} for p in result.get("points", [])]
        return {"points": points, "count": len(points)}

    def estimate_time(self, params: dict) -> float:
        return 2000.0
