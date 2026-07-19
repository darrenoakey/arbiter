"""InsightFace adapter — face detection and 512-d ArcFace embeddings.

Wraps insightface.app.FaceAnalysis with the buffalo_l model pack. The
face-embed job type returns every detected face with its bounding box,
5-point landmarks, detection score, and 512-d embedding vector that
callers use for face clustering / identification.

Weights are downloaded on first use into ~/.insightface/models/buffalo_l/
by the insightface package itself — no manual download needed.
"""

from __future__ import annotations

import importlib
import json
import logging
import threading
import time
from pathlib import Path

from arbiter.adapters.base import InferenceError, ModelAdapter
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

_MODEL_PACK = "buffalo_l"
_DET_SIZE = (640, 640)


@register
class InsightFaceAdapter(ModelAdapter):
    model_id = "insightface"

    def __init__(self):
        self._app = None
        self._device = "cpu"

    def load(self, device: str = "cuda") -> None:
        import onnxruntime

        FaceAnalysis = importlib.import_module("insightface.app").FaceAnalysis

        available = onnxruntime.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        log.info(
            "Loading insightface %s (providers=%s, ctx_id=%d)",
            _MODEL_PACK,
            providers,
            ctx_id,
        )

        app = FaceAnalysis(name=_MODEL_PACK, providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=_DET_SIZE)

        self._app = app
        self._device = device
        log.info("insightface ready.")

    def unload(self) -> None:
        log.info("Unloading insightface.")
        self._app = None
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        import cv2
        import numpy as np

        self._check_cancel(cancel_flag)

        if self._app is None:
            raise InferenceError("insightface not loaded")

        image = self._resolve_image(params)  # PIL RGB
        self._check_cancel(cancel_flag)

        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        start = time.perf_counter()
        faces = self._app.get(img_bgr)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        results = []
        for face in faces:
            entry = {
                "bbox": [float(x) for x in face.bbox.tolist()],
                "det_score": float(face.det_score),
                "kps": [[float(x), float(y)] for x, y in face.kps.tolist()],
                "embedding": [float(x) for x in face.embedding.tolist()],
            }
            if getattr(face, "gender", None) is not None:
                entry["gender"] = int(face.gender)
            if getattr(face, "age", None) is not None:
                entry["age"] = int(face.age)
            results.append(entry)

        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "faces": results,
            "count": len(results),
            "elapsed_ms": round(elapsed_ms, 1),
            "model_pack": _MODEL_PACK,
            "image_width": int(image.width),
            "image_height": int(image.height),
        }
        (output_dir / "result.json").write_text(json.dumps(result))
        return result

    def estimate_time(self, params: dict) -> float:
        return 500.0
