"""BiRefNet background-removal adapter."""
from __future__ import annotations

import base64
import io
import logging
import threading
from pathlib import Path

from arbiter.adapters.base import ModelAdapter, InferenceError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

_INPUT_SIZE = (1024, 1024)


@register
class BiRefNetAdapter(ModelAdapter):
    model_id = "birefnet"

    def __init__(self):
        self._model = None

    def load(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForImageSegmentation

        log.info("Loading BiRefNet_HR on %s ...", device)
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet_HR",
            trust_remote_code=True,
        )
        model.eval()
        self._model = model.to(device).float()
        self._device = device
        log.info("BiRefNet ready.")

    def unload(self) -> None:
        log.info("Unloading BiRefNet.")
        del self._model
        self._model = None
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        import torch
        from torchvision import transforms
        from PIL import Image, ImageOps

        self._check_cancel(cancel_flag)

        # Decode input image from base64
        image_b64 = params.get("image") or params.get("image_url", "")
        if image_b64.startswith("data:"):
            _, image_b64 = image_b64.split(",", 1)
        try:
            original = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        except Exception as e:
            raise InferenceError(f"Failed to decode image: {e}")

        original = ImageOps.exif_transpose(original)
        rgb = original.convert("RGB")

        transform = transforms.Compose([
            transforms.Resize(_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self._check_cancel(cancel_flag)

        input_tensor = transform(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            outputs = self._model(input_tensor)
            prediction = outputs[-1].sigmoid().cpu()

        self._check_cancel(cancel_flag)

        alpha_tensor = prediction[0].squeeze()
        alpha_image = transforms.ToPILImage()(alpha_tensor)
        alpha_mask = alpha_image.resize(original.size)

        result = original.convert("RGBA")
        result.putalpha(alpha_mask)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "result.png"
        result.save(str(out_path), format="PNG")

        return {
            "format": "png",
            "width": result.width,
            "height": result.height,
            "file": "result.png",
        }

    def estimate_time(self, params: dict) -> float:
        return 200.0
