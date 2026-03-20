"""Test input generators for calibration."""
from __future__ import annotations

import base64
import io
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent.parent.parent.parent / "assets"


def get_test_image_b64() -> str:
    """Get a test image as base64. Creates a simple test image if no asset exists."""
    asset = _ASSETS / "test_face_512.png"
    if asset.exists():
        return base64.b64encode(asset.read_bytes()).decode()
    # Generate a simple test image
    try:
        from PIL import Image
        img = Image.new("RGB", (512, 512), (128, 128, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        return ""


def get_test_audio_b64() -> str:
    """Get a test audio clip as base64. Creates silence if no asset exists."""
    asset = _ASSETS / "test_audio_3s.wav"
    if asset.exists():
        return base64.b64encode(asset.read_bytes()).decode()
    # Generate 3 seconds of silence
    try:
        import numpy as np
        import soundfile as sf
        samples = np.zeros(24000 * 3, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, samples, 24000, format="WAV")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except ImportError:
        return ""
