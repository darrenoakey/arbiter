"""Whisper large-v3 transcription adapter."""
from __future__ import annotations

import base64
import json
import logging
import tempfile
import threading
from pathlib import Path

from arbiter.adapters.base import ModelAdapter, InferenceError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)


@register
class WhisperLargeAdapter(ModelAdapter):
    model_id = "whisper-large"

    def __init__(self):
        self._model = None

    def load(self, device: str = "cuda") -> None:
        import whisper

        log.info("Loading Whisper large-v3 on %s ...", device)
        self._model = whisper.load_model("large-v3", device=device)
        self._device = device
        log.info("Whisper large-v3 ready.")

    def unload(self) -> None:
        log.info("Unloading Whisper large-v3.")
        del self._model
        self._model = None
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        self._check_cancel(cancel_flag)

        # Decode base64 audio to a temp file
        audio_b64 = params.get("audio", "")
        audio_format = params.get("format", "wav")
        language = params.get("language", None)

        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            raise InferenceError(f"Failed to decode audio: {e}")

        with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        self._check_cancel(cancel_flag)

        try:
            transcribe_kwargs = {"word_timestamps": True}
            if language:
                transcribe_kwargs["language"] = language
            result = self._model.transcribe(tmp_path, **transcribe_kwargs)
        except Exception as e:
            raise InferenceError(f"Transcription failed: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        self._check_cancel(cancel_flag)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "result.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))

        return {
            "format": "json",
            "file": "result.json",
            "text": result.get("text", ""),
            "language": result.get("language", ""),
        }

    def estimate_time(self, params: dict) -> float:
        # Estimate based on audio duration if provided, otherwise default 10s
        duration_s = params.get("duration", None)
        if duration_s is not None:
            # Roughly 1:1 real-time on large-v3 with CUDA
            return float(duration_s) * 1000.0
        return 10000.0
