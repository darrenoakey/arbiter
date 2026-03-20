"""Qwen3-TTS CustomVoice adapter."""
from __future__ import annotations

import io
import json
import threading
from pathlib import Path

from .base import ModelAdapter, LoadError, InferenceError
from .registry import register


@register
class TTSCustomAdapter(ModelAdapter):
    model_id = "tts-custom"

    _HF_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    def __init__(self):
        self._model = None

    def load(self, device: str = "cuda") -> None:
        import torch
        from qwen_tts import Qwen3TTSModel
        kwargs = {"device_map": f"{device}:0" if device == "cuda" else device, "dtype": torch.bfloat16}
        try:
            kwargs["attn_implementation"] = "flash_attention_2"
            self._model = Qwen3TTSModel.from_pretrained(self._HF_MODEL, **kwargs)
        except Exception:
            kwargs.pop("attn_implementation", None)
            self._model = Qwen3TTSModel.from_pretrained(self._HF_MODEL, **kwargs)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        import soundfile as sf
        self._check_cancel(cancel_flag)

        text = params["text"]
        speaker = params.get("speaker", "Aiden")
        language = params.get("language", "English")
        temperature = params.get("temperature", 0.9)

        wavs, sr = self._model.generate_custom_voice(
            text=text, language=language, speaker=speaker, temperature=temperature,
        )

        self._check_cancel(cancel_flag)

        out_path = output_dir / "result.wav"
        sf.write(str(out_path), wavs[0].cpu().numpy(), sr)

        return {"format": "wav", "sample_rate": sr}

    def estimate_time(self, params: dict) -> float:
        text = params.get("text", "")
        return max(1000, len(text.split()) * 150)
