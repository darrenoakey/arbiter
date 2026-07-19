"""Qwen3-TTS CustomVoice adapter."""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from .base import ModelAdapter
from .registry import register


class _CustomVoiceModel(Protocol):
    def generate_custom_voice(
        self, **kwargs: object
    ) -> tuple[list[np.ndarray | _CpuAudio], int]: ...


class _CpuAudio(Protocol):
    def cpu(self) -> "_CpuAudio": ...
    def numpy(self) -> np.ndarray: ...


@register
class TTSCustomAdapter(ModelAdapter):
    model_id = "tts-custom"

    _HF_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    def __init__(self):
        self._model: _CustomVoiceModel | None = None

    def load(self, device: str = "cuda") -> None:
        import torch

        Qwen3TTSModel = importlib.import_module("qwen_tts").Qwen3TTSModel

        kwargs = {
            "device_map": f"{device}:0" if device == "cuda" else device,
            "dtype": torch.bfloat16,
        }
        try:
            kwargs["attn_implementation"] = "flash_attention_2"
            self._model = cast(
                _CustomVoiceModel,
                Qwen3TTSModel.from_pretrained(self._HF_MODEL, **kwargs),
            )
        except Exception:
            kwargs.pop("attn_implementation", None)
            self._model = cast(
                _CustomVoiceModel,
                Qwen3TTSModel.from_pretrained(self._HF_MODEL, **kwargs),
            )

    def unload(self) -> None:
        del self._model
        self._model = None
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        import numpy as np

        sf = importlib.import_module("soundfile")

        self._check_cancel(cancel_flag)

        text = params["text"]
        speaker = params.get("speaker", "Aiden")
        language = params.get("language", "English")
        temperature = params.get("temperature", 0.9)

        model = self._model
        if model is None:
            raise RuntimeError("tts-custom not loaded")
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            temperature=temperature,
        )

        self._check_cancel(cancel_flag)

        wav = wavs[0]
        if not isinstance(wav, np.ndarray):
            wav = cast(_CpuAudio, wav).cpu().numpy()

        out_path = output_dir / "result.wav"
        sf.write(str(out_path), wav, sr)

        return {"format": "wav", "sample_rate": sr}

    def estimate_time(self, params: dict) -> float:
        text = params.get("text", "")
        return max(1000, len(text.split()) * 150)
