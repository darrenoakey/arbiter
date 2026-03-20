"""Qwen3-TTS VoiceClone adapter."""
from __future__ import annotations

import base64
import tempfile
import threading
from pathlib import Path

from .base import ModelAdapter, LoadError, InferenceError
from .registry import register


@register
class TTSCloneAdapter(ModelAdapter):
    model_id = "tts-clone"

    _HF_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

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
        ref_audio_b64 = params["ref_audio"]
        ref_text = params.get("ref_text")
        language = params.get("language", "English")
        temperature = params.get("temperature", 0.9)

        # Write reference audio to a temp file for the model API
        tmp_file = None
        try:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_file.write(base64.b64decode(ref_audio_b64))
            tmp_file.close()

            voice_clone_prompt = self._model.create_voice_clone_prompt(
                ref_audio=tmp_file.name,
                ref_text=ref_text,
            )

            self._check_cancel(cancel_flag)

            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                temperature=temperature,
            )
        finally:
            if tmp_file is not None:
                Path(tmp_file.name).unlink(missing_ok=True)

        self._check_cancel(cancel_flag)

        out_path = output_dir / "result.wav"
        sf.write(str(out_path), wavs[0].cpu().numpy(), sr)

        return {"format": "wav", "sample_rate": sr}

    def estimate_time(self, params: dict) -> float:
        text = params.get("text", "")
        # Voice cloning has extra overhead for encoding the reference audio
        return max(2000, len(text.split()) * 200)
