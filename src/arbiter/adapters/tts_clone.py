"""Qwen3-TTS VoiceClone adapter."""
from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

log = logging.getLogger(__name__)

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
        kwargs = {
            "device_map": f"{device}:0" if device == "cuda" else device,
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
        self._model = Qwen3TTSModel.from_pretrained(self._HF_MODEL, **kwargs)

    def unload(self) -> None:
        del self._model
        self._model = None
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        import soundfile as sf
        self._check_cancel(cancel_flag)

        text = params["text"]
        # ref_audio resolved via _resolve_media (supports both base64 and file path)
        ref_text = params.get("ref_text")
        language = params.get("language", "English")
        temperature = params.get("temperature", 0.7)

        # Write reference audio to a temp file, trimmed to max 20s for ICL speed
        MAX_REF_SECONDS = 20
        tmp_file = None
        try:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_file.write(self._resolve_media(params, "ref_audio"))
            tmp_file.close()

            # Trim if longer than MAX_REF_SECONDS
            import soundfile as sf
            data, sr = sf.read(tmp_file.name)
            max_samples = MAX_REF_SECONDS * sr
            if len(data) > max_samples:
                orig_dur = len(data) / sr
                data = data[:max_samples]
                sf.write(tmp_file.name, data, sr)
                log.info("Trimmed reference audio from %.1fs to %ds", orig_dur, MAX_REF_SECONDS)

            voice_clone_prompt = self._model.create_voice_clone_prompt(
                ref_audio=tmp_file.name,
                ref_text=ref_text,
                x_vector_only_mode=True,
            )

            self._check_cancel(cancel_flag)

            # Scale max_new_tokens to input text length.
            # At 12Hz, ~2 tokens per word. Add headroom for pauses.
            word_count = len(text.split())
            max_tokens = min(max(word_count * 4, 60), 2048)

            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                temperature=max(temperature, 0.3),  # floor at 0.3 for voice stability
                max_new_tokens=max_tokens,
                repetition_penalty=1.1,
                top_p=0.9,
            )
        finally:
            if tmp_file is not None:
                Path(tmp_file.name).unlink(missing_ok=True)

        self._check_cancel(cancel_flag)

        import numpy as np
        wav = wavs[0]
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        elif not isinstance(wav, np.ndarray):
            wav = np.array(wav)

        # Validate output duration
        expected_max_s = max(5, len(text.split()) * 1.5)
        actual_s = len(wav) / sr
        if actual_s > expected_max_s:
            import sys
            print(f"WARNING: TTS output {actual_s:.1f}s exceeds expected max {expected_max_s:.1f}s for {len(text.split())} words", file=sys.stderr)

        out_path = output_dir / "result.wav"
        sf.write(str(out_path), wav, sr)

        return {"format": "wav", "sample_rate": sr}

    def estimate_time(self, params: dict) -> float:
        text = params.get("text", "")
        # Voice cloning has extra overhead for encoding the reference audio
        return max(2000, len(text.split()) * 200)
