"""ACE-Step 1.5 XL Supervised Fine-Tuned (SFT) music generation adapter."""

from __future__ import annotations

import importlib
import logging
import threading
from pathlib import Path
from typing import Any

from .base import HeapTrimGuard, ModelAdapter
from .registry import register

_log = logging.getLogger(__name__)


@register
class MusicGenerateAdapter(ModelAdapter):
    model_id = "music-generate"

    _DEFAULT_MODEL = "ACE-Step/acestep-v15-xl-sft-diffusers"

    def __init__(self) -> None:
        self._pipe: Any = None
        self._current_model: str = self._DEFAULT_MODEL
        self._device: str = "cuda"

    def load(self, device: str = "cuda") -> None:
        import torch
        from diffusers import AceStepPipeline

        self._device = device
        target_device = f"{device}:0" if device == "cuda" else device

        _log.info("Loading music generation pipeline from %s on %s...", self._DEFAULT_MODEL, target_device)
        with HeapTrimGuard():
            self._pipe = AceStepPipeline.from_pretrained(
                self._DEFAULT_MODEL,
                torch_dtype=torch.bfloat16,
            )
            self._pipe = self._pipe.to(target_device)
            self._current_model = self._DEFAULT_MODEL
        _log.info("Music generation pipeline loaded successfully.")

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        import numpy as np
        import soundfile as sf
        import torch

        self._check_cancel(cancel_flag)

        if self._pipe is None:
            raise RuntimeError("music-generate pipeline is not loaded")

        prompt = params.get("prompt", "")
        lyrics = params.get("lyrics", "") or ""
        audio_duration = float(params.get("audio_duration", 30.0))
        num_inference_steps = int(params.get("num_inference_steps", 50))
        guidance_scale = float(params.get("guidance_scale", 7.0))
        shift = float(params.get("shift", 3.0))
        vocal_language = params.get("vocal_language", "en")
        bpm = params.get("bpm")
        keyscale = params.get("keyscale")
        timesignature = params.get("timesignature")
        seed = params.get("seed")
        out_format = str(params.get("format", "wav")).lower().strip(".")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(int(seed))

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "lyrics": lyrics,
            "audio_duration": audio_duration,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "shift": shift,
            "vocal_language": vocal_language,
            "output_type": "np",
        }
        if generator is not None:
            call_kwargs["generator"] = generator
        if bpm is not None:
            call_kwargs["bpm"] = int(bpm)
        if keyscale is not None:
            call_kwargs["keyscale"] = str(keyscale)
        if timesignature is not None:
            call_kwargs["timesignature"] = str(timesignature)

        _log.info(
            "Generating music: prompt=%r, lyrics_len=%d, duration=%.1fs, steps=%d, guidance=%.1f",
            prompt[:60] if prompt else "",
            len(lyrics),
            audio_duration,
            num_inference_steps,
            guidance_scale,
        )

        output = self._pipe(**call_kwargs)

        self._check_cancel(cancel_flag)

        # Output audio array processing
        # audio shape in diffusers can be (batch, samples, channels) or (batch, channels, samples) or (channels, samples)
        audios = output.audios
        if isinstance(audios, torch.Tensor):
            audios = audios.cpu().float().numpy()
        elif not isinstance(audios, np.ndarray):
            audios = np.asarray(audios)

        if audios.ndim == 3:
            wav = audios[0]
        else:
            wav = audios

        # Determine channels and sample orientation for soundfile (expects [samples, channels])
        if wav.ndim == 2:
            if wav.shape[0] <= 2 and wav.shape[1] > wav.shape[0]:
                wav = wav.T
        elif wav.ndim == 1:
            wav = wav.reshape(-1, 1)

        sample_rate = getattr(self._pipe, "sample_rate", 48000)

        filename = f"result.{out_format}" if out_format in ("wav", "flac", "ogg", "mp3") else "result.wav"
        out_path = output_dir / filename
        sf.write(str(out_path), wav, sample_rate)

        actual_duration = float(len(wav) / sample_rate) if sample_rate > 0 else audio_duration

        return {
            "format": out_format,
            "sample_rate": sample_rate,
            "duration": actual_duration,
            "channels": int(wav.shape[1]) if wav.ndim > 1 else 1,
            "prompt": prompt,
            "file": filename,
        }

    def estimate_time(self, params: dict) -> float:
        duration = float(params.get("audio_duration", 30.0))
        steps = int(params.get("num_inference_steps", 50))
        # Estimate ~0.5s per step for 30s audio on Grace Blackwell GB10
        return max(5000, float(steps * 500 * (duration / 30.0)))
