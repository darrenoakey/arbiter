"""Kokoro-82M TTS adapter.

Kokoro is a tiny (82M param), Apache-licensed TTS model that is hundreds of
times faster than the Qwen3-TTS family. It has no voice cloning — instead it
ships a fixed bank of named "voice packs" (256-dim style tensors). A voice is
selected by name (e.g. "af_heart", "bm_george") and may be blended from
several named voices with weights, plus a speech-rate multiplier.

This adapter is read-only with respect to the model: it loads one shared
KModel and one KPipeline per language code (the first letter of a voice name
selects the language: 'a'=American English, 'b'=British English, etc.).

Voice spec format (the ``voice`` param):
    "af_heart"                       single named voice
    "af_heart*0.6+am_michael*0.4"    weighted blend of two voices

Future work (documented, not implemented): RobViren/kvoicewalk can *evolve* a
real Kokoro .pt style tensor to match a target WAV (e.g. a Qwen3-TTS sample),
giving true per-character cloning. It costs ~30-90 min of GPU time per voice,
so the current pipeline maps character descriptions onto the built-in bank
instead. A hybrid model — pre-evolve a few hundred custom voices once, then
map onto that enlarged bank — is the natural next step.
"""
from __future__ import annotations

import threading
from pathlib import Path

from .base import ModelAdapter, InferenceError
from .registry import register

# Language code = first letter of a Kokoro voice name.
_KNOWN_LANG_CODES = {"a", "b", "e", "f", "h", "i", "j", "p", "z"}


@register
class KokoroTTSAdapter(ModelAdapter):
    model_id = "tts-kokoro"

    _HF_REPO = "hexgrad/Kokoro-82M"

    def __init__(self):
        self._model = None
        self._pipelines: dict = {}

    # ----------------------------------------------------------------
    # load
    # build one shared KModel; pipelines are created lazily per lang code
    def load(self, device: str = "cuda") -> None:
        import torch
        from kokoro import KModel
        dev = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self._model = KModel(repo_id=self._HF_REPO).to(dev).eval()
        self._device = dev
        self._pipelines = {}

    def unload(self) -> None:
        self._pipelines = {}
        del self._model
        self._model = None
        self._cleanup_gpu()

    # ----------------------------------------------------------------
    # pipeline for
    # return (creating if needed) the KPipeline for one language code,
    # sharing the single loaded KModel so we never load weights twice
    def _pipeline_for(self, lang_code: str):
        if lang_code not in self._pipelines:
            from kokoro import KPipeline
            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code, repo_id=self._HF_REPO, model=self._model,
            )
        return self._pipelines[lang_code]

    # ----------------------------------------------------------------
    # resolve voice
    # turn a voice spec into something KPipeline accepts: a bare name, or a
    # weighted-average style tensor for blends
    def _resolve_voice(self, pipeline, voice_spec: str):
        spec = voice_spec.strip()
        if "+" not in spec and "*" not in spec:
            return spec
        acc = None
        total = 0.0
        for part in spec.split("+"):
            name, _, w = part.partition("*")
            name = name.strip()
            weight = float(w) if w.strip() else 1.0
            pack = pipeline.load_single_voice(name)
            contrib = pack * weight
            acc = contrib if acc is None else acc + contrib
            total += weight
        if acc is None or total == 0:
            raise InferenceError(f"could not resolve voice blend: {voice_spec!r}")
        return acc / total

    # ----------------------------------------------------------------
    # lang code for
    # derive the language code from the (first) voice name
    @staticmethod
    def _lang_code_for(voice_spec: str, override: str) -> str:
        if override:
            return override
        first = voice_spec.strip().lstrip()
        letter = first[0].lower() if first else "a"
        return letter if letter in _KNOWN_LANG_CODES else "a"

    # ----------------------------------------------------------------
    # infer
    # synthesize one line of text and write a 24 kHz mono result.wav
    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        import numpy as np
        import soundfile as sf
        self._check_cancel(cancel_flag)

        text = params["text"]
        voice_spec = params.get("voice", "af_heart")
        speed = float(params.get("speed", 1.0))
        lang_code = self._lang_code_for(voice_spec, params.get("lang_code", ""))

        pipeline = self._pipeline_for(lang_code)
        voice = self._resolve_voice(pipeline, voice_spec)

        chunks: list = []
        for result in pipeline(text, voice=voice, speed=speed):
            self._check_cancel(cancel_flag)
            audio = result.audio if hasattr(result, "audio") else result[2]
            if audio is None:
                continue
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            elif not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            chunks.append(audio.astype(np.float32))

        if not chunks:
            raise InferenceError(f"kokoro produced no audio for text: {text[:80]!r}")
        wav = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        sr = 24000
        out_path = output_dir / "result.wav"
        sf.write(str(out_path), wav, sr)
        return {"format": "wav", "sample_rate": sr}

    def estimate_time(self, params: dict) -> float:
        # Kokoro runs far faster than realtime on the GB10; a few hundred ms
        # for typical lines. Scale gently with word count.
        text = params.get("text", "")
        return max(300, len(text.split()) * 20)
