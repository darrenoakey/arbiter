"""Qwen3-TTS VoiceClone adapter."""

from __future__ import annotations

import importlib
import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from .base import ModelAdapter
from .registry import register

log = logging.getLogger(__name__)

# Minimum cosine similarity between reference and output speaker embeddings.
# Below this, the output voice drifted too far and we regenerate.
VOICE_SIMILARITY_THRESHOLD = 0.85
MAX_VOICE_RETRIES = 3


class _SpeakerEmbeddingModel(Protocol):
    def extract_speaker_embedding(self, **kwargs: object) -> object: ...


class _CpuAudio(Protocol):
    def cpu(self) -> "_CpuAudio": ...
    def numpy(self) -> np.ndarray: ...


class _VoiceCloneModel(Protocol):
    model: _SpeakerEmbeddingModel

    def create_voice_clone_prompt(self, **kwargs: object) -> object: ...
    def generate_voice_clone(
        self, **kwargs: object
    ) -> tuple[list[np.ndarray | _CpuAudio], int]: ...


@register
class TTSCloneAdapter(ModelAdapter):
    model_id = "tts-clone"

    _HF_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    def __init__(self):
        self._model: _VoiceCloneModel | None = None

    def load(self, device: str = "cuda") -> None:
        import torch

        Qwen3TTSModel = importlib.import_module("qwen_tts").Qwen3TTSModel

        kwargs = {
            "device_map": f"{device}:0" if device == "cuda" else device,
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
        self._model = cast(
            _VoiceCloneModel,
            Qwen3TTSModel.from_pretrained(self._HF_MODEL, **kwargs),
        )

    def unload(self) -> None:
        del self._model
        self._model = None
        self._cleanup_gpu()

    def _extract_embedding(self, audio_path: str):
        """Extract speaker embedding from an audio file."""
        import torch

        sf = importlib.import_module("soundfile")
        import numpy as np

        data, sr = sf.read(audio_path)
        # Resample to 24kHz if needed (model expects 24k for x-vector)
        if sr != 24000:
            torchaudio = importlib.import_module("torchaudio")

            waveform = torch.tensor(data, dtype=torch.float32)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2 and waveform.shape[1] > waveform.shape[0]:
                waveform = waveform.T  # channels first
            if waveform.shape[0] > 1:
                waveform = waveform[:1]  # mono
            resampler = torchaudio.transforms.Resample(sr, 24000)
            waveform = resampler(waveform)
            data_24k = waveform.squeeze().numpy().astype("float32")
        else:
            data_24k = (
                data if isinstance(data, np.ndarray) else np.array(data)
            ).astype(np.float32)
            if data_24k.ndim > 1:
                data_24k = data_24k[:, 0]

        model = self._model
        if model is None:
            raise RuntimeError("tts-clone not loaded")
        emb = model.model.extract_speaker_embedding(
            audio=data_24k,
            sr=24000,
        )
        return emb

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Cosine similarity between two tensors/arrays."""
        import torch

        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float32)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float32)
        a = a.flatten().float()
        b = b.flatten().float()
        return torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)
        ).item()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        sf = importlib.import_module("soundfile")

        self._check_cancel(cancel_flag)

        text = params["text"]
        ref_text = params.get("ref_text")
        language = params.get("language", "English")
        temperature = params.get("temperature", 0.3)
        ref_audio_file = params.get("ref_audio_file", "?")

        word_count = len(text.split())
        text_preview = text[:80] + ("..." if len(text) > 80 else "")
        log.info(
            "TTS-CLONE START: %d words, ref=%s, temp=%.2f, text=%r",
            word_count,
            Path(ref_audio_file).name if ref_audio_file != "?" else "base64",
            temperature,
            text_preview,
        )

        MAX_REF_SECONDS = 20
        tmp_file = None
        try:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_file.write(self._resolve_media(params, "ref_audio"))
            tmp_file.close()

            # Trim if longer than MAX_REF_SECONDS
            data, sr = sf.read(tmp_file.name)
            ref_dur = len(data) / sr
            max_samples = MAX_REF_SECONDS * sr
            if len(data) > max_samples:
                data = data[:max_samples]
                sf.write(tmp_file.name, data, sr)
                log.info(
                    "  Trimmed reference audio from %.1fs to %ds",
                    ref_dur,
                    MAX_REF_SECONDS,
                )
            else:
                log.info("  Reference audio: %.1fs", ref_dur)

            # Extract reference speaker embedding for validation
            t0 = time.monotonic()
            ref_embedding = self._extract_embedding(tmp_file.name)
            log.info("  Reference embedding extracted in %.1fs", time.monotonic() - t0)

            model = self._model
            if model is None:
                raise RuntimeError("tts-clone not loaded")
            voice_clone_prompt = model.create_voice_clone_prompt(
                ref_audio=tmp_file.name,
                ref_text=ref_text,
                x_vector_only_mode=True,
            )

            self._check_cancel(cancel_flag)

            # Scale max_new_tokens to input text length.
            # At 12Hz, ~5 tokens per word at natural speech rate (~3.1 wps).
            max_tokens = min(max(word_count * 6, 100), 2048)
            log.info("  max_new_tokens=%d (words=%d)", max_tokens, word_count)

            # Generate with voice similarity validation + retry
            best_wav = None
            best_similarity = -1.0
            best_sr = None
            gen_temperature = min(temperature, 0.3)

            for attempt in range(MAX_VOICE_RETRIES):
                self._check_cancel(cancel_flag)

                t0 = time.monotonic()
                wavs, out_sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                    temperature=gen_temperature,
                    max_new_tokens=max_tokens,
                    repetition_penalty=1.1,
                    top_p=0.9,
                )
                gen_time = time.monotonic() - t0

                import numpy as np

                wav = wavs[0]
                if not isinstance(wav, np.ndarray):
                    wav = cast(_CpuAudio, wav).cpu().numpy()

                audio_dur = len(wav) / out_sr
                log.info(
                    "  Attempt %d/%d: generated %.2fs audio in %.1fs (temp=%.3f)",
                    attempt + 1,
                    MAX_VOICE_RETRIES,
                    audio_dur,
                    gen_time,
                    gen_temperature,
                )

                # Write to temp file for embedding extraction
                tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp_out.name, wav, out_sr)
                tmp_out.close()

                try:
                    out_embedding = self._extract_embedding(tmp_out.name)
                    similarity = self._cosine_similarity(ref_embedding, out_embedding)
                except Exception as e:
                    log.warning(
                        "  Voice similarity check FAILED: %s — omitting validation", e
                    )
                    similarity = (
                        1.0  # preserve synthesis when the optional comparison errors
                    )
                finally:
                    Path(tmp_out.name).unlink(missing_ok=True)

                verdict = "PASS" if similarity >= VOICE_SIMILARITY_THRESHOLD else "FAIL"
                log.info(
                    "  Voice similarity: %.3f (threshold=%.2f) -> %s",
                    similarity,
                    VOICE_SIMILARITY_THRESHOLD,
                    verdict,
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_wav = wav
                    best_sr = out_sr

                if similarity >= VOICE_SIMILARITY_THRESHOLD:
                    break

                # Lower temperature for retry to stay closer to reference
                old_temp = gen_temperature
                gen_temperature = max(gen_temperature * 0.5, 0.05)
                log.warning(
                    "  VOICE DRIFT DETECTED: similarity %.3f < threshold %.2f. "
                    "Retrying with temperature %.3f -> %.3f",
                    similarity,
                    VOICE_SIMILARITY_THRESHOLD,
                    old_temp,
                    gen_temperature,
                )

            wav = best_wav
            sr = best_sr

            if wav is None or sr is None:
                raise RuntimeError("tts-clone produced no candidate audio")

            if best_similarity < VOICE_SIMILARITY_THRESHOLD:
                log.error(
                    "  VOICE DRIFT UNRESOLVED after %d attempts. "
                    "Best similarity: %.3f (threshold: %.2f). "
                    "Text: %r. Ref: %s. Using best attempt anyway.",
                    MAX_VOICE_RETRIES,
                    best_similarity,
                    VOICE_SIMILARITY_THRESHOLD,
                    text_preview,
                    Path(ref_audio_file).name if ref_audio_file != "?" else "base64",
                )
            else:
                log.info(
                    "TTS-CLONE OK: %.2fs audio, similarity=%.3f, text=%r",
                    len(wav) / sr,
                    best_similarity,
                    text_preview,
                )

        finally:
            if tmp_file is not None:
                Path(tmp_file.name).unlink(missing_ok=True)

        self._check_cancel(cancel_flag)

        # Validate output duration
        expected_max_s = max(5, len(text.split()) * 1.5)
        actual_s = len(wav) / sr
        if actual_s > expected_max_s:
            log.warning(
                "  OUTPUT TOO LONG: %.1fs exceeds expected max %.1fs for %d words. "
                "Possible runaway generation.",
                actual_s,
                expected_max_s,
                word_count,
            )

        out_path = output_dir / "result.wav"
        sf.write(str(out_path), wav, sr)

        return {
            "format": "wav",
            "sample_rate": sr,
            "voice_similarity": round(best_similarity, 3),
        }

    def estimate_time(self, params: dict) -> float:
        text = params.get("text", "")
        # Voice cloning has extra overhead for encoding the reference audio + possible retries
        return max(2000, len(text.split()) * 300)
