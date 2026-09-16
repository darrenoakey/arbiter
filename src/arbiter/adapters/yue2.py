"""YuE2 (m-a-p/YuE2-3B) full-song music generation adapter.

YuE2 turns a style prompt and lyrics into a complete song with vocals and
accompaniment via symbolic planning (melody + chords) and flow-matching
acoustic decoding. Runs BF16 on ~24GB VRAM. Weights and the inference
package (yue2_infer) live in the dedicated ``venvs/yue2`` worker venv on
spark; model files resolve from the shared HF cache.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from .base import HeapTrimGuard, ModelAdapter
from .registry import register

_log = logging.getLogger(__name__)


@register
class Yue2Adapter(ModelAdapter):
    model_id = "yue2"

    _DEFAULT_MODEL = "m-a-p/YuE2-3B"
    _DEFAULT_VAE = "m-a-p/YuE2-Vae"
    _VALID_COT = ("full", "melody", "off")
    _SAMPLE_RATE = 48000

    def __init__(self) -> None:
        self._pipe: Any = None
        self._device: str = "cuda"

    def load(self, device: str = "cuda") -> None:
        from yue2 import YuE2Pipeline

        self._device = device
        target_device = f"{device}:0" if device == "cuda" else device

        _log.info(
            "Loading YuE2 pipeline (model=%s, vae=%s) on %s...",
            self._DEFAULT_MODEL,
            self._DEFAULT_VAE,
            target_device,
        )
        # memory_budget_gib caps this worker's CUDA fraction (the pipeline
        # enforces it via torch.cuda.set_per_process_memory_fraction), which
        # keeps the unified-memory host safe on the GB10.
        with HeapTrimGuard():
            self._pipe = YuE2Pipeline.from_pretrained(
                self._DEFAULT_MODEL,
                vae=self._DEFAULT_VAE,
                device=target_device,
                memory_budget_gib=22,
                progress=False,
            )
        _log.info("YuE2 pipeline loaded successfully.")

    def unload(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.close()
            except Exception:
                _log.exception("Error while closing YuE2 pipeline")
            del self._pipe
            self._pipe = None
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        if self._pipe is None:
            raise RuntimeError("yue2 pipeline is not loaded")

        style = str(params.get("style") or params.get("prompt") or "")
        lyrics = str(params.get("lyrics") or "")
        cot = str(params.get("cot", "full"))
        if cot not in self._VALID_COT:
            raise ValueError(f"cot must be one of {self._VALID_COT}")
        seed = params.get("seed")
        cfg_scale = params.get("cfg_scale")
        abc = params.get("abc")
        out_format = str(params.get("format", "mp3")).lower().lstrip(".")
        if out_format not in ("wav", "mp3", "flac", "ogg"):
            out_format = "mp3"

        if not style.strip():
            raise ValueError("style is required")
        if not lyrics.strip():
            raise ValueError("lyrics are required")

        call_kwargs: dict[str, Any] = {}
        if seed is not None:
            call_kwargs["seed"] = int(seed)
        if cfg_scale is not None:
            call_kwargs["cfg_scale"] = float(cfg_scale)
        if abc:
            call_kwargs["abc"] = str(abc)

        def _cancelled() -> bool:
            self._check_cancel(cancel_flag)
            return False

        _log.info(
            "Generating song with YuE2: style=%r, lyrics_len=%d, cot=%s, seed=%s",
            style[:60],
            len(lyrics),
            cot,
            seed,
        )

        song = self._pipe(
            style=style,
            lyrics=lyrics,
            cot=cot,
            cancelled=_cancelled,
            **call_kwargs,
        )

        self._check_cancel(cancel_flag)

        sample_rate = int(getattr(song, "sample_rate", None) or self._SAMPLE_RATE)
        truncated = getattr(song, "truncated", None) or {}
        timing = getattr(song, "timing", None) or {}
        audio = getattr(song, "audio", None)
        duration = 0.0
        channels = 2
        if audio is not None:
            duration = float(len(audio) / sample_rate)
            channels = int(audio.shape[1]) if getattr(audio, "ndim", 1) > 1 else 1

        if out_format in ("wav", "ogg"):
            # YuE2's SongResult.save only supports .flac/.wav; route ogg
            # through a wav intermediate as well.
            filename = f"result.{out_format}"
            out_path = output_dir / filename
            tmp_wav = output_dir / "_result_src.wav"
            song.save(tmp_wav)
            if out_format == "wav":
                tmp_wav.replace(out_path)
            else:
                ffmpeg_bin = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
                subprocess.run(
                    [ffmpeg_bin, "-y", "-i", str(tmp_wav), "-codec:a", "libvorbis", "-q:a", "6", str(out_path)],
                    check=True,
                    capture_output=True,
                )
                tmp_wav.unlink(missing_ok=True)
        else:
            # flac is the pipeline's native lossless output; mp3 is a
            # delivery conversion via ffmpeg (320kbps CBR), matching the
            # music-generate adapter's quality-first default.
            tmp_name = "_result_src.flac"
            song.save(output_dir / tmp_name)
            filename = f"result.{out_format}"
            out_path = output_dir / filename
            if out_format == "flac":
                (output_dir / tmp_name).replace(out_path)
            else:
                ffmpeg_bin = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
                subprocess.run(
                    [
                        ffmpeg_bin, "-y", "-i", str(output_dir / tmp_name),
                        "-codec:a", "libmp3lame", "-b:a", "320k", str(out_path),
                    ],
                    check=True,
                    capture_output=True,
                )
                (output_dir / tmp_name).unlink(missing_ok=True)

        _log.info(
            "YuE2 song generated: %.1fs (%dch @ %dHz), e2e=%.1fs, truncated=%s",
            duration,
            channels,
            sample_rate,
            float(timing.get("e2e_seconds", 0.0) or 0.0),
            bool(truncated.get("abc") or truncated.get("semantic")),
        )

        return {
            "format": out_format,
            "sample_rate": sample_rate,
            "duration": duration,
            "channels": channels,
            "style": style,
            "cot": cot,
            "truncated": bool(truncated.get("abc") or truncated.get("semantic")),
            "e2e_seconds": float(timing.get("e2e_seconds", 0.0) or 0.0),
            "file": filename,
        }

    def estimate_time(self, params: dict) -> float:
        # Rough: ~60s pipeline floor plus ~1.5s per lyric character on a GB10
        # (RTX 4090 reference: ~71s for a 3.6-minute song with full CoT).
        lyrics = str(params.get("lyrics") or "")
        return max(60_000.0, 60_000.0 + len(lyrics) * 1_500.0)
