"""Shared utilities for the voice-bench adapters (MiniCPM-o vs Moshi vs Kyutai cascade).

Not a registered adapter — imported by voice_bench_minicpmo / voice_bench_moshi /
voice_bench_kyutai.

Benchmark protocol (identical scripted input for every system):
  - Synthetic utterance WAVs (params u1_file..u4_file) are streamed into the
    system at real time (paced by wall clock).
  - u1: short question (EOU scenario), u2: longer prompt (interrupt/talkover),
    u3: barge-in with stop intent ("Stop. Actually, what's the weather..."),
    u4: talk-over probe without stop intent ("...my favourite colour is purple").
  - Metrics are measured in-process with time.monotonic() around the model
    API calls — no server/network overhead, so numbers are pure model-path
    latency (the ESP32 box would add identical Wi-Fi/Opus transport on top).
"""

from __future__ import annotations

import logging
import os
import time

# The spark login shell points HF_HOME at /mnt/t9 (full); workers must use the
# root-disk cache where the voice-bench weights were staged.
os.environ.setdefault("HF_HOME", "/home/darren/hf-voicebench")

from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Activity gate for 16-bit-scaled float audio in [-1, 1]. Synthetic `say`
# utterances peak around 0.3; silence/noise floor sits well below 0.01.
RMS_ACTIVE = 0.01
# Sample rates per system are fixed: MiniCPM-o in = 16 kHz, everything else 24 kHz.
SR_IN_MINICPMO = 16000
SR_IN_KYUTAI = 24000
FRAME_24K = 1920  # 80 ms mimi frame


def read_wav_float32(path: str) -> np.ndarray:
    """Read a wav file as mono float32 numpy in [-1, 1] (any sample rate)."""
    import soundfile as sf

    data, _sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=-1)
    return np.ascontiguousarray(data, dtype=np.float32)


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Cheap linear resample (benchmark-grade; avoids librosa version pins)."""
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    xp = np.linspace(0.0, len(x), num=len(x), endpoint=False)
    xq = np.linspace(0.0, len(x), num=n_out, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)


def pad_to(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=np.float32)])


def frames_of(x: np.ndarray, n: int):
    for off in range(0, len(x), n):
        yield x[off : off + n]


def rms(x) -> float:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr) + 1e-12))


def is_active(x) -> bool:
    return rms(x) > RMS_ACTIVE


def pace(t_chunk_start: float, chunk_dur_s: float) -> None:
    """Sleep so that chunk_dur_s of wall time elapses from chunk start."""
    dt = time.monotonic() - t_chunk_start
    if dt < chunk_dur_s:
        time.sleep(chunk_dur_s - dt)


class Events:
    """Monotonic event log; rendered as ms-since-first-event."""

    def __init__(self) -> None:
        self._t0: float | None = None
        self.items: list[dict] = []

    def add(self, kind: str, **note) -> float:
        t = time.monotonic()
        if self._t0 is None:
            self._t0 = t
        self.items.append({"t_ms": round((t - self._t0) * 1000.0, 1), "kind": kind, **note})
        return t

    def elapsed_ms(self, t: float) -> float:
        if self._t0 is None:
            return 0.0
        return (t - self._t0) * 1000.0


def save_artifacts(output_dir: Path, tag: str, audio: np.ndarray, sr: int, transcript: str) -> dict:
    """Write assistant audio + transcript for a scenario run; return file list."""
    import soundfile as sf

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    if audio is not None and len(audio) > 0:
        wav_path = output_dir / f"{tag}_assistant.wav"
        sf.write(str(wav_path), audio, sr)
        files["assistant_audio"] = wav_path.name
        files["assistant_audio_s"] = round(len(audio) / sr, 3)
    txt_path = output_dir / f"{tag}_transcript.txt"
    txt_path.write_text(transcript)
    files["transcript"] = txt_path.name
    return files


def resolve_utterances(params: dict) -> dict[str, np.ndarray]:
    """Read the four staged utterance wavs (u1..u4) as float32."""
    out = {}
    for key in ("u1", "u2", "u3", "u4"):
        p = params.get(f"{key}_file")
        if not p:
            raise ValueError(f"missing param {key}_file")
        out[key] = read_wav_float32(p)
    return out
