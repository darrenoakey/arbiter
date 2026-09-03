"""Vocal-stem adapter: Demucs htdemucs separation + loudness normalization.

The mv-lipsync renderer's L1 stage: separate the on-disk audio master into a
vocal stem, then normalize that stem to a target integrated loudness
(default -14 LUFS) with a -1 dBTP ceiling, emitting a stats.json loudness
record. The downstream a2v lever (ltx25-denoise1) keys off this normalized
stem, and the A/B forensics key off stats.json — so the stats must be real
measured values, never assumed.

Unlike the `demucs` job type (base64 in, two stems back for remixing), this
job type operates on an absolute file path on spark local disk (the renderer
stages its master there) and exists purely to produce the normalized stem.

Contract (output_dir):
    vocals.wav            raw htdemucs vocal stem, PCM_16, 44.1 kHz
    vocals_normalized.wav stem gain-matched to target_lufs, true-peak ceiling
                          -1 dBTP (if the LUFS gain would breach the ceiling
                          the whole stem is attenuated to the ceiling — the
                          recorded output_lufs/gain_db are the ACTUAL values)
    stats.json            {"input_lufs", "output_lufs", "gain_db",
                           "peak_dbtp", "model", "seconds"}

Fail loud: no supported-model negotiation (anything but "htdemucs" is
rejected), no silent clipping, no assumed stats. Every abnormal path raises
InferenceError / LoadError.

Runs in a subprocess (venvs/demucs via sys.executable, same as demucs_sep) —
the low-level demucs API only; the `demucs.api` CLI path pulls in `sphn`,
which has no aarch64 wheel. pyloudnorm and torchaudio must also be present in
that venv (see VOICE_RVC.md recreate block).
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import subprocess
import sys
import threading
from pathlib import Path

from arbiter.adapters.base import InferenceError, LoadError, ModelAdapter
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

SUPPORTED_MODEL = "htdemucs"
DEFAULT_TARGET_LUFS = -14.0
PEAK_CEILING_DBTP = -1.0

# Separation + normalization in one subprocess: reads argv[1] (input audio),
# argv[2] (outdir), argv[3] (target_lufs). Writes vocals.wav,
# vocals_normalized.wav, stats.json. Uses only sphn-free demucs modules.
_STEM_SCRIPT = r"""
import json
import sys
import numpy as np
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
from pyloudnorm import Meter
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import AudioFile, convert_audio

inp = sys.argv[1]
outdir = Path(sys.argv[2])
target_lufs = float(sys.argv[3])
ceiling = 10.0 ** (-1.0 / 20.0)  # -1 dBTP linear

model = get_model("htdemucs")
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
sr = int(model.samplerate)
ch = int(model.audio_channels)
sources = list(model.sources)
if "vocals" not in sources:
    raise SystemExit(f"VOCAL_STEM_FAIL htdemucs has no vocals stem; sources={sources}")

wav = AudioFile(inp).read(streams=0, samplerate=sr, channels=ch)
wav = convert_audio(wav, sr, sr, ch)
seconds = wav.shape[-1] / sr
ref = wav.mean(0)
wav = (wav - ref.mean()) / (ref.std() + 1e-8)
device = "cuda" if torch.cuda.is_available() else "cpu"
with torch.no_grad():
    out = apply_model(model, wav[None], device=device, shifts=1, split=True, overlap=0.25, progress=False)[0]
out = out * ref.std() + ref.mean()
vocals = out[sources.index("vocals")]
vocals_np = vocals.cpu().numpy().T  # (samples, channels), PCM float
sf.write(str(outdir / "vocals.wav"), vocals_np, sr, subtype="PCM_16")

# Loudness: pyloudnorm validates channels as shape[1] -> (samples, ch).
meter = Meter(sr)
input_lufs = float(meter.integrated_loudness(vocals_np))
if not np.isfinite(input_lufs):
    raise SystemExit("VOCAL_STEM_FAIL vocals stem has no measurable loudness (silent?)")
gain_db = target_lufs - input_lufs
norm = vocals_np * (10.0 ** (gain_db / 20.0))
output_lufs = float(meter.integrated_loudness(norm))
if abs(output_lufs - target_lufs) > 0.1:
    raise SystemExit(
        f"VOCAL_STEM_FAIL LUFS not invariant: target {target_lufs}, got {output_lufs:.2f}"
    )

# True peak via 4x oversampling (dBTP, not sample peak).
t = torch.from_numpy(norm.T[None]).float()
over = torchaudio.transforms.Resample(sr, sr * 4)(t).numpy()
peak = float(np.max(np.abs(over)))
peak_dbtp = 20.0 * np.log10(max(peak, 1e-12))
if peak > ceiling:
    # Ceiling attenuation (whole stem) — no clipping, dynamics preserved.
    atten_db = 20.0 * np.log10(ceiling / peak)
    norm = norm * (ceiling / peak)
    gain_db += atten_db
    output_lufs = float(meter.integrated_loudness(norm))
    peak_dbtp = -1.0
sf.write(str(outdir / "vocals_normalized.wav"), norm, sr, subtype="PCM_16")

stats = {
    "input_lufs": round(input_lufs, 2),
    "output_lufs": round(output_lufs, 2),
    "gain_db": round(float(gain_db), 2),
    "peak_dbtp": round(float(peak_dbtp), 2),
    "model": "htdemucs",
    "seconds": round(float(seconds), 3),
}
(outdir / "stats.json").write_text(json.dumps(stats, indent=1))
print(f"VOCAL_STEM_OK input_lufs={stats['input_lufs']} output_lufs={stats['output_lufs']}")
"""


@register
class VocalStemAdapter(ModelAdapter):
    model_id = "vocal-stem"

    def __init__(self):
        self._ready = False

    def load(self, device: str = "cuda") -> None:
        # Verify the whole subprocess import surface in this venv; the model
        # loads inside the subprocess (no persistent GPU held between jobs).
        for module in (
            "demucs.apply",
            "demucs.pretrained",
            "pyloudnorm",
            "torchaudio",
            "soundfile",
        ):
            try:
                importlib.import_module(module)
            except Exception as e:  # noqa: BLE001
                raise LoadError(
                    f"vocal-stem dep {module!r} not importable in this venv: {e}"
                ) from e
        self._ready = True
        log.info("vocal-stem ready (subprocess mode).")

    def unload(self) -> None:
        log.info("Unloading vocal-stem (no persistent GPU held).")
        self._ready = False

    @staticmethod
    def _validate_params(params: dict) -> tuple[str, float]:
        """Fail loud on anything but the exact supported contract."""
        audio_file = params.get("audio_file")
        if not audio_file:
            raise InferenceError("audio_file is required (absolute path on spark)")
        if not Path(audio_file).is_file():
            raise InferenceError(f"audio_file does not exist: {audio_file}")
        model = params.get("model", SUPPORTED_MODEL)
        if model != SUPPORTED_MODEL:
            raise InferenceError(
                f"unsupported separation model {model!r}; only {SUPPORTED_MODEL!r}"
            )
        target_lufs = float(params.get("target_lufs", DEFAULT_TARGET_LUFS))
        if not math.isfinite(target_lufs):
            raise InferenceError(f"target_lufs must be finite, got {target_lufs}")
        return str(audio_file), target_lufs

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        # Params validate BEFORE the ready check so a malformed job always
        # reports the real defect, never "not loaded".
        audio_file, target_lufs = self._validate_params(params)
        self._check_cancel(cancel_flag)
        if not self._ready:
            raise InferenceError("vocal-stem not loaded (call load first)")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        duration = _probe_duration(audio_file)
        timeout_s = max(600, int(duration * 5 + 120))

        log.info(
            "Separating + normalizing %.1fs of audio (htdemucs -> %.1f LUFS, timeout=%ds) ...",
            duration,
            target_lufs,
            timeout_s,
        )
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _STEM_SCRIPT,
                    audio_file,
                    str(output_dir),
                    str(target_lufs),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            raise InferenceError(f"vocal-stem timed out after {timeout_s}s") from e
        self._check_cancel(cancel_flag)

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-800:]
            raise InferenceError(
                f"vocal-stem separation failed (exit {proc.returncode}): {tail}"
            )

        vocals_path = output_dir / "vocals.wav"
        normalized_path = output_dir / "vocals_normalized.wav"
        stats_path = output_dir / "stats.json"
        for path in (vocals_path, normalized_path, stats_path):
            if not path.is_file():
                raise InferenceError(f"vocal-stem did not produce {path.name}")
        try:
            stats = json.loads(stats_path.read_text())
        except Exception as e:  # noqa: BLE001
            raise InferenceError(f"vocal-stem stats.json unreadable: {e}") from e

        return {
            "format": "wav",
            "file": "vocals_normalized.wav",
            "vocals": "vocals.wav",
            "normalized": "vocals_normalized.wav",
            "stats": "stats.json",
            "samplerate": 44100,
            "duration_seconds": round(duration, 3),
            **stats,
        }

    def estimate_time(self, params: dict) -> float:
        duration_s = params.get("duration")
        if duration_s is None:
            audio_file = params.get("audio_file", "")
            if audio_file and Path(audio_file).is_file():
                duration_s = _probe_duration(audio_file)
        if duration_s:
            return float(duration_s) * 2000.0 + 8000.0
        return 60000.0


def _probe_duration(path: str) -> float:
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return 30.0
