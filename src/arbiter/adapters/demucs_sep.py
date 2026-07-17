"""Demucs source-separation adapter (htdemucs, two-stem vocals/accompaniment).

Splits a music clip into an isolated vocal stem and everything-else
("accompaniment"). Used by the voice pipeline: separate a Suno cover's
vocals, run rvc-convert on the vocal stem, then remix the converted vocals
back over the original accompaniment.

Runs the separation in a subprocess (in venvs/demucs via sys.executable) using
demucs' low-level API (apply/pretrained/audio) — NOT the `python -m demucs`
CLI, whose `demucs.api` import pulls in `sphn`, a Rust package that has no
aarch64 wheel and won't build here. Subprocess isolation also avoids an
in-process apply_model crash seen when demucs' internal pool runs inside the
arbiter worker thread. htdemucs reloads per call (~1-2s) — fine for the
low-volume verse workload.
"""
from __future__ import annotations

import base64
import logging
import subprocess
import sys
import threading
from pathlib import Path

from arbiter.adapters.base import InferenceError, LoadError, ModelAdapter
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

# Cap inline base64 per stem (raw bytes). demucs returns TWO stems in one
# result, so keep 2x this under the worker stdout pipe ceiling (96MB, see
# proc.go). 30MB raw/stem -> ~40MB b64/stem -> ~80MB for both, safely under.
# Larger stems get a skip note and must be fetched as files. Files always written.
_B64_CAP_BYTES = 30 * 1024 * 1024

# Standalone separation: reads argv[1] (input audio), writes vocals.wav +
# accompaniment.wav into argv[2]. Uses only sphn-free demucs modules.
_SEP_SCRIPT = r"""
import sys
import soundfile as sf
import torch
from pathlib import Path
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import AudioFile, convert_audio

inp = sys.argv[1]
outdir = Path(sys.argv[2])
model = get_model("htdemucs")
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
sr = int(model.samplerate)
ch = int(model.audio_channels)
sources = list(model.sources)
if "vocals" not in sources:
    raise SystemExit(f"htdemucs has no vocals stem; sources={sources}")

wav = AudioFile(inp).read(streams=0, samplerate=sr, channels=ch)
wav = convert_audio(wav, sr, sr, ch)
ref = wav.mean(0)
wav = (wav - ref.mean()) / (ref.std() + 1e-8)
device = "cuda" if torch.cuda.is_available() else "cpu"
with torch.no_grad():
    out = apply_model(model, wav[None], device=device, shifts=1, split=True, overlap=0.25, progress=False)[0]
out = out * ref.std() + ref.mean()
vi = sources.index("vocals")
vocals = out[vi]
accompaniment = sum(out[i] for i in range(len(sources)) if i != vi)
sf.write(str(outdir / "vocals.wav"), vocals.cpu().numpy().T, sr, subtype="PCM_16")
sf.write(str(outdir / "accompaniment.wav"), accompaniment.cpu().numpy().T, sr, subtype="PCM_16")
print(f"DEMUCS_OK sr={sr}")
"""


@register
class DemucsAdapter(ModelAdapter):
    model_id = "demucs"

    def __init__(self):
        self._ready = False

    def load(self, device: str = "cuda") -> None:
        # Verify the sphn-free import path works in this venv; the model loads
        # inside the subprocess (no persistent GPU held between jobs).
        try:
            import demucs.apply  # noqa: F401
            import demucs.pretrained  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise LoadError(f"demucs not importable in this venv: {e}") from e
        self._ready = True
        log.info("demucs ready (subprocess mode).")

    def unload(self) -> None:
        log.info("Unloading demucs (no persistent GPU held).")
        self._ready = False

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        self._check_cancel(cancel_flag)
        if not self._ready:
            raise InferenceError("demucs not loaded (call load first)")

        try:
            audio_bytes = self._resolve_media(params, "audio")
        except Exception as e:  # noqa: BLE001
            raise InferenceError(f"failed to resolve audio: {e}") from e

        return_b64 = bool(params.get("return_b64", False))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        src_path = output_dir / "_input_audio"
        src_path.write_bytes(audio_bytes)

        duration = _probe_duration(str(src_path))
        timeout_s = max(600, int(duration * 5 + 120))

        log.info("Separating %.1fs of audio (demucs two-stem, timeout=%ds) ...", duration, timeout_s)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _SEP_SCRIPT, str(src_path), str(output_dir)],
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            raise InferenceError(f"demucs timed out after {timeout_s}s") from e
        finally:
            src_path.unlink(missing_ok=True)

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-800:]
            raise InferenceError(f"demucs separation failed (exit {proc.returncode}): {tail}")
        self._check_cancel(cancel_flag)

        vocals_path = output_dir / "vocals.wav"
        accompaniment_path = output_dir / "accompaniment.wav"
        if not vocals_path.is_file() or not accompaniment_path.is_file():
            raise InferenceError("demucs did not produce both stems")

        result = {
            "format": "wav",
            "file": "vocals.wav",
            "vocals": "vocals.wav",
            "accompaniment": "accompaniment.wav",
            "samplerate": 44100,
            "duration_seconds": round(duration, 3),
        }
        if return_b64:
            for name, path in (("vocals_b64", vocals_path), ("accompaniment_b64", accompaniment_path)):
                raw = path.read_bytes()
                if len(raw) <= _B64_CAP_BYTES:
                    result[name] = base64.b64encode(raw).decode("ascii")
                else:
                    result[name + "_skipped"] = f"stem {len(raw)} bytes exceeds {_B64_CAP_BYTES} cap; fetch the file"
        return result

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
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return 30.0
