"""SadTalker talking-head video generation adapter (subprocess-based).

Invokes SadTalker's inference.py as a subprocess. Unlike Sonic (which loads
models in-process), SadTalker loads and releases GPU memory per invocation.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from arbiter.adapters.base import ModelAdapter, InferenceError, LoadError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

SADTALKER_DIR = Path("/home/darren/src/talking-head/local-sadtalker")
SADTALKER_PYTHON = SADTALKER_DIR / "venv" / "bin" / "python"
SADTALKER_INFERENCE = SADTALKER_DIR / "inference.py"
SADTALKER_CHECKPOINTS = SADTALKER_DIR / "checkpoints"

REQUIRED_CHECKPOINTS = [
    "SadTalker_V0.0.2_256.safetensors",
    "epoch_00190_iteration_000400000_checkpoint.pt",
    "mapping_00229-model.pth.tar",
]


@register
class SadTalkerAdapter(ModelAdapter):
    model_id = "sadtalker"

    def __init__(self):
        self._ready = False

    def load(self, device: str = "cuda") -> None:
        """Verify SadTalker installation. No GPU memory held after load."""
        log.info("Verifying SadTalker installation...")

        if not SADTALKER_PYTHON.exists():
            raise LoadError(f"SadTalker venv not found: {SADTALKER_PYTHON}")
        if not SADTALKER_INFERENCE.exists():
            raise LoadError(f"SadTalker inference.py not found: {SADTALKER_INFERENCE}")

        for ckpt in REQUIRED_CHECKPOINTS:
            p = SADTALKER_CHECKPOINTS / ckpt
            if not p.exists():
                raise LoadError(f"Missing checkpoint: {p}")

        self._ready = True
        log.info("SadTalker ready (subprocess mode, no persistent GPU usage).")

    def unload(self) -> None:
        """No-op — SadTalker doesn't hold GPU memory between calls."""
        log.info("SadTalker unloaded (no GPU memory was held).")
        self._ready = False

    def infer(
        self,
        params: dict,
        output_dir: Path,
        cancel_flag: threading.Event,
    ) -> dict:
        """Generate a talking-head video via SadTalker subprocess.

        params:
            image / image_file: portrait image
            audio / audio_file: driving audio
            size: render size (256 or 512, default 256)
            facerender: renderer (pirender or facevid2vid, default pirender)
        """
        if not self._ready:
            raise InferenceError("SadTalker is not loaded (call load first)")

        self._check_cancel(cancel_flag)

        # Resolve inputs
        try:
            image_bytes = self._resolve_media(params, "image")
        except Exception as e:
            raise InferenceError(f"Failed to resolve image: {e}") from e
        try:
            audio_bytes = self._resolve_media(params, "audio")
        except Exception as e:
            raise InferenceError(f"Failed to resolve audio: {e}") from e

        size = int(params.get("size", 256))
        facerender = params.get("facerender", "pirender")
        expression_scale = float(params.get("expression_scale", 1.0))
        preprocess = params.get("preprocess", "crop")
        enhancer = params.get("enhancer", "")
        still = params.get("still", False)

        tmp_dir = tempfile.mkdtemp(prefix="sadtalker_")
        tmp_image = os.path.join(tmp_dir, "input.png")
        tmp_audio = os.path.join(tmp_dir, "input.wav")
        result_dir = os.path.join(tmp_dir, "results")
        os.makedirs(result_dir)

        try:
            with open(tmp_image, "wb") as f:
                f.write(image_bytes)
            with open(tmp_audio, "wb") as f:
                f.write(audio_bytes)

            self._check_cancel(cancel_flag)

            # Build SadTalker command
            cmd = [
                str(SADTALKER_PYTHON),
                str(SADTALKER_INFERENCE),
                "--source_image", tmp_image,
                "--driven_audio", tmp_audio,
                "--result_dir", result_dir,
                "--size", str(size),
                "--facerender", facerender,
                "--expression_scale", str(expression_scale),
                "--preprocess", preprocess,
                "--device", "cuda",
            ]
            if enhancer:
                cmd.extend(["--enhancer", enhancer])
            if still:
                cmd.append("--still")

            log.info(
                "Running SadTalker: size=%d, facerender=%s",
                size, facerender,
            )

            # Run subprocess (SadTalker requires cwd=sadtalker_dir)
            proc = subprocess.run(
                cmd,
                cwd=str(SADTALKER_DIR),
                capture_output=True,
                text=True,
                timeout=25200,  # 7 hour timeout
            )

            if proc.returncode != 0:
                stderr = proc.stderr.strip()[-500:] if proc.stderr else ""
                raise InferenceError(
                    f"SadTalker exited with code {proc.returncode}: {stderr}"
                )

            self._check_cancel(cancel_flag)

            # Find output video
            mp4s = list(Path(result_dir).rglob("*.mp4"))
            if not mp4s:
                raise InferenceError("SadTalker produced no output video")

            # SadTalker writes to <result_dir>/<timestamp>.mp4
            generated = mp4s[0]

            # Copy to arbiter output dir
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "result.mp4"
            shutil.copy2(str(generated), str(output_path))

            # Probe dimensions
            width, height = self._probe_video_dimensions(str(output_path))

            return {
                "format": "mp4",
                "width": width,
                "height": height,
                "file": "result.mp4",
            }

        except subprocess.TimeoutExpired:
            raise InferenceError("SadTalker subprocess timed out after 7h")
        finally:
            # Clean up temp files
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def estimate_time(self, params: dict) -> float:
        """Estimate inference time in milliseconds.

        SadTalker on CUDA: ~10 fps at 256px = ~100ms per frame.
        At 25fps video, 1 second of audio = 25 frames = 2500ms + ~5s overhead.
        """
        # Try to estimate audio duration
        audio_file = params.get("audio_file", "")
        if audio_file and Path(audio_file).is_file():
            try:
                size = Path(audio_file).stat().st_size
                # Rough WAV estimate: 48000 bytes/s at 16-bit mono
                duration = size / 48000
                return duration * 2500 + 5000
            except Exception:
                pass
        return 15000  # default: 15 seconds

    @staticmethod
    def _probe_video_dimensions(path: str) -> tuple[int, int]:
        """Use ffprobe to get video width and height."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=p=0:s=x",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            parts = result.stdout.strip().split("x")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
        return 256, 256
