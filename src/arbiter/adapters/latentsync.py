"""LatentSync lip-sync refinement adapter (subprocess-based).

Takes a video + audio and re-inpaints the lip region using diffusion-based
latent space inpainting for accurate lip synchronization. Typically used as
a post-processing step after SadTalker or other talking-head generators.
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

LATENTSYNC_DIR = Path("/home/darren/src/talking-head/latentsync")
LATENTSYNC_PYTHON = LATENTSYNC_DIR / ".venv" / "bin" / "python"
LATENTSYNC_UNET_CONFIG = LATENTSYNC_DIR / "configs" / "unet" / "stage2_512.yaml"
LATENTSYNC_CHECKPOINT = LATENTSYNC_DIR / "checkpoints" / "latentsync_unet.pt"
LATENTSYNC_WHISPER = LATENTSYNC_DIR / "checkpoints" / "whisper" / "tiny.pt"


@register
class LatentSyncAdapter(ModelAdapter):
    model_id = "latentsync"

    def __init__(self):
        self._ready = False

    def load(self, device: str = "cuda") -> None:
        """Verify LatentSync installation. No GPU memory held after load."""
        log.info("Verifying LatentSync installation...")

        if not LATENTSYNC_PYTHON.exists():
            raise LoadError(f"LatentSync venv not found: {LATENTSYNC_PYTHON}")
        if not LATENTSYNC_UNET_CONFIG.exists():
            raise LoadError(f"LatentSync config not found: {LATENTSYNC_UNET_CONFIG}")
        if not LATENTSYNC_CHECKPOINT.exists():
            raise LoadError(f"LatentSync checkpoint not found: {LATENTSYNC_CHECKPOINT}")
        if not LATENTSYNC_WHISPER.exists():
            raise LoadError(f"LatentSync whisper model not found: {LATENTSYNC_WHISPER}")

        self._ready = True
        log.info("LatentSync ready (subprocess mode, no persistent GPU usage).")

    def unload(self) -> None:
        """No-op — LatentSync doesn't hold GPU memory between calls."""
        log.info("LatentSync unloaded (no GPU memory was held).")
        self._ready = False

    def infer(
        self,
        params: dict,
        output_dir: Path,
        cancel_flag: threading.Event,
    ) -> dict:
        """Refine lip sync on a video using LatentSync.

        params:
            video / video_file: input video (from SadTalker or similar)
            audio / audio_file: driving audio
            inference_steps: diffusion steps (default 20)
            guidance_scale: classifier-free guidance (default 1.5)
        """
        if not self._ready:
            raise InferenceError("LatentSync is not loaded (call load first)")

        self._check_cancel(cancel_flag)

        # Resolve inputs
        try:
            video_bytes = self._resolve_media(params, "video")
        except Exception as e:
            raise InferenceError(f"Failed to resolve video: {e}") from e
        try:
            audio_bytes = self._resolve_media(params, "audio")
        except Exception as e:
            raise InferenceError(f"Failed to resolve audio: {e}") from e

        inference_steps = int(params.get("inference_steps", 20))
        guidance_scale = float(params.get("guidance_scale", 1.5))

        tmp_dir = tempfile.mkdtemp(prefix="latentsync_")
        tmp_video = os.path.join(tmp_dir, "input.mp4")
        tmp_audio = os.path.join(tmp_dir, "input.wav")
        tmp_output = os.path.join(tmp_dir, "output.mp4")

        try:
            with open(tmp_video, "wb") as f:
                f.write(video_bytes)
            with open(tmp_audio, "wb") as f:
                f.write(audio_bytes)

            self._check_cancel(cancel_flag)

            cmd = [
                str(LATENTSYNC_PYTHON),
                "-m", "scripts.inference",
                "--unet_config_path", str(LATENTSYNC_UNET_CONFIG),
                "--inference_ckpt_path", str(LATENTSYNC_CHECKPOINT),
                "--video_path", tmp_video,
                "--audio_path", tmp_audio,
                "--video_out_path", tmp_output,
                "--inference_steps", str(inference_steps),
                "--guidance_scale", str(guidance_scale),
                "--enable_deepcache",
            ]

            log.info(
                "Running LatentSync: steps=%d, guidance=%.1f",
                inference_steps, guidance_scale,
            )

            proc = subprocess.run(
                cmd,
                cwd=str(LATENTSYNC_DIR),
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
            )

            if proc.returncode != 0:
                stderr = proc.stderr.strip()[-500:] if proc.stderr else ""
                raise InferenceError(
                    f"LatentSync exited with code {proc.returncode}: {stderr}"
                )

            if not os.path.isfile(tmp_output):
                raise InferenceError("LatentSync did not produce an output video")

            self._check_cancel(cancel_flag)

            # Copy to arbiter output dir
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "result.mp4"
            shutil.copy2(tmp_output, str(output_path))

            width, height = self._probe_video_dimensions(str(output_path))

            return {
                "format": "mp4",
                "width": width,
                "height": height,
                "file": "result.mp4",
            }

        except subprocess.TimeoutExpired:
            raise InferenceError("LatentSync subprocess timed out after 1800s")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def estimate_time(self, params: dict) -> float:
        """Estimate inference time in milliseconds.

        LatentSync: ~50s per second of video at 512px with 20 steps.
        """
        video_file = params.get("video_file", "")
        if video_file and Path(video_file).is_file():
            try:
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", video_file],
                    capture_output=True, text=True, timeout=10,
                )
                duration = float(result.stdout.strip())
                return duration * 50000 + 10000  # 50s/s + 10s overhead
            except Exception:
                pass
        return 60000  # default: 60 seconds

    @staticmethod
    def _probe_video_dimensions(path: str) -> tuple[int, int]:
        """Use ffprobe to get video width and height."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=p=0:s=x",
                    path,
                ],
                capture_output=True, text=True, timeout=10,
            )
            parts = result.stdout.strip().split("x")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
        return 512, 512
