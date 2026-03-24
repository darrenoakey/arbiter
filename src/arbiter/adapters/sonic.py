"""Sonic talking-head video generation adapter.

Loads 8 sub-models atomically (SVD VAE, SVD image encoder, Sonic UNet with IP
adapter, Audio2Token, Audio2Bucket, Whisper Tiny, YOLOFace, RIFE) and bridges
between the Arbiter byte-oriented API and Sonic's file-path-oriented internals
using temporary files.
"""
from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from arbiter.adapters.base import GroupAdapter, InferenceError, LoadError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

SONIC_DIR = Path("/home/darren/src/talking-head/Sonic")


@register
class SonicAdapter(GroupAdapter):
    model_id = "sonic"

    def __init__(self):
        self._sonic = None
        self._device = None

    # ------------------------------------------------------------------
    # load / unload
    # ------------------------------------------------------------------
    def load(self, device: str = "cuda") -> None:
        """Load all 8 Sonic sub-models atomically."""
        log.info("Loading Sonic sub-models on %s ...", device)

        # Sonic imports rely on its own repo root being on sys.path
        sonic_dir = str(SONIC_DIR)
        if sonic_dir not in sys.path:
            sys.path.insert(0, sonic_dir)

        try:
            from sonic import Sonic

            device_id = 0
            if device.startswith("cuda:"):
                device_id = int(device.split(":")[1])
            elif device == "cuda":
                device_id = 0
            elif device == "cpu":
                device_id = -1

            self._sonic = Sonic(device_id=device_id, enable_interpolate_frame=True)
            self._device = device
            log.info("Sonic ready (%s).", device)
        except Exception as exc:
            # Atomic guarantee: clean up anything partially loaded
            self._sonic = None
            self._cleanup_gpu()
            raise LoadError(f"Failed to load Sonic: {exc}") from exc

    def unload(self) -> None:
        """Release all Sonic sub-models and free GPU memory."""
        log.info("Unloading Sonic.")
        if self._sonic is not None:
            # Explicitly delete heavy attributes so refcounts drop immediately
            for attr in (
                "pipe",
                "whisper",
                "audio2token",
                "audio2bucket",
                "image_encoder",
                "feature_extractor",
                "face_det",
                "rife",
            ):
                if hasattr(self._sonic, attr):
                    delattr(self._sonic, attr)
            del self._sonic
            self._sonic = None
        self._device = None
        self._cleanup_gpu()

    # ------------------------------------------------------------------
    # infer
    # ------------------------------------------------------------------
    def infer(
        self,
        params: dict,
        output_dir: Path,
        cancel_flag: threading.Event,
    ) -> dict:
        """Generate a talking-head video from a portrait image and audio clip.

        params:
            image:          base64-encoded PNG/JPEG portrait
            audio:          base64-encoded WAV audio
            dynamic_scale:  (optional) motion intensity, default 1.0
            seed:           (optional) RNG seed
            min_resolution: (optional) minimum resolution, default 256
        """
        if self._sonic is None:
            raise InferenceError("Sonic model is not loaded")

        self._check_cancel(cancel_flag)

        # ---- decode inputs to temp files ----
        image_b64 = params.get("image", "")
        audio_b64 = params.get("audio", "")

        if not image_b64 and not params.get("image_file"):
            raise InferenceError("Missing required parameter: image or image_file")
        if not audio_b64 and not params.get("audio_file"):
            raise InferenceError("Missing required parameter: audio or audio_file")

        # Strip data-URI prefix if present
        if image_b64.startswith("data:"):
            _, image_b64 = image_b64.split(",", 1)
        if audio_b64.startswith("data:"):
            _, audio_b64 = audio_b64.split(",", 1)

        tmp_dir = tempfile.mkdtemp(prefix="sonic_")
        tmp_image = os.path.join(tmp_dir, "input.png")
        tmp_audio = os.path.join(tmp_dir, "input.wav")

        try:
            try:
                with open(tmp_image, "wb") as f:
                    f.write(self._resolve_media(params, "image"))
            except InferenceError:
                raise
            except Exception as exc:
                raise InferenceError(f"Failed to decode image: {exc}") from exc

            try:
                with open(tmp_audio, "wb") as f:
                    f.write(self._resolve_media(params, "audio"))
            except InferenceError:
                raise
            except Exception as exc:
                raise InferenceError(f"Failed to decode audio: {exc}") from exc

            self._check_cancel(cancel_flag)

            # ---- face detection / preprocessing ----
            try:
                face_info = self._sonic.preprocess(tmp_image, expand_ratio=0.5)
            except Exception as exc:
                raise InferenceError(f"Face preprocessing failed: {exc}") from exc

            if face_info["face_num"] <= 0:
                raise InferenceError(
                    "No face detected in the input image. Please provide a clear "
                    "portrait photo with a visible face."
                )

            self._check_cancel(cancel_flag)

            # ---- run Sonic pipeline ----
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / "result.mp4")

            dynamic_scale = float(params.get("dynamic_scale", 1.0))
            seed = params.get("seed")
            if seed is not None:
                seed = int(seed)
            min_resolution = int(params.get("min_resolution", 512))

            rc = self._sonic.process(
                image_path=tmp_image,
                audio_path=tmp_audio,
                output_path=output_path,
                min_resolution=min_resolution,
                inference_steps=int(params.get("inference_steps", 25)),
                dynamic_scale=dynamic_scale,
                seed=seed,
            )

            if rc != 0:
                raise InferenceError(
                    f"Sonic pipeline returned error code {rc}. Face detection may "
                    "have failed during tensor preparation."
                )

            if not os.path.isfile(output_path):
                raise InferenceError("Sonic did not produce an output video file.")

            self._check_cancel(cancel_flag)

            # ---- extract output dimensions ----
            width, height = self._probe_video_dimensions(output_path)

            return {
                "format": "mp4",
                "width": width,
                "height": height,
                "file": "result.mp4",
            }

        finally:
            # Clean up temp files
            for p in (tmp_image, tmp_audio):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # estimate_time
    # ------------------------------------------------------------------
    def estimate_time(self, params: dict) -> float:
        """Estimate inference time in milliseconds.

        Heuristic: ~2500 ms per second of audio.
        """
        audio_file = params.get("audio_file", "")
        audio_b64 = params.get("audio", "")

        try:
            if audio_file and Path(audio_file).is_file():
                audio_bytes = Path(audio_file).read_bytes()
            else:
                if audio_b64.startswith("data:"):
                    _, audio_b64 = audio_b64.split(",", 1)
                audio_bytes = base64.b64decode(audio_b64)
            duration = self._wav_duration(audio_bytes)
        except Exception:
            # Fallback: assume 5 seconds of audio
            duration = 5.0

        return duration * 2500.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _wav_duration(raw: bytes) -> float:
        """Estimate WAV duration from raw bytes using the header."""
        # Standard WAV: bytes 28-31 = byte rate, bytes 40-43 = data chunk size
        if len(raw) < 44 or raw[:4] != b"RIFF":
            return 5.0  # not a WAV or too short; fallback
        byte_rate = int.from_bytes(raw[28:32], "little")
        if byte_rate == 0:
            return 5.0
        data_size = int.from_bytes(raw[40:44], "little")
        return data_size / byte_rate

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
        # Fallback
        return 512, 512
