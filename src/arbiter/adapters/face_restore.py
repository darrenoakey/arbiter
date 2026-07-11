"""Face-restore adapter — GFPGAN per-frame video face restoration.

Decodes an input mp4 via ffmpeg into raw RGB frames, runs GFPGAN on each
frame to clean up distorted/AI-generated faces, then re-encodes the result
back to mp4 (muxing audio from the original).

Expected params:
    video_file : str  — absolute spark path to input mp4 (must be staged)
    weight     : float — 0.0 = max restoration, 1.0 = most faithful to original (default 0.5)

Output dict:
    {"file": "result.mp4", "format": "mp4"}
"""
from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

from arbiter.adapters.base import (
    CancelledException,
    InferenceError,
    LoadError,
    ModelAdapter,
)
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)


@register
class FaceRestoreAdapter(ModelAdapter):
    """GFPGAN face-restoration over video frames."""

    model_id = "face-restore"

    def __init__(self):
        self._restorer = None
        self._device: str = "cuda"
        self._gpu_lock = threading.Lock()

    def load(self, device: str = "cuda") -> None:
        self._device = device
        try:
            from gfpgan import GFPGANer  # type: ignore
        except ImportError as e:
            raise LoadError(f"gfpgan import failed: {e}")

        try:
            # GFPGANv1.4 is the latest stable; auto-downloads on first use
            # to ~/.cache/gfpgan or similar.
            self._restorer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            log.info("face-restore: GFPGANv1.4 loaded on %s", device)
        except Exception as e:
            self._restorer = None
            raise LoadError(f"GFPGANer init failed: {e}") from e

    def unload(self) -> None:
        log.info("Unloading face-restore")
        if self._restorer is not None:
            # GFPGANer holds references to face_helper, face_parse, gfpgan nets
            del self._restorer
            self._restorer = None
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        import numpy as np

        if self._restorer is None:
            raise InferenceError("face-restore not loaded")

        self._check_cancel(cancel_flag)

        video_file = params.get("video_file")
        if not video_file or not Path(video_file).exists():
            raise InferenceError(f"video_file missing or not found: {video_file}")

        weight = float(params.get("weight", 0.5))
        if not 0.0 <= weight <= 1.0:
            raise InferenceError(f"weight must be in [0, 1], got {weight}")

        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "result.mp4"

        # Probe input video dims + fps
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "csv=p=0",
                str(video_file),
            ],
            capture_output=True, text=True, check=True,
        )
        w_str, h_str, fps_str = probe.stdout.strip().split(",")
        width = int(w_str)
        height = int(h_str)
        fr_num, fr_den = fps_str.split("/")
        fps = float(fr_num) / float(fr_den) if float(fr_den) > 0 else 25.0
        frame_bytes = width * height * 3

        log.info("face-restore: input %dx%d @ %.2f fps, weight=%.2f", width, height, fps, weight)

        # PHASE 1 (CPU): spawn ffmpeg to decode video into rawvideo rgb24 bytes
        dec = subprocess.Popen(
            [
                "ffmpeg", "-v", "error", "-i", str(video_file),
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        # PHASE 3 (CPU): spawn ffmpeg to encode raw rgb back into a temp video
        tmp_video = str(output_dir / "_restored_video.mp4")
        enc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", f"{fps}",
                "-i", "-",
                "-c:v", "h264_nvenc", "-preset", "fast", "-pix_fmt", "yuv420p",
                tmp_video,
            ],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if dec.stdout is None or enc.stdin is None:
            raise InferenceError("ffmpeg pipes failed to open")

        frame_count = 0
        faces_found = 0
        try:
            while True:
                self._check_cancel(cancel_flag)
                chunk = dec.stdout.read(frame_bytes)
                if len(chunk) < frame_bytes:
                    break
                rgb = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
                # GFPGAN expects BGR
                bgr = rgb[:, :, ::-1].copy()

                # PHASE 2 (GPU): GFPGAN enhance, under the GPU lock
                with self._gpu_lock:
                    _, _, restored_bgr = self._restorer.enhance(
                        bgr,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True,
                        weight=weight,
                    )
                if restored_bgr is None:
                    # No face detected — keep original
                    restored_bgr = bgr
                else:
                    faces_found += 1

                restored_rgb = restored_bgr[:, :, ::-1].copy()
                enc.stdin.write(restored_rgb.tobytes())
                frame_count += 1
                if frame_count % 100 == 0:
                    log.info("face-restore: processed %d frames (%d with faces)", frame_count, faces_found)
        except CancelledException:
            dec.kill()
            enc.kill()
            raise
        except Exception as e:
            dec.kill()
            enc.kill()
            raise InferenceError(f"face-restore inner loop failed: {e}") from e

        dec.stdout.close()
        enc.stdin.close()
        dec.wait()
        enc.wait()
        if enc.returncode != 0:
            err = enc.stderr.read().decode()[-500:] if enc.stderr else ""
            raise InferenceError(f"ffmpeg encode failed: {err}")

        # Mux the restored video with the original audio track
        mux = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", tmp_video,
                "-i", str(video_file),
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v", "-map", "1:a",
                "-shortest",
                str(result_path),
            ],
            capture_output=True,
        )
        Path(tmp_video).unlink(missing_ok=True)
        if mux.returncode != 0:
            err = mux.stderr.decode()[-500:]
            raise InferenceError(f"ffmpeg mux failed: {err}")

        log.info(
            "face-restore done: %d frames, %d with faces (%.1f%%)",
            frame_count, faces_found,
            100.0 * faces_found / max(frame_count, 1),
        )
        return {
            "format": "mp4",
            "file": "result.mp4",
            "frames": frame_count,
            "faces_restored": faces_found,
            "width": width,
            "height": height,
            "fps": fps,
        }

    def estimate_time(self, params: dict) -> float:
        return 600_000.0  # 10 min default
