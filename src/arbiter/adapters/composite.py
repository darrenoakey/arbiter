"""Chroma-key compositing adapter using ffmpeg with NVENC hardware encoding.

Composites a green-screen talking head video onto a background image.
No GPU model to load — just runs ffmpeg with h264_nvenc for fast encoding.

Filter chain runs on CPU (eq, chromakey, despill have no CUDA equivalents),
but NVENC encode is the big win on long videos.
"""
from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

from arbiter.adapters.base import ModelAdapter, InferenceError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)


@register
class CompositeAdapter(ModelAdapter):
    model_id = "composite"

    def load(self, device: str = "cuda") -> None:
        # Verify ffmpeg has NVENC support
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        if "h264_nvenc" not in result.stdout:
            from arbiter.adapters.base import LoadError
            raise LoadError("ffmpeg missing h264_nvenc encoder")
        log.info("Composite adapter ready (h264_nvenc available).")

    def unload(self) -> None:
        log.info("Composite adapter unloaded.")

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        """Composite a green-screen video onto a background image.

        params:
            video_file: path to talking head video (green screen)
            background_file: path to background image
            video_width: output width (default 1920)
            video_height: output height (default 1080)
            head_height: height to scale talking head to (default video_height)
            chromakey_color: hex color e.g. "00b000" (default: auto-detect)
            chromakey_similarity: float (default 0.18)
            chromakey_blend: float (default 0.05)
            contrast: float (default 1.4)
            saturation: float (default 1.3)
            brightness: float (default -0.05)
            crf: int (default 23)
            fps: int (default 25)
        """
        self._check_cancel(cancel_flag)

        video_file = params.get("video_file")
        bg_file = params.get("background_file")

        if not video_file or not Path(video_file).is_file():
            raise InferenceError(f"video_file not found: {video_file}")
        if not bg_file or not Path(bg_file).is_file():
            raise InferenceError(f"background_file not found: {bg_file}")

        video_width = int(params.get("video_width", 1920))
        video_height = int(params.get("video_height", 1080))
        head_h = int(params.get("head_height", video_height))
        head_w = head_h  # square
        x_pos = (video_width - head_w) // 2
        y_pos = int(params.get("y_pos", 0))

        similarity = float(params.get("chromakey_similarity", 0.18))
        blend = float(params.get("chromakey_blend", 0.05))
        contrast = float(params.get("contrast", 1.4))
        saturation = float(params.get("saturation", 1.3))
        brightness = float(params.get("brightness", -0.05))
        crf = int(params.get("crf", 23))
        fps = int(params.get("fps", 25))

        # Auto-detect or use provided chroma key color
        green_hex = params.get("chromakey_color")
        if not green_hex:
            green_hex = self._detect_green(video_file)

        # Get video duration
        duration = self._probe_duration(video_file)
        if duration < 0.1:
            raise InferenceError(f"Video too short: {duration:.2f}s")

        filter_complex = (
            f"[1:v]scale={head_w}:{head_h}:flags=lanczos,"
            f"eq=contrast={contrast}:saturation={saturation}:brightness={brightness},"
            f"chromakey=0x{green_hex}:{similarity}:{blend},"
            f"despill=type=green:mix=0.5:expand=0[fg];"
            f"[0:v]scale={video_width}:{video_height}[bg];"
            f"[bg][fg]overlay={x_pos}:{y_pos}"
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "result.mp4"

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-loop", "1", "-i", bg_file,
            "-i", video_file,
            "-filter_complex", filter_complex,
            "-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf),
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-an",
            "-t", str(duration),
            str(out_path),
        ]

        log.info(
            "Compositing %.1fs video (%dx%d), green=#%s, NVENC encode ...",
            duration, video_width, video_height, green_hex,
        )

        self._check_cancel(cancel_flag)

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            raise InferenceError(f"ffmpeg failed: {proc.stderr[:500]}")

        if not out_path.exists() or out_path.stat().st_size < 1000:
            raise InferenceError("ffmpeg produced no output")

        # Verify output
        out_dur = self._probe_duration(str(out_path))
        log.info("Composite done: %.1fs, %d bytes", out_dur, out_path.stat().st_size)

        return {
            "format": "mp4",
            "width": video_width,
            "height": video_height,
            "duration": out_dur,
            "file": "result.mp4",
        }

    def estimate_time(self, params: dict) -> float:
        video_file = params.get("video_file", "")
        if video_file and Path(video_file).is_file():
            duration = self._probe_duration(video_file)
            return duration * 100 + 1000
        return 30000

    @staticmethod
    def _probe_duration(path: str) -> float:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=30,
            )
            return float(result.stdout.strip()) if result.stdout.strip() else 0
        except Exception:
            return 0

    @staticmethod
    def _detect_green(video_path: str) -> str:
        """Sample corner pixels of first frame to detect chroma key color."""
        import tempfile
        tmp = tempfile.mktemp(suffix=".png")
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-i", video_path, "-vframes", "1", "-q:v", "2", tmp],
                capture_output=True, timeout=30,
            )
            from PIL import Image
            img = Image.open(tmp).convert("RGB")
            w, h = img.size
            samples = []
            for x, y in [(5, 5), (5, h - 5), (w - 5, 5), (w // 2, 5), (5, h // 2)]:
                r, g, b = img.getpixel((x, y))[:3]
                if g > 100 and g > r * 1.3 and g > b * 1.3:
                    samples.append((r, g, b))
            if samples:
                avg_r = sum(s[0] for s in samples) // len(samples)
                avg_g = sum(s[1] for s in samples) // len(samples)
                avg_b = sum(s[2] for s in samples) // len(samples)
                return f"{avg_r:02X}{avg_g:02X}{avg_b:02X}"
        except Exception as e:
            log.warning("Green detection failed, using default: %s", e)
        finally:
            import os
            if os.path.exists(tmp):
                os.unlink(tmp)
        return "00b000"
