"""LTX-2 video generation adapter — GroupAdapter with phased model loading.

This adapter wraps the ltx2-spark FastPipeline, which internally loads/unloads
sub-models across 7 phases to fit the 19B-parameter transformer within GPU memory.
From the memory manager's perspective the adapter occupies its full memory_gb
(~55 GB) for the entire duration of a job; phased loading is an internal
optimization invisible to the scheduler.

Expected params dict:
    prompt         : str          — per-chunk or single prompt
    segments       : list[dict]   — [{description, start_time, end_time, start_image_b64, end_image_b64}, ...]
    audio_b64      : str          — base64-encoded audio file (mp3/wav/flac)
    resolution     : str          — "small" | "small-portrait" | "large" | "large-portrait"
    fps            : int          — frame rate (default 24)
    seed           : int          — RNG seed (default 42)
    chunk_frames   : int          — frames per chunk (default 121)
"""
from __future__ import annotations

import base64
import gc
import io
import logging
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from arbiter.adapters.base import (
    CancelledException,
    GroupAdapter,
    InferenceError,
    LoadError,
)
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

LTX2_SPARK_DIR = Path("/home/darren/src/ltx2-spark")


def _get_audio_duration(audio_path: str) -> float:
    """Probe audio duration with ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _assemble_mp4(
    chunks_dir: Path,
    num_chunks: int,
    audio_path: str | None,
    output_path: Path,
    fps: int,
) -> tuple[int, int, int]:
    """Assemble .npy chunk files into an MP4. Returns (total_frames, height, width)."""
    import numpy as np

    all_frames = []
    for i in range(num_chunks):
        chunk_npy = chunks_dir / f"chunk_{i:03d}.npy"
        chunk_frames = np.load(str(chunk_npy))
        if i == 0:
            all_frames.append(chunk_frames)
        else:
            # Skip first frame (overlap with previous chunk's last frame)
            all_frames.append(chunk_frames[1:])

    video_frames = np.concatenate(all_frames, axis=0)
    total_frames, h, w = video_frames.shape[0], video_frames.shape[1], video_frames.shape[2]
    log.info("Assembling %d frames (%dx%d) at %d fps", total_frames, w, h, fps)

    # Encode raw frames to temp video via ffmpeg pipe
    temp_video = str(output_path) + ".temp.mp4"
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            temp_video,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    batch_size = 50
    for start in range(0, len(video_frames), batch_size):
        batch = video_frames[start:start + batch_size]
        try:
            proc.stdin.write(batch.tobytes())
        except BrokenPipeError:
            break
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise InferenceError(f"ffmpeg encode failed (rc={proc.returncode}): {stderr[-500:]}")

    # Mux with audio if available
    if audio_path:
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", temp_video, "-i", audio_path,
                    "-c:v", "copy", "-c:a", "aac", "-shortest",
                    str(output_path),
                ],
                check=True, capture_output=True,
            )
            Path(temp_video).unlink(missing_ok=True)
        except subprocess.CalledProcessError:
            log.warning("Audio mux failed; saving video-only output")
            Path(temp_video).rename(output_path)
    else:
        Path(temp_video).rename(output_path)

    return total_frames, h, w


@register
class LTX2Adapter(GroupAdapter):
    """LTX-2 video generation with phased sub-model loading.

    load()   — prepares lightweight config objects (ModelLedger, PipelineComponents).
    unload() — tears down everything and frees GPU.
    infer()  — runs the full 7-phase FastPipeline internally, managing 41 GB+
               transformer loads within the call.
    """

    model_id = "ltx2"

    def __init__(self):
        self._pipeline = None  # FastPipeline instance
        self._device: str = "cuda"
        self._tmp_dir: tempfile.TemporaryDirectory | None = None

    # ------------------------------------------------------------------
    # load / unload
    # ------------------------------------------------------------------

    def load(self, device: str = "cuda") -> None:
        """Set up sys.path and create the FastPipeline config objects.

        This loads only the lightweight ledger/component metadata — NOT the
        heavy transformer weights (those are loaded per-phase inside infer).
        """
        self._device = device

        # Ensure ltx2-spark is importable
        spark_str = str(LTX2_SPARK_DIR)
        if spark_str not in sys.path:
            sys.path.insert(0, spark_str)
            log.info("Added %s to sys.path", spark_str)

        try:
            # Verify core packages are importable
            import ltx_core  # noqa: F401
            import ltx_pipelines  # noqa: F401
        except ImportError as e:
            raise LoadError(
                f"ltx_core / ltx_pipelines not importable. "
                f"Ensure they are installed in the current environment: {e}"
            )

        try:
            from video_fast_gpu import FastPipeline

            self._pipeline = FastPipeline()
            log.info("LTX-2 FastPipeline config objects created (no heavy weights yet)")
        except Exception as e:
            self._pipeline = None
            raise LoadError(f"Failed to create FastPipeline: {e}") from e

    def unload(self) -> None:
        """Release all pipeline objects and GPU memory."""
        log.info("Unloading LTX-2 adapter")
        if self._pipeline is not None:
            # Delete internal ledger/component refs
            if hasattr(self._pipeline, "components"):
                del self._pipeline.components
            if hasattr(self._pipeline, "stage_1_ledger"):
                del self._pipeline.stage_1_ledger
            if hasattr(self._pipeline, "stage_2_ledger"):
                del self._pipeline.stage_2_ledger
            del self._pipeline
            self._pipeline = None

        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None

        self._cleanup_gpu()

    # ------------------------------------------------------------------
    # infer
    # ------------------------------------------------------------------

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        """Generate a video from segments + audio using phased FastPipeline.

        Steps:
          1. Decode base64 audio and images to temp files
          2. Build the chunk plan
          3. Run FastPipeline.generate_all_chunks (phases 1-7)
          4. Assemble .npy chunks into result.mp4
        """
        import numpy as np
        from PIL import Image

        if self._pipeline is None:
            raise InferenceError("LTX-2 pipeline not loaded — call load() first")

        self._check_cancel(cancel_flag)

        # -- Read params -----------------------------------------------
        segments = params.get("segments", [])
        audio_b64 = params.get("audio_b64", "")
        resolution = params.get("resolution", "large")
        fps = int(params.get("fps", 24))
        seed = int(params.get("seed", 42))
        chunk_frames = int(params.get("chunk_frames", 121))

        from constants import RESOLUTION_PRESETS
        if resolution not in RESOLUTION_PRESETS:
            raise InferenceError(
                f"Unknown resolution '{resolution}'. "
                f"Choose from: {list(RESOLUTION_PRESETS.keys())}"
            )
        height, width = RESOLUTION_PRESETS[resolution]

        # -- Set up temp workspace -------------------------------------
        output_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="ltx2_")
        work_dir = Path(self._tmp_dir.name)
        chunks_dir = work_dir / "chunks"
        chunks_dir.mkdir()

        # -- Decode audio to file --------------------------------------
        audio_path: str | None = None
        if audio_b64:
            audio_path = str(work_dir / "audio.mp3")
            raw = audio_b64
            if raw.startswith("data:"):
                _, raw = raw.split(",", 1)
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(raw))
            log.info("Decoded audio to %s", audio_path)

        if not audio_path:
            raise InferenceError("audio_b64 is required for LTX-2 video generation")

        self._check_cancel(cancel_flag)

        audio_duration = _get_audio_duration(audio_path)
        log.info("Audio duration: %.1fs", audio_duration)

        # -- Decode segment images to files ----------------------------
        images_dir = work_dir / "images"
        images_dir.mkdir()

        processed_segments = []
        for idx, seg in enumerate(segments):
            start_img_path = self._decode_image(
                seg.get("start_image_b64", ""),
                images_dir / f"seg_{idx:03d}_start.png",
            )
            end_img_path = self._decode_image(
                seg.get("end_image_b64", ""),
                images_dir / f"seg_{idx:03d}_end.png",
            )
            processed_segments.append({
                "description": seg.get("description", seg.get("prompt", "")),
                "start_time": float(seg.get("start_time", 0)),
                "end_time": float(seg.get("end_time", audio_duration)),
                "start_image": str(start_img_path) if start_img_path else None,
                "end_image": str(end_img_path) if end_img_path else None,
            })

        self._check_cancel(cancel_flag)

        # -- Build chunk plan ------------------------------------------
        from video import plan_chunks

        chunk_plan = plan_chunks(processed_segments, audio_duration, chunk_frames, fps)
        num_chunks = len(chunk_plan)
        log.info(
            "Chunk plan: %d chunks, %dx%d @ %d fps, seed=%d",
            num_chunks, width, height, fps, seed,
        )

        if num_chunks == 0:
            raise InferenceError("Chunk plan produced zero chunks — check segments/audio")

        # -- Progress callback that also checks cancellation -----------
        def _progress(stage: str, status: str, **kwargs):
            log.info("LTX-2 progress: %s/%s %s", stage, status, kwargs)
            if cancel_flag.is_set():
                raise CancelledException(f"Job cancelled during {stage}/{status}")

        # -- Run phased pipeline (phases 1-7) --------------------------
        try:
            self._pipeline.generate_all_chunks(
                chunk_plan=chunk_plan,
                audio_path=audio_path,
                height=height,
                width=width,
                fps=fps,
                seed=seed,
                chunks_dir=str(chunks_dir),
                progress_fn=_progress,
            )
        except CancelledException:
            raise
        except Exception as e:
            raise InferenceError(f"FastPipeline failed: {e}") from e

        self._check_cancel(cancel_flag)

        # -- Assemble into MP4 -----------------------------------------
        result_path = output_dir / "result.mp4"
        total_frames, h, w = _assemble_mp4(
            chunks_dir=chunks_dir,
            num_chunks=num_chunks,
            audio_path=audio_path,
            output_path=result_path,
            fps=fps,
        )

        duration_seconds = round(total_frames / fps, 2)
        log.info("LTX-2 result: %s (%.1fs)", result_path, duration_seconds)

        # Clean up temp workspace
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None

        return {
            "format": "mp4",
            "file": "result.mp4",
            "width": w,
            "height": h,
            "fps": fps,
            "duration_seconds": duration_seconds,
            "total_frames": total_frames,
            "num_chunks": num_chunks,
        }

    # ------------------------------------------------------------------
    # estimate_time
    # ------------------------------------------------------------------

    def estimate_time(self, params: dict) -> float:
        """Rough estimate: ~800 ms per output frame."""
        from constants import RESOLUTION_PRESETS

        resolution = params.get("resolution", "large")
        fps = int(params.get("fps", 24))
        chunk_frames = int(params.get("chunk_frames", 121))
        segments = params.get("segments", [])

        # Quick estimate from segments duration or fall back to chunk count
        if segments:
            max_end = max(float(s.get("end_time", 0)) for s in segments)
            total_frames = round(max_end * fps)
        else:
            total_frames = chunk_frames

        return total_frames * 800.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_image(b64_data: str, dest: Path) -> Path | None:
        """Decode a base64-encoded image string to a file. Returns None if empty."""
        if not b64_data:
            return None

        from PIL import Image

        raw = b64_data
        if raw.startswith("data:"):
            _, raw = raw.split(",", 1)

        img = Image.open(io.BytesIO(base64.b64decode(raw)))
        img.save(str(dest))
        return dest
