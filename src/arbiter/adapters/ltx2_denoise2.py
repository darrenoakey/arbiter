"""LTX-2 denoise stage 2 adapter — final denoising + video decode.

Reads a stage-1 output file (upscaled latent + contexts + stage2 conditionings)
and runs the stage-2 transformer + video VAE decoder to produce an mp4 chunk.

This is the last stage of the 3-stage split (encode → denoise1 → denoise2).

Input file produced by either:
  - video_fast_gpu.FastPipeline.run_stage1()        (legacy stage1_output.pt)
  - ltx2_denoise1 adapter                            (denoise1_output.pt)

Both produce the same torch.save() dict format.

Expected params dict:
    denoise1_file : str  — absolute path to the .pt file from denoise1/run_stage1
    audio_file    : str  — absolute path to audio mp3/wav on spark local disk
    start_time    : float — chunk start in seconds
    end_time      : float — chunk end in seconds
    fps           : int   — output frame rate
    width         : int   — exact requested output width after model-only padding
    height        : int   — exact requested output height after model-only padding
"""
from __future__ import annotations

import gc
import logging
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from arbiter.adapters.base import (
    CancelledException,
    GroupAdapter,
    HeapTrimGuard,
    InferenceError,
    LoadError,
)
from arbiter.adapters.registry import register
from arbiter.adapters.ltx2_clock import native_frame_rate

log = logging.getLogger(__name__)

LTX2_SPARK_DIR = Path("/home/darren/src/ltx2-spark")


def _crop_frames_to_target(frames, target_width: int, target_height: int):
    """Center-crop model-padded frames back to the exact public dimensions."""
    height, width = frames.shape[1:3]
    if target_width <= 0 or target_height <= 0:
        raise InferenceError("target video dimensions must be positive")
    if target_width > width or target_height > height:
        raise InferenceError(
            f"target {target_width}x{target_height} exceeds decoded model frame {width}x{height}"
        )
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    return frames[:, top:top + target_height, left:left + target_width, :]


def _assemble_single_chunk_mp4(frames, audio_path, start_time, end_time, output_path, fps):
    """Assemble numpy frames + audio slice into an mp4 via ffmpeg nvenc."""
    import os

    tmp_vid = tempfile.mktemp(suffix=".mp4")
    h, w = frames.shape[1], frames.shape[2]
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-r", str(fps), "-pix_fmt", "rgb24",
            "-i", "-",
            "-c:v", "h264_nvenc", "-preset", "fast", "-pix_fmt", "yuv420p",
            tmp_vid,
        ],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode()[-500:]
        raise InferenceError(f"ffmpeg nvenc failed: {err}")

    duration = end_time - start_time
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", tmp_vid,
            "-ss", str(start_time), "-t", str(duration), "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v", "-map", "1:a", "-shortest", str(output_path),
        ],
        capture_output=True,
    )
    os.unlink(tmp_vid)
    if result.returncode != 0:
        err = result.stderr.decode()[-500:]
        raise InferenceError(f"ffmpeg mux failed: {err}")


@register
class LTX2Denoise2Adapter(GroupAdapter):
    """Stage-2 transformer + video decoder, both cached across jobs."""

    model_id = "ltx2-denoise2"

    def __init__(self):
        self._pipeline = None
        self._device: str = "cuda"
        # Serialises the GPU phase only. The biggest pipelining win on this
        # adapter is overlapping ffmpeg mp4 encode/mux (all CPU/nvenc, slow)
        # with the next job's GPU denoise + decode.
        self._gpu_lock = threading.Lock()

    def load(self, device: str = "cuda") -> None:
        self._device = device

        spark_str = str(LTX2_SPARK_DIR)
        if spark_str not in sys.path:
            sys.path.insert(0, spark_str)

        try:
            import ltx_core  # noqa: F401
            import ltx_pipelines  # noqa: F401
        except ImportError as e:
            raise LoadError(f"ltx_core / ltx_pipelines not importable: {e}")

        try:
            from video_fast_gpu import FastPipeline
            self._pipeline = FastPipeline()

            # Pre-load stage-2 transformer + video decoder so they are cached
            # across chunks (key speedup — 40GB of weights stay resident).
            log.info("LTX2-denoise2: pre-loading stage-2 transformer + decoder")
            with HeapTrimGuard():
                self._pipeline._s2_transformer = self._pipeline.stage_2_ledger.transformer()
                self._pipeline._s2_decoder = self._pipeline.stage_1_ledger.video_decoder()
            self._pipeline._s2_loaded = True
            log.info("LTX2-denoise2: models loaded")
        except Exception as e:
            self._pipeline = None
            raise LoadError(f"Failed to load denoise2 models: {e}") from e

    def unload(self) -> None:
        log.info("Unloading LTX2-denoise2")
        if self._pipeline is not None:
            for attr in ("_s2_transformer", "_s2_decoder"):
                if hasattr(self._pipeline, attr):
                    delattr(self._pipeline, attr)
            self._pipeline._s2_loaded = False
            if hasattr(self._pipeline, "components"):
                del self._pipeline.components
            if hasattr(self._pipeline, "stage_1_ledger"):
                del self._pipeline.stage_1_ledger
            if hasattr(self._pipeline, "stage_2_ledger"):
                del self._pipeline.stage_2_ledger
            del self._pipeline
            self._pipeline = None
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        if self._pipeline is None:
            raise InferenceError("denoise2 pipeline not loaded")

        self._check_cancel(cancel_flag)

        denoise1_file = params.get("denoise1_file") or params.get("stage1_file")
        if not denoise1_file:
            raise InferenceError("denoise1_file is required")
        if not Path(denoise1_file).exists():
            raise InferenceError(f"denoise1_file does not exist: {denoise1_file}")

        audio_file = params.get("audio_file")
        if not audio_file or not Path(audio_file).exists():
            raise InferenceError(f"audio_file missing or does not exist: {audio_file}")

        start_time = float(params.get("start_time", 0.0))
        end_time = float(params.get("end_time", 0.0))
        try:
            fps = native_frame_rate(params)
        except ValueError as error:
            raise InferenceError(str(error)) from error

        output_dir.mkdir(parents=True, exist_ok=True)
        tmp_npy = str(output_dir / "frames.npy")

        def _progress(stage, status, **kw):
            log.info("denoise2 progress: %s/%s %s", stage, status, kw)
            if cancel_flag.is_set():
                raise CancelledException(f"Cancelled during {stage}/{status}")

        try:
            # PHASE 1 (CPU/IO): torch.load stage1_output.pt.
            data = self._pipeline.load_stage2_input(denoise1_file)
            self._check_cancel(cancel_flag)

            # PHASE 2 (GPU): stage-2 denoise loop + video decode.
            # Serialised; the next job's load_stage2_input overlaps with this.
            with self._gpu_lock:
                self._check_cancel(cancel_flag)
                frames_np = self._pipeline.run_stage2_gpu(data, progress_fn=_progress)
            frames_np = _crop_frames_to_target(
                frames_np,
                int(params.get("width", frames_np.shape[2])),
                int(params.get("height", frames_np.shape[1])),
            )
        except CancelledException:
            raise
        except Exception as e:
            raise InferenceError(f"run_stage2 failed: {e}") from e

        self._check_cancel(cancel_flag)

        # PHASE 3 (CPU/nvenc): ffmpeg encode raw frames + mux audio into mp4.
        # Runs OUTSIDE the GPU lock — the next job can hold the lock and do
        # its GPU denoise while this thread finishes ffmpeg.
        result_path = output_dir / "result.mp4"
        _assemble_single_chunk_mp4(
            frames=frames_np,
            audio_path=audio_file,
            start_time=start_time,
            end_time=end_time,
            output_path=result_path,
            fps=fps,
        )

        # Clean up the temp .npy (it's no longer written; keep the unlink a no-op
        # for backwards compat in case an old path lingers).
        Path(tmp_npy).unlink(missing_ok=True)

        h, w = frames_np.shape[1], frames_np.shape[2]
        duration = round(len(frames_np) / fps, 2)

        # Force GPU cleanup between chunks (keep cached models but free tensors)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return {
            "format": "mp4",
            "file": "result.mp4",
            "width": w,
            "height": h,
            "fps": fps,
            "duration_seconds": duration,
            "total_frames": int(len(frames_np)),
        }

    def estimate_time(self, params: dict) -> float:
        return 120_000.0
