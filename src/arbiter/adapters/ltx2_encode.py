"""LTX-2 encode adapter — first stage of the 3-stage video pipeline.

Runs text encoder (Gemma) + audio VAE encoder. Produces an `encoded.pt` file
containing prompt contexts + audio latent, consumed by ltx2-denoise1.

Expected params dict:
    description   : str   — chunk text prompt
    audio_file    : str   — absolute path to audio on spark local disk
    start_time    : float — chunk start in seconds
    end_time      : float — chunk end in seconds
    fps           : int
    seed          : int
    chunk_index   : int
    start_image_file : str — optional path to start keyframe (carried through)
    end_image_file   : str — optional path to end keyframe (carried through)
"""
from __future__ import annotations

import gc
import logging
import sys
import threading
from pathlib import Path

from arbiter.adapters.base import (
    CancelledException,
    GroupAdapter,
    InferenceError,
    LoadError,
)
from arbiter.adapters.registry import register
from arbiter.adapters.ltx2_clock import lattice_frame_count, native_frame_rate

log = logging.getLogger(__name__)

LTX2_SPARK_DIR = Path("/home/darren/src/ltx2-spark")


@register
class LTX2EncodeAdapter(GroupAdapter):
    """Text + audio encoder only. Lightweight (~12GB)."""

    model_id = "ltx2-encode"

    def __init__(self):
        self._pipeline = None
        self._device: str = "cuda"
        # Serialises GPU-bound work across parallel infer() calls.
        # max_concurrent=2 lets the arbiter dispatch 2 jobs concurrently;
        # one holds this lock and runs on the GPU while the other does CPU
        # prep (audio decode) and tail save in parallel.
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
            # Pre-load the encoders so subsequent chunks reuse them
            self._pipeline._ensure_encode_models()
            log.info("LTX2-encode: encoders loaded")
        except Exception as e:
            self._pipeline = None
            raise LoadError(f"Failed to load encode models: {e}") from e

    def unload(self) -> None:
        log.info("Unloading LTX2-encode")
        if self._pipeline is not None:
            self._pipeline.unload_encode_models()
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
            raise InferenceError("encode pipeline not loaded")

        self._check_cancel(cancel_flag)

        audio_file = params.get("audio_file")
        if not audio_file or not Path(audio_file).exists():
            raise InferenceError(f"audio_file missing or not found: {audio_file}")

        description = params.get("description", "")
        if not description:
            raise InferenceError("description is required")

        start_time = float(params.get("start_time", 0.0))
        end_time = float(params.get("end_time", 0.0))
        try:
            fps = native_frame_rate(params)
        except ValueError as error:
            raise InferenceError(str(error)) from error
        seed = int(params.get("seed", 42))
        chunk_index = int(params.get("chunk_index", 0))

        try:
            num_frames = lattice_frame_count(params, fps, start_time, end_time)
        except ValueError as error:
            raise InferenceError(str(error)) from error
        start_frame_value = params.get("start_frame")
        start_frame = int(start_frame_value) if start_frame_value is not None else round(start_time * fps)

        chunk = {
            "index": chunk_index,
            "description": description,
            "start_frame": start_frame,
            "num_frames": num_frames,
            "start_image": params.get("start_image_file"),
            "end_image": params.get("end_image_file"),
        }

        output_dir.mkdir(parents=True, exist_ok=True)

        def _progress(stage, status, **kw):
            log.info("encode progress: %s/%s %s", stage, status, kw)
            if cancel_flag.is_set():
                raise CancelledException(f"Cancelled during {stage}/{status}")

        try:
            # PHASE 1 (CPU): decode audio via ffmpeg — can overlap with
            # another infer call's GPU phase.
            prep = self._pipeline.load_encode_input(chunk, audio_file, fps)
            self._check_cancel(cancel_flag)

            # PHASE 2 (GPU): text + audio encoder forward passes. Must be
            # serialised against any other concurrent infer() on this adapter.
            with self._gpu_lock:
                self._check_cancel(cancel_flag)
                result = self._pipeline.run_encode_gpu(
                    prep=prep, fps=fps, seed=seed, progress_fn=_progress,
                )

            self._check_cancel(cancel_flag)

            # PHASE 3 (CPU): torch.save encoded.pt — can overlap with the
            # next infer() acquiring the GPU lock.
            self._pipeline.save_encode_output(result, str(output_dir))
        except CancelledException:
            raise
        except Exception as e:
            raise InferenceError(f"run_encode failed: {e}") from e

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return {
            "file": "encoded.pt",
            "format": "pt",
            "chunk_index": chunk_index,
            "num_frames": num_frames,
        }

    def estimate_time(self, params: dict) -> float:
        return 15_000.0
