"""LTX-2 denoise stage 1 adapter — middle stage of the 3-stage video pipeline.

Loads video encoder + upsampler + stage-1 transformer (~42GB). Reads an
`encoded.pt` from ltx2-encode, computes image conditionings, runs the stage-1
denoise loop, upsamples, and saves `stage1_output.pt` for ltx2-denoise2.

The output file uses the same schema as the legacy FastPipeline.run_stage1
output so existing stage1_output.pt files (from prior runs) can be fed directly
to ltx2-denoise2.

Expected params dict:
    encoded_file  : str — absolute path to encoded.pt on spark local disk
    width         : int
    height        : int
"""

from __future__ import annotations

import gc
import logging
import importlib
import sys
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

log = logging.getLogger(__name__)

LTX2_SPARK_DIR = Path("/home/darren/src/ltx2-spark")


@register
class LTX2Denoise1Adapter(GroupAdapter):
    """Video encoder + upsampler + stage-1 transformer. ~42GB."""

    model_id = "ltx2-denoise1"

    def __init__(self):
        self._pipeline = None
        self._device: str = "cuda"
        # Serialises the GPU phase only. With max_concurrent=2, the second
        # infer() can do its torch.load encoded.pt from NFS while the first
        # is holding the lock and running the denoise loop.
        self._gpu_lock = threading.Lock()

    def load(self, device: str = "cuda") -> None:
        self._device = device

        spark_str = str(LTX2_SPARK_DIR)
        if spark_str not in sys.path:
            sys.path.insert(0, spark_str)

        try:
            importlib.import_module("ltx_core")
            importlib.import_module("ltx_pipelines")
        except ImportError as e:
            raise LoadError(f"ltx_core / ltx_pipelines not importable: {e}")

        try:
            FastPipeline = importlib.import_module("video_fast_gpu").FastPipeline

            self._pipeline = FastPipeline()
            # Safetensors staging clones are short lived, but glibc otherwise
            # retains their heap pages and creates a second checkpoint-sized
            # allocation in GB10 unified memory during model load.
            with HeapTrimGuard():
                self._pipeline._ensure_denoise1_models()
            log.info("LTX2-denoise1: models loaded")
        except Exception as e:
            self._pipeline = None
            raise LoadError(f"Failed to load denoise1 models: {e}") from e

    def unload(self) -> None:
        log.info("Unloading LTX2-denoise1")
        if self._pipeline is not None:
            self._pipeline.unload_denoise1_models()
            if hasattr(self._pipeline, "components"):
                del self._pipeline.components
            if hasattr(self._pipeline, "stage_1_ledger"):
                del self._pipeline.stage_1_ledger
            if hasattr(self._pipeline, "stage_2_ledger"):
                del self._pipeline.stage_2_ledger
            del self._pipeline
            self._pipeline = None
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        if self._pipeline is None:
            raise InferenceError("denoise1 pipeline not loaded")

        self._check_cancel(cancel_flag)

        encoded_file = params.get("encoded_file")
        if not encoded_file:
            raise InferenceError("encoded_file is required")
        if not Path(encoded_file).exists():
            raise InferenceError(f"encoded_file does not exist: {encoded_file}")

        width = int(params.get("width", 768))
        height = int(params.get("height", 512))

        output_dir.mkdir(parents=True, exist_ok=True)

        def _progress(stage, status, **kw):
            log.info("denoise1 progress: %s/%s %s", stage, status, kw)
            if cancel_flag.is_set():
                raise CancelledException(f"Cancelled during {stage}/{status}")

        try:
            # PHASE 1 (CPU/IO): load encoded.pt from disk. No GPU work.
            data = self._pipeline.load_denoise1_input(encoded_file)
            self._check_cancel(cancel_flag)

            # PHASE 2 (GPU): image conditioning + denoise loop + upsample.
            # Serialised against any other concurrent infer() on this adapter.
            with self._gpu_lock:
                self._check_cancel(cancel_flag)
                result = self._pipeline.run_denoise1_gpu(
                    data,
                    width,
                    height,
                    progress_fn=_progress,
                )

            self._check_cancel(cancel_flag)

            # PHASE 3 (CPU/IO): torch.save stage1_output.pt to disk.
            self._pipeline.save_denoise1_output(result, str(output_dir))
        except CancelledException:
            raise
        except Exception as e:
            raise InferenceError(f"run_denoise1 failed: {e}") from e

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return {
            "file": "stage1_output.pt",
            "format": "pt",
        }

    def estimate_time(self, params: dict) -> float:
        return 180_000.0
