"""LTX-2.5 encode adapter — Stage A of the 2-way LTX 2.5 split pipeline.

LTX 2.5 is a single 22B dual-stream audio-video DiT (`AVTransformer3DModel`)
with a Gemma 4 12B text/audio encoder, not the dual-checkpoint (dev + distilled)
architecture LTX 2.3 uses. That collapses the 2.3 lane's 3-way split
(encode -> denoise1 -> denoise2) into a 2-way split for 2.5
(encode -> denoise1). There is intentionally NO `ltx25-denoise2` — see
`ltx25_denoise1.py`'s module docstring and LTX_CUSTOMIZATIONS.md's
"LTX 2.5 (2-way split)" section for the full rationale.

Runs the Gemma 4 12B text encoder (+ connectors) and the Audio/Video VAE
encoders, driven by the dedicated `ltx25-spark` runner tree
(`~/src/ltx25-spark`, its own isolated venv — NOT the 2.3 `ltx2-spark` tree
or the arbiter main venv; this adapter runs under `worker_cmd` pointed at
`venvs/ltx25/bin/python`). Produces the `encoded.pt` artifact defined by the
`ltx25-spark` README ("Stage A: ltx25-encode" contract), consumed by
`ltx25-denoise1`.

Expected params dict (README Stage A "Input parameters"):
    prompt / description : str   — text prompt (either key accepted)
    negative_prompt       : str  — optional, defaults to the runner's
                                    DEFAULT_NEGATIVE_PROMPT
    audio_file            : str  — absolute path to audio (wav/mp3) on spark
                                    local disk — REQUIRED (2.5 is audio-
                                    conditioned; see AudioConditioner)
    audio_start_time      : float — seconds into audio_file to slice from
    audio_duration        : float — seconds of audio to slice
    image_file             : str  — optional frame-0 reference image path
    num_frames             : int  — REQUIRED, must satisfy num_frames % 8 == 1
                                     (e.g. 121 == 4.84s @ 25fps); no silent
                                     default — duration is caller-authoritative
    height / width          : int — optional, default 1088x1920 (both must be
                                     divisible by 64 for 2-stage latent upscale)
    fps                     : float — optional, default 25.0
    seed                    : int   — optional, default 42
    chunk_index             : int   — optional, default 0

Output: `encoded.pt` written directly into `output_dir` (the file, not the
directory, per `FastPipeline.save_encode_output`'s exact-path contract).
"""

from __future__ import annotations

import gc
import importlib
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

log = logging.getLogger(__name__)

# The dedicated LTX 2.5 runner tree — deliberately NOT ltx2-spark (the 2.3
# lane's runner). See ltx25-spark/README.md section 2 ("PYTHONPATH & Arbiter
# Worker Rule").
LTX25_SPARK_DIR = Path("/home/darren/src/ltx25-spark")


def _validate_frame_count(num_frames_raw) -> int:
    if num_frames_raw is None:
        raise InferenceError(
            "num_frames is required (must satisfy num_frames % 8 == 1, "
            "e.g. 121 for 4.84s @ 25fps) — duration is caller-authoritative, "
            "not defaulted"
        )
    num_frames = int(num_frames_raw)
    if num_frames < 1 or (num_frames - 1) % 8 != 0:
        raise InferenceError(f"num_frames must satisfy 8n+1, got {num_frames}")
    return num_frames


@register
class LTX25EncodeAdapter(GroupAdapter):
    """Gemma 4 12B text encoder + Audio VAE + Video VAE encoders. ~30GB peak."""

    model_id = "ltx25-encode"

    def __init__(self):
        self._pipeline = None
        self._device: str = "cuda"
        # Serialises the GPU-bound forward passes only; audio decode (CPU)
        # and torch.save (CPU) run outside the lock so a second concurrent
        # infer() can overlap its CPU phase with this one's GPU phase.
        self._gpu_lock = threading.Lock()

    def load(self, device: str = "cuda") -> None:
        self._device = device

        spark_str = str(LTX25_SPARK_DIR)
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
            # Pre-load the encoders so subsequent chunks reuse them.
            self._pipeline._ensure_encode_models()
            log.info("LTX25-encode: encoders loaded")
        except Exception as e:
            self._pipeline = None
            raise LoadError(f"Failed to load LTX 2.5 encode models: {e}") from e

    def unload(self) -> None:
        log.info("Unloading LTX25-encode")
        if self._pipeline is not None:
            self._pipeline.unload_encode_models()
            del self._pipeline
            self._pipeline = None
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        if self._pipeline is None:
            raise InferenceError("LTX 2.5 encode pipeline not loaded")

        self._check_cancel(cancel_flag)

        audio_file = params.get("audio_file")
        if not audio_file or not Path(audio_file).exists():
            raise InferenceError(f"audio_file missing or not found: {audio_file}")

        prompt = params.get("prompt") or params.get("description") or ""
        if not prompt:
            raise InferenceError("prompt/description is required")

        num_frames = _validate_frame_count(params.get("num_frames"))
        fps = float(params.get("fps", 25.0))
        seed = int(params.get("seed", 42))
        chunk_index = int(params.get("chunk_index", params.get("index", 0)))

        output_dir.mkdir(parents=True, exist_ok=True)

        def _progress(stage, status, **kw):
            log.info("ltx25 encode progress: %s/%s %s", stage, status, kw)
            if cancel_flag.is_set():
                raise CancelledException(f"Cancelled during {stage}/{status}")

        try:
            # PHASE 1 (CPU): decode audio via ffmpeg — can overlap another
            # infer call's GPU phase. `load_encode_input` reads prompt,
            # negative_prompt, height, width, images/image_file, and
            # chunk_index straight out of `params` itself.
            prep = self._pipeline.load_encode_input(params, audio_path=audio_file, fps=fps)
            self._check_cancel(cancel_flag)

            # PHASE 2 (GPU): Gemma 4 text encoder + Audio VAE forward passes.
            with self._gpu_lock:
                self._check_cancel(cancel_flag)
                result = self._pipeline.run_encode_gpu(
                    prep=prep,
                    fps=fps,
                    seed=seed,
                    progress_fn=_progress,
                )

            self._check_cancel(cancel_flag)

            # PHASE 3 (CPU): torch.save encoded.pt. `save_encode_output`
            # takes the exact destination FILE path (not a directory).
            encoded_path = output_dir / "encoded.pt"
            self._pipeline.save_encode_output(result, str(encoded_path))
        except CancelledException:
            raise
        except Exception as e:
            raise InferenceError(f"ltx25 run_encode failed: {e}") from e

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
        return 20_000.0
