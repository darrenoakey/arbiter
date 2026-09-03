"""LTX-2.5 denoise adapter — Stage B, the ONLY denoise stage of the 2-way
LTX 2.5 split pipeline.

*** There is no `ltx25-denoise2`, and there never will be. This is not a
*** truncated split — it is the correct one. Do NOT add a fake denoise2 stage.
Read this docstring before "completing the pattern" from the 2.3 lane.

Why 2-way, not 3-way like LTX 2.3:
    LTX 2.3 splits encode -> denoise1 -> denoise2 because it has TWO distinct
    transformer checkpoints (a stage-1 "dev" transformer and a stage-2
    distilled transformer) plus a spatial upsampler in between — three
    independently-scheduled weight sets. LTX 2.5 has exactly ONE 22B
    dual-stream DiT (`ltx-2.5-22b-dev-transformer-bf16.safetensors`); "stage 1"
    and "stage 2" are the SAME base transformer run twice — once bare at half
    resolution (544x960), once with the distilled LoRA applied at full
    resolution (1088x1920) after a 2x latent spatial upscale. Splitting that
    into two Arbiter models would force two workers to each hold the full
    ~39GB base transformer resident at once (~78GB just for the duplicated
    base weights) for zero benefit — the entire stage1 -> upscale -> stage2 ->
    decode chain already runs in-memory, back-to-back, inside one GPU-locked
    call. See LTX_CUSTOMIZATIONS.md's "LTX 2.5 (2-way split)" section and
    local/ltx25-stage-map.md (task t3) for the full inference-graph mapping
    and the explicit "why not 3-way" rejection.

    Renderer/orchestrator note (consumed by later tasks wiring the
    music_video_ltx25_full.json engine lane): the 2.3 renderer's
    `video-denoise2` step MUST be SKIPPED for the ltx25 lane — this adapter's
    `result.mp4` output IS the final per-chunk artifact, exactly like
    `ltx2-denoise2`'s output, just produced by one stage instead of two.
    There is no `denoise1_file`/`stage1_output.pt` hand-off to feed a second
    stage, and no `ltx25-denoise2` model exists to receive one.

Loads (resident, pre-loaded in `load()` via the runner's public
`FastPipeline.load_denoise_models()` hook — same "keep the big weights
resident across jobs" pattern as `ltx2_denoise2.py`):
    - 22B Dev Transformer (bf16)                    39.13 GB
    - Distilled LoRA (rank 450, bf16)                 8.29 GB
    - Latent Spatial Upscaler x2                      0.93 GB
    - CausalDiffusionVAE video decoder                1.37 GB
    Resident total: 49.72 GB. Peak during stage-2 1088x1920 refinement: ~80GB.

Driven by the dedicated `ltx25-spark` runner tree (`~/src/ltx25-spark`, its
own isolated venv via `worker_cmd` -> `venvs/ltx25/bin/python`) — deliberately
NOT `ltx2-spark`.

Expected params dict (README "Stage B: ltx25-denoise" contract):
    encoded_file        : str   — absolute path to encoded.pt from ltx25-encode
    audio_file          : str   — absolute path to the ORIGINAL Suno audio
                                   master on spark local disk. This is muxed
                                   into the final mp4 verbatim; any audio LTX
                                   2.5 itself decodes/generates internally is
                                   unconditionally discarded (see
                                   `video_fast_gpu.encode_video_nvenc`).
    start_time          : float — chunk start in seconds (audio mux slice offset)
    fps                 : float — output frame rate, default 25.0
    num_inference_steps : int   — stage-1 diffusion steps, default 30
    a2v_guidance_scale  : float — stage-1 audio-conditioning guidance scale,
                                  default 3.0; must be >= 1.0 else InferenceError.
                                  Forwarded verbatim to FastPipeline.run_denoise_gpu
                                  (ltx25-spark runner; the A/B lever that trades
                                  mouth fidelity against identity drift).

Output: `result.mp4` written directly into `output_dir` (the file, not the
directory, per `FastPipeline.save_denoise_output` / `encode_video_nvenc`'s
exact-path contract) — the FINAL chunk artifact.
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
    HeapTrimGuard,
    InferenceError,
    LoadError,
)
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

# The dedicated LTX 2.5 runner tree — deliberately NOT ltx2-spark (the 2.3
# lane's runner). See ltx25-spark/README.md section 2 ("PYTHONPATH & Arbiter
# Worker Rule").
LTX25_SPARK_DIR = Path("/home/darren/src/ltx25-spark")


@register
class LTX25Denoise1Adapter(GroupAdapter):
    """22B transformer + distilled LoRA + upscaler + VAE decoder, all
    resident. ~49.7GB resident, ~80GB peak during stage-2 refinement.

    Intentionally the ONLY denoise stage for LTX 2.5 — see module docstring.
    """

    model_id = "ltx25-denoise1"

    def __init__(self):
        self._pipeline = None
        self._device: str = "cuda"
        # Serialises the GPU phase only (run_denoise_gpu: stage1 + upscale +
        # stage2 + decode). save_denoise_output's NVENC encode/mux is CPU/
        # ffmpeg work that runs OUTSIDE the lock so it overlaps the next
        # job's GPU phase — same pipelining pattern as ltx2-denoise2.
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
            log.info(
                "LTX25-denoise1: pre-loading 22B transformer + LoRA + "
                "upscaler + VAE decoder (~49.7GB resident)"
            )
            with HeapTrimGuard():
                # Public preload hook — keeps the resident set loaded across
                # every subsequent chunk instead of re-loading it per job.
                self._pipeline.load_denoise_models()
            log.info("LTX25-denoise1: models resident")
        except Exception as e:
            self._pipeline = None
            raise LoadError(f"Failed to load LTX 2.5 denoise models: {e}") from e

    def unload(self) -> None:
        log.info("Unloading LTX25-denoise1")
        if self._pipeline is not None:
            self._pipeline.unload_denoise_models()
            del self._pipeline
            self._pipeline = None
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        # Validate params BEFORE the loaded-pipeline check so a malformed job
        # always reports the real defect, never "not loaded".
        raw_scale = params.get("a2v_guidance_scale", 3.0)
        try:
            a2v_guidance_scale = float(raw_scale)
        except (TypeError, ValueError) as e:
            raise InferenceError(
                f"a2v_guidance_scale must be a number, got {raw_scale!r}"
            ) from e
        if a2v_guidance_scale < 1.0:
            raise InferenceError(
                f"a2v_guidance_scale must be >= 1.0, got {a2v_guidance_scale}"
            )

        if self._pipeline is None:
            raise InferenceError("LTX 2.5 denoise pipeline not loaded")

        self._check_cancel(cancel_flag)

        encoded_file = params.get("encoded_file")
        if not encoded_file:
            raise InferenceError("encoded_file is required")
        if not Path(encoded_file).exists():
            raise InferenceError(f"encoded_file does not exist: {encoded_file}")

        audio_file = params.get("audio_file")
        if not audio_file or not Path(audio_file).exists():
            raise InferenceError(f"audio_file missing or does not exist: {audio_file}")

        start_time = float(params.get("start_time", 0.0))
        fps = float(params.get("fps", 25.0))
        num_inference_steps = int(params.get("num_inference_steps", 30))

        output_dir.mkdir(parents=True, exist_ok=True)

        def _progress(stage, status, **kw):
            log.info("ltx25 denoise progress: %s/%s %s", stage, status, kw)
            if cancel_flag.is_set():
                raise CancelledException(f"Cancelled during {stage}/{status}")

        try:
            # PHASE 1 (CPU/IO): torch.load encoded.pt, map_location="cpu" —
            # overlaps a concurrent call's GPU phase (same asymmetric
            # map_location rationale as the 2.3 lane's load_denoise1_input;
            # see LTX_CUSTOMIZATIONS.md §G).
            data = self._pipeline.load_denoise_input(encoded_file)
            self._check_cancel(cancel_flag)

            # PHASE 2 (GPU, locked): stage-1 diffusion (544x960) -> 2x latent
            # upscale -> stage-2 distilled-LoRA refine (1088x1920) -> tiled
            # VAE decode -> mandatory 1080p center-crop. This IS the "denoise1
            # -> denoise2" chain of the 2.3 lane, run in one call because
            # there is only one transformer to keep resident.
            with self._gpu_lock:
                self._check_cancel(cancel_flag)
                frames_np = self._pipeline.run_denoise_gpu(
                    data,
                    num_inference_steps=num_inference_steps,
                    a2v_guidance_scale=a2v_guidance_scale,
                    progress_fn=_progress,
                )

            self._check_cancel(cancel_flag)

            # PHASE 3 (CPU/NVENC, outside the lock): encode_video_nvenc
            # (inside save_denoise_output) strips any audio LTX 2.5 itself
            # produced and muxes the ORIGINAL Suno audio slice — never
            # generated audio. Overlaps the next job's GPU phase.
            result_path = output_dir / "result.mp4"
            self._pipeline.save_denoise_output(
                frames_np,
                str(result_path),
                fps=fps,
                audio_path=audio_file,
                start_time=start_time,
            )
        except CancelledException:
            raise
        except Exception as e:
            raise InferenceError(f"ltx25 run_denoise failed: {e}") from e

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return {
            "file": "result.mp4",
            "format": "mp4",
        }

    def estimate_time(self, params: dict) -> float:
        # One call covers what the 2.3 lane splits across denoise1 (180s) +
        # denoise2 (120s); size at least as generously as their sum, with
        # margin for the larger 22B (vs split dev/distilled) forward passes.
        return 320_000.0
