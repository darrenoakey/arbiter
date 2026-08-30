"""MiniMax FastH3 local video generation adapter — GroupAdapter on the GB10 GPU.

FastH3 is FastVideo's 4-step DMD2 distillation of MiniMax-H3. Same fl2va
modular pipeline, VAE, tokenizer, and Qwen3-VL text encoder as
minimax-h3-local; only the denoiser transformer is the distilled student.

The published FastH3 card is T2VA-only; FL2VA is not distilled. This adapter
still requires both first and last keyframes and runs the shared H3 fl2va
workflow — music-video clips are unusable without that conditioning. Audio-in
(driving a clip from an existing soundtrack) is not supported; FastH3 emits
its own audio and the music-video lane muxes the authoritative Suno track.

The published FastH3 card is explicit that `num_inference_steps=4` is the
wrong call: MiniMaxH3SetTimestepsStep then builds a native 4-point sigma
grid and runs 3 forwards, not the 4 trained jump points. This adapter
therefore always asks for 4 steps and monkey-patches both schedulers to the
trained ladder `[0.999, 0.749, 0.500, 0.250, 0.0]` on the shared 1000-step
clock.

The NVFP4 text-encoder snapshot is reused from minimax-h3-local
(`/mnt/t9/models/h3-nvfp4-fl2va/text_encoder`). The FastH3 transformer is
quantized once and snapshotted under `/mnt/t9/models/fasth3-nvfp4-fl2va`.

Expected params dict matches VideoGenerateH3Params. `num_inference_steps` is
accepted then ignored — FastH3 is 4-step only. Missing first or last
keyframe is an InferenceError.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from arbiter.adapters.base import InferenceError, LoadError
from arbiter.adapters.minimax_h3_local import (
    H3_BASE_REPO,
    H3_FPS,
    H3_MAX_SECONDS,
    H3_MIN_SECONDS,
    H3_SNAPSHOT_DIR,
    H3_TEXT_ENCODER_FP_MODULES,
    H3_TRANSFORMER_FP_MODULES,
    H3_WORKFLOW,
    MinimaxH3LocalAdapter,
    _nvfp4_config,
    snap_frames,
)
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

FASTH3_REPO = "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2"
FASTH3_SNAPSHOT_DIR = Path("/mnt/t9/models/fasth3-nvfp4-fl2va")
FASTH3_STEPS = 4
FASTH3_DMD_STEPS = (999, 749, 500, 250)


def fasth3_video_sigmas() -> list[float]:
    """Trained FastH3 video ladder on the shared 1000-step clock, plus terminal 0."""
    return [step / 1000.0 for step in FASTH3_DMD_STEPS] + [0.0]


def fasth3_audio_sigmas() -> list[float]:
    """Audio uses the same 4 trained jump points as the video student grid."""
    return fasth3_video_sigmas()


def fasth3_steps(_params: dict | None = None) -> int:
    """FastH3 is 4-step only; caller overrides are ignored."""
    return FASTH3_STEPS


def _fasth3_snapshot_ready() -> bool:
    """True when the FastH3 transformer snapshot and shared H3 text encoder exist."""
    return (FASTH3_SNAPSHOT_DIR / "transformer").is_dir() and (
        H3_SNAPSHOT_DIR / "text_encoder"
    ).is_dir()


def _install_trained_sigma_ladder(pipe) -> None:
    """Force both MiniMax schedulers onto the trained 4-jump sigma ladder.

    MiniMaxH3SetTimestepsStep always calls set_timesteps(num_inference_steps=N)
    and never accepts explicit sigmas. Wrapping the bound method is the
    adapter-local seam that keeps the 4 trained forwards without forking
    the upstream block.
    """
    import torch

    video = torch.tensor(fasth3_video_sigmas(), dtype=torch.float32)
    audio = torch.tensor(fasth3_audio_sigmas(), dtype=torch.float32)
    orig_video = pipe.scheduler.set_timesteps
    orig_audio = pipe.audio_scheduler.set_timesteps

    def video_set_timesteps(num_inference_steps=None, device=None, sigmas=None, **kwargs):
        orig_video(device=device, sigmas=video)

    def audio_set_timesteps(num_inference_steps=None, device=None, sigmas=None, **kwargs):
        orig_audio(device=device, sigmas=audio)

    pipe.scheduler.set_timesteps = video_set_timesteps
    pipe.audio_scheduler.set_timesteps = audio_set_timesteps


def require_first_and_last_keyframes(params: dict) -> None:
    """Fail closed unless both first and last keyframes are present.

    FastH3 Preview is distilled for T2VA only. The music-video lane still
    cannot use a clip without first/last conditioning, so this adapter
    refuses to generate rather than silently dropping to text-only.
    """
    first = params.get("first_image_file") or params.get("first_image_b64")
    last = params.get("last_image_file") or params.get("last_image_b64")
    if not first or not last:
        raise InferenceError(
            "FastH3 requires both first and last keyframes "
            "(first_image_file/first_image_b64 and last_image_file/last_image_b64)"
        )


@register
class MinimaxFastH3Adapter(MinimaxH3LocalAdapter):
    """FastH3 4-step distilled MiniMax-H3 denoiser on GB10."""

    model_id = "minimax-fast-h3"

    def load(self, device: str = "cuda") -> None:
        """Load fl2va with the FastH3 transformer and the shared NVFP4 text encoder."""
        import torch
        from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
        from transformers import Qwen3VLForConditionalGeneration

        self._device = device
        try:
            self._pipe = ModularPipeline.from_pretrained(H3_BASE_REPO)
            transformer_ready = (FASTH3_SNAPSHOT_DIR / "transformer").is_dir()
            text_encoder_ready = (H3_SNAPSHOT_DIR / "text_encoder").is_dir()
            if transformer_ready:
                log.info("FastH3: loading NVFP4 transformer from %s", FASTH3_SNAPSHOT_DIR)
                transformer = MiniMaxH3Transformer3DModel.from_pretrained(
                    FASTH3_SNAPSHOT_DIR / "transformer",
                    dtype=torch.bfloat16,
                )
            else:
                log.info("FastH3: quantizing transformer from %s to NVFP4", FASTH3_REPO)
                transformer = MiniMaxH3Transformer3DModel.from_pretrained(
                    FASTH3_REPO,
                    subfolder="transformer",
                    dtype=torch.bfloat16,
                    quantization_config=_nvfp4_config(H3_TRANSFORMER_FP_MODULES),
                )
            if text_encoder_ready:
                log.info("FastH3: reusing H3 NVFP4 text encoder from %s", H3_SNAPSHOT_DIR)
                text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                    H3_SNAPSHOT_DIR / "text_encoder",
                    dtype=torch.bfloat16,
                )
            else:
                log.info("FastH3: quantizing H3 text encoder to NVFP4")
                text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                    H3_BASE_REPO,
                    subfolder="text_encoder",
                    dtype=torch.bfloat16,
                    quantization_config=_nvfp4_config(
                        H3_TEXT_ENCODER_FP_MODULES, transformers_flavour=True
                    ),
                )
            self._write_nvfp4_snapshot(transformer, text_encoder)
            self._pipe.update_components(
                transformer=transformer,
                text_encoder=text_encoder,
            )
            self._pipe.load_components(workflow=H3_WORKFLOW, dtype=torch.bfloat16)
            self._pipe.transformer.requires_grad_(False)
            self._pipe.text_encoder.requires_grad_(False)
            self._pipe.transformer.to(device)
            self._pipe.text_encoder.to(device)
            self._pipe.vae.to(device)
            self._pipe.audio_vae.to(device)
            _install_trained_sigma_ladder(self._pipe)
            torch.cuda.empty_cache()
            self._log_memory(torch, "post-load")
            log.info("MiniMax FastH3 pipeline loaded (NVFP4, 4-step ladder, on-device)")
        except LoadError:
            self.unload()
            raise
        except Exception as e:
            self.unload()
            raise LoadError(f"Failed to load FastH3 pipeline: {e}") from e

    def _write_nvfp4_snapshot(self, transformer, text_encoder) -> None:
        """Persist the FastH3 transformer snapshot. Text encoder stays on the H3 path."""
        try:
            FASTH3_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            dest = FASTH3_SNAPSHOT_DIR / "transformer"
            if dest.exists():
                return
            tmp = dest.with_name(dest.name + ".tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            log.info("FastH3: writing NVFP4 transformer snapshot")
            transformer.save_pretrained(tmp)
            tmp.rename(dest)
            log.info("FastH3: NVFP4 transformer snapshot complete at %s", dest)
        except Exception:
            log.exception(
                "FastH3: NVFP4 snapshot write failed; next load will retry "
                "on-the-fly quantization"
            )

    def infer(self, params: dict, output_dir: Path, cancel_flag) -> dict:
        """Generate one clip, always on the trained 4-step ladder."""
        require_first_and_last_keyframes(params)
        forced = dict(params)
        forced["num_inference_steps"] = fasth3_steps(params)
        return super().infer(forced, output_dir, cancel_flag)

    def estimate_time(self, params: dict) -> float:
        """4-step distilled denoise is ~2x the local 8-step H3 estimate."""
        duration = int(params.get("duration", 6))
        duration = max(H3_MIN_SECONDS, min(H3_MAX_SECONDS, duration))
        return snap_frames(duration * H3_FPS) * 1000.0
