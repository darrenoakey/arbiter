"""MiniMax H3 local video generation adapter — GroupAdapter on the GB10 GPU.

Generates short video clips (5–15s) with MiniMax H3 through the diffusers
modular pipeline, running natively on the GB10's unified 128 GB pool.

Why not the Unsloth FP8 checkpoint: that export is a Comfy-Org *pruned*
transformer whose AdaLN modulation is a precomputed curve table
(`adaln_proj.linear` of shape [96768, 8] fed by `time_embedder.table`
[1025, 8]). diffusers' `MiniMaxH3Transformer3DModel` implements the full
modulation projection ([96768, 2688]), so the FP8 state dict cannot be loaded
into it — 51 tensors mismatch on shape. This adapter therefore loads the
official bfloat16 checkpoint and quantizes at load time.

Why NVFP4 rather than int8: int8 brought the two large components down to
~80 GB of resident weights, which only ever completed a frame on an otherwise
empty box. With moondream/insightface/aesthetic-scorer resident (~27 GB), every
inference start was force-killed at MemAvailable 1.7–1.9 GB. NVFP4
(Blackwell-native FP4 weight-only via torchao) targets ~35 GB of weights so H3
can co-reside inside the armed 90 GB budget with real activation headroom. The
same official checkpoint is used; only the torchao config changes.


Expected params dict:
    prompt              : str  — text conditioning
    first_image_b64     : str  — optional, first-frame keyframe (base64 JPEG/PNG)
    last_image_b64      : str  — optional, last-frame keyframe (base64 JPEG/PNG)
    first_image_file    : str  — optional, first-frame file path (staged)
    last_image_file     : str  — optional, last-frame file path (staged)
    duration            : int  — clip length in seconds (5–15, default 6)
    width               : int  — multiple of 32 (default 960)
    height              : int  — multiple of 32 (default 544)
    seed                : int  — RNG seed (default 42)
    num_inference_steps : int  — denoising steps (default 8, CFG-distilled)
"""

from __future__ import annotations

import base64
import gc
import io
import logging
import math
import subprocess
import threading
from pathlib import Path

from arbiter.adapters.base import (
    GroupAdapter,
    InferenceError,
    LoadError,
)
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

# H3 model constants. The frame rate, the duration window and the canvas
# lattice are model properties, not tunables: the video VAE only decodes
# 17*n + 5 frames and the packed sequence is built on a 32px grid.
H3_FPS = 24
H3_MIN_SECONDS = 5
H3_MAX_SECONDS = 15
H3_DEFAULT_STEPS = 8
H3_CANVAS_MULTIPLE = 32
H3_MAX_CANVAS_PIXELS = 1032192  # 1344x768; the pipeline's canvas_max_pixels

H3_BASE_REPO = "MiniMaxAI/MiniMax-H3"
# "fl2va" loads only the first/last-keyframe transformer partition. Without a
# workflow the pipeline pulls BOTH partitions (the omni-reference one too),
# which does not fit. The blocks still auto-route to text-only generation when
# no keyframe is supplied, so this workflow covers every request this adapter
# accepts.
H3_WORKFLOW = "fl2va"

# Modules torchao must leave at bfloat16. These are the small projections and
# embedders either side of the block stack; quantizing them costs almost no
# memory and measurably degrades output. Lifted from the diffusers recipe.
H3_TRANSFORMER_FP_MODULES = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
H3_TEXT_ENCODER_FP_MODULES = [
    "model.visual",
    "model.language_model.embed_tokens",
    "model.language_model.norm",
    "lm_head",
]


def snap_frames(target_frames: int) -> int:
    """Snap a frame count up to the next 17*n + 5 the H3 video VAE can decode."""
    n = max(1, math.ceil((target_frames - 5) / 17))
    return 17 * n + 5


def snap_canvas(width: int, height: int) -> tuple[int, int]:
    """Snap a requested canvas onto the H3 lattice: both sides a multiple of 32,
    total pixels within canvas_max_pixels, aspect ratio preserved as closely as
    the lattice allows. A caller asking for 1920x1080 gets the largest legal H3
    canvas of that shape; conforming back up to the project size is the
    renderer's job, not the model's."""
    width = max(H3_CANVAS_MULTIPLE, int(width))
    height = max(H3_CANVAS_MULTIPLE, int(height))
    scale = math.sqrt(H3_MAX_CANVAS_PIXELS / float(width * height))
    if scale < 1.0:
        width = int(width * scale)
        height = int(height * scale)
    width = max(H3_CANVAS_MULTIPLE, (width // H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE)
    height = max(H3_CANVAS_MULTIPLE, (height // H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE)
    while width * height > H3_MAX_CANVAS_PIXELS:
        if width >= height:
            width -= H3_CANVAS_MULTIPLE
        else:
            height -= H3_CANVAS_MULTIPLE
    return width, height


def _nvfp4_config(modules_to_not_convert: list[str], transformers_flavour: bool = False):
    """Build a torchao NVFP4 weight-only quantization config.

    NVFP4WeightOnlyConfig lives in torchao's prototype mx_formats workflow and
    requires last-2 weight dims divisible by 16 (H3's 2688/5376/96768 all are).
    `modules_to_not_convert` is enforced by the HuggingFace/diffusers
    TorchAoConfig wrapper, not by the torchao config itself — same seam the
    previous int8 path used.
    """
    from torchao.prototype.mx_formats.inference_workflow import NVFP4WeightOnlyConfig

    if transformers_flavour:
        from transformers import TorchAoConfig as Config
    else:
        from diffusers import TorchAoConfig as Config
    return Config(
        NVFP4WeightOnlyConfig(use_dynamic_per_tensor_scale=True),
        modules_to_not_convert=list(modules_to_not_convert),
    )



@register
class MinimaxH3Adapter(GroupAdapter):
    """MiniMax H3 video generation with an NVFP4-quantized denoiser on GB10."""

    model_id = "minimax-h3-local"

    def __init__(self):
        self._pipe = None
        self._device = "cuda"

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def load(self, device: str = "cuda") -> None:
        """Load the fl2va component set with both large components at NVFP4."""
        import torch
        from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
        from transformers import Qwen3VLForConditionalGeneration

        self._device = device
        try:
            self._pipe = ModularPipeline.from_pretrained(H3_BASE_REPO)
            log.info("H3: quantizing transformer and conditioner to NVFP4")
            self._pipe.update_components(
                # The published recipe passes low_cpu_mem_usage=False; the
                # installed diffusers rejects that outright when a
                # quantization_config is present, so the default meta-device
                # path (which quantizes shard by shard) is used instead.
                transformer=MiniMaxH3Transformer3DModel.from_pretrained(
                    H3_BASE_REPO,
                    subfolder="transformer",
                    dtype=torch.bfloat16,
                    quantization_config=_nvfp4_config(H3_TRANSFORMER_FP_MODULES),
                ),
                text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
                    H3_BASE_REPO,
                    subfolder="text_encoder",
                    dtype=torch.bfloat16,
                    quantization_config=_nvfp4_config(
                        H3_TEXT_ENCODER_FP_MODULES, transformers_flavour=True
                    ),
                ),
            )
            # fl2va is the first/last-keyframe partition; loading it by name
            # keeps the omni-reference transformer partition off the box.
            self._pipe.load_components(
                workflow=H3_WORKFLOW, dtype=torch.bfloat16
            )
            # Freezing removes the one autograd path quantized tensors cannot
            # serve. Keep both large components resident on device: NVFP4 is
            # small enough that group-offload's pin/stream path is unnecessary
            # and NVFP4Tensor pin behaviour is still prototype-grade.
            self._pipe.transformer.requires_grad_(False)
            self._pipe.text_encoder.requires_grad_(False)
            self._pipe.transformer.to(device)
            self._pipe.text_encoder.to(device)
            self._pipe.vae.to(device)
            self._pipe.audio_vae.to(device)
            log.info("MiniMax H3 pipeline loaded (NVFP4, on-device)")
        except LoadError:
            self.unload()
            raise
        except Exception as e:
            self.unload()
            raise LoadError(f"Failed to load H3 pipeline: {e}") from e

    def unload(self) -> None:
        """Release the pipeline and its share of the unified memory pool."""
        log.info("Unloading MiniMax H3 adapter")
        self._pipe = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        """Generate a single video clip with optional first/last keyframes."""
        import torch

        if self._pipe is None:
            raise InferenceError("H3 pipeline not loaded — call load() first")

        self._check_cancel(cancel_flag)

        prompt = str(params.get("prompt", "Cinematic music video shot."))[:7000]
        seed = int(params.get("seed", 42))
        duration = int(params.get("duration", 6))
        duration = max(H3_MIN_SECONDS, min(H3_MAX_SECONDS, duration))
        steps = int(params.get("num_inference_steps", H3_DEFAULT_STEPS))
        num_frames = snap_frames(duration * H3_FPS)
        width, height = snap_canvas(
            params.get("width", 960), params.get("height", 544)
        )

        first_image = self._decode_keyframe(
            params, "first_image_b64", "first_image_file"
        )
        last_image = self._decode_keyframe(
            params, "last_image_b64", "last_image_file"
        )

        self._check_cancel(cancel_flag)

        log.info(
            "H3 generating %d frames (%dx%d), %ds, %d steps, seed=%d, first=%s last=%s",
            num_frames, width, height, duration, steps, seed,
            first_image is not None, last_image is not None,
        )

        kwargs = dict(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            num_inference_steps=steps,
        )
        if first_image is not None:
            kwargs["image"] = first_image
        if last_image is not None:
            kwargs["last_image"] = last_image

        try:
            # H3 emits video and a generated soundtrack; asking only for
            # "videos" keeps the audio tensor out of the returned state, since
            # the music-video pipeline muxes the authoritative song itself.
            videos = self._pipe(output="videos", **kwargs)
        except Exception as e:
            raise InferenceError(f"H3 generation failed: {e}") from e

        self._check_cancel(cancel_flag)

        frames = self._extract_video_frames(videos)
        result_path = output_dir / "result.mp4"
        w, h = self._encode_mp4(frames, str(result_path), H3_FPS)

        log.info("H3 clip done: %d frames, %dx%d, %s", len(frames), w, h, result_path)

        return {
            "format": "mp4",
            "file": "result.mp4",
            "width": w,
            "height": h,
            "fps": H3_FPS,
            "duration_seconds": round(len(frames) / H3_FPS, 2),
            "total_frames": len(frames),
        }

    def estimate_time(self, params: dict) -> float:
        """Estimate wall time in ms. GB10 does roughly 1.5–2 s/frame at FP8."""
        duration = int(params.get("duration", 6))
        duration = max(H3_MIN_SECONDS, min(H3_MAX_SECONDS, duration))
        return snap_frames(duration * H3_FPS) * 2000.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_keyframe(params: dict, b64_key: str, file_key: str):
        """Decode a keyframe from a staged file path or inline base64."""
        from PIL import Image

        # Staged file first: it avoids base64 inflation for large keyframes.
        file_path = params.get(file_key)
        if file_path and Path(file_path).is_file():
            return Image.open(file_path).convert("RGB")

        b64_data = params.get(b64_key) or ""
        if not b64_data:
            return None
        if b64_data.startswith("data:"):
            _, b64_data = b64_data.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")

    @staticmethod
    def _extract_video_frames(videos) -> list:
        """Normalise the pipeline's `videos` output to a list of RGB frames.

        The blocks return one entry per request; a batch of one therefore
        arrives as a single-element list whose entry is itself the frame
        sequence (PIL images at the default output_type).
        """
        import numpy as np

        if videos is None:
            raise InferenceError("H3 pipeline returned no videos output")
        frames = videos
        if hasattr(frames, "cpu"):
            frames = frames.cpu().numpy()
        if isinstance(frames, np.ndarray):
            while frames.ndim == 5:
                frames = frames[0]
            return [frames[i] for i in range(frames.shape[0])]
        if isinstance(frames, (list, tuple)):
            if len(frames) == 0:
                raise InferenceError("H3 pipeline returned an empty videos list")
            first = frames[0]
            # A batch wrapper: one video, itself a sequence of frames.
            if isinstance(first, (list, tuple)) or (
                hasattr(first, "ndim") and getattr(first, "ndim", 0) >= 4
            ):
                return MinimaxH3Adapter._extract_video_frames(first)
            return list(frames)
        raise InferenceError(
            f"Could not extract video frames from H3 output of type {type(videos)}"
        )

    @staticmethod
    def _encode_mp4(frames: list, dest: str, fps: int) -> tuple[int, int]:
        """Encode RGB frames to MP4 via ffmpeg pipe. Returns (width, height)."""
        import numpy as np

        if not frames:
            raise InferenceError("No frames to encode")

        sample = np.asarray(frames[0])
        h, w = sample.shape[:2]

        proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-an",  # no audio — the renderer muxes the original song
                dest,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # stderr is drained on a thread: a 124-frame clip overflows the pipe
        # buffer long before the writes finish, and ffmpeg would then block on
        # its own stderr while this side blocks on stdin. communicate() cannot
        # be used after writing, because it re-flushes the closed stdin.
        stderr_tail = bytearray()

        def drain_stderr() -> None:
            while chunk := proc.stderr.read(8192):
                stderr_tail.extend(chunk)
                if len(stderr_tail) > 65536:
                    del stderr_tail[:-65536]

        drain = threading.Thread(target=drain_stderr)
        drain.start()
        write_error = None
        try:
            for frame in frames:
                proc.stdin.write(np.asarray(frame).astype("uint8").tobytes())
        except BrokenPipeError as e:
            write_error = e
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError as e:
                write_error = write_error or e
            proc.wait()
            drain.join()
        if proc.returncode != 0 or write_error is not None:
            raise InferenceError(
                f"ffmpeg encode failed (rc={proc.returncode}): "
                f"{stderr_tail.decode(errors='replace')[-500:]}"
            )
        return w, h
