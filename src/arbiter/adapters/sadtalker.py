"""SadTalker talking-head video generation adapter (in-process).

Loads sub-models for both 256px (pirender) and 512px (facevid2vid) and keeps
them on GPU. Inference runs the full pipeline without spawning subprocesses.
"""
from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from time import strftime

from arbiter.adapters.base import ModelAdapter, InferenceError, LoadError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

SADTALKER_DIR = Path("/home/darren/src/talking-head/local-sadtalker")
SADTALKER_CHECKPOINTS = SADTALKER_DIR / "checkpoints"


def _patch_numpy_compat():
    """Monkey-patch numpy 2.x to restore attributes removed in 2.0."""
    import numpy as np
    for name, repl in [("float", float), ("int", int), ("complex", complex),
                        ("object", object), ("bool", bool), ("str", str)]:
        if not hasattr(np, name):
            setattr(np, name, repl)
    if not hasattr(np, "VisibleDeprecationWarning"):
        np.VisibleDeprecationWarning = FutureWarning


def _add_sadtalker_to_path():
    _patch_numpy_compat()
    d = str(SADTALKER_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)


# Size -> renderer mapping. PIRender only works at 256; facevid2vid for 512.
SIZE_RENDERER = {
    256: "pirender",
    512: "facevid2vid",
}


@register
class SadTalkerAdapter(ModelAdapter):
    model_id = "sadtalker"

    def __init__(self):
        self._models = {}  # size -> {preprocess, audio2coeff, animate, paths, renderer}
        self._device = None

    def load(self, device: str = "cuda") -> None:
        """Load SadTalker sub-models for all available sizes."""
        log.info("Loading SadTalker sub-models on %s ...", device)

        _add_sadtalker_to_path()

        from src.utils.init_path import init_path
        from src.utils.preprocess import CropAndExtract
        from src.test_audio2coeff import Audio2Coeff
        from src.facerender.pirender_animate import AnimateFromCoeff_PIRender
        from src.facerender.animate import AnimateFromCoeff as AnimateFromCoeff_FaceVid2Vid

        config_dir = str(SADTALKER_DIR / "src" / "config")

        for size, renderer in SIZE_RENDERER.items():
            ckpt = SADTALKER_CHECKPOINTS / f"SadTalker_V0.0.2_{size}.safetensors"
            if not ckpt.exists():
                log.warning("Checkpoint missing for %dpx: %s — skipping", size, ckpt)
                continue

            try:
                paths = init_path(
                    str(SADTALKER_CHECKPOINTS), config_dir,
                    size=size, old_version=False, preprocess="crop",
                )

                if renderer == "pirender":
                    animate = AnimateFromCoeff_PIRender(paths, device)
                else:
                    animate = AnimateFromCoeff_FaceVid2Vid(paths, device)

                self._models[size] = {
                    "preprocess": CropAndExtract(paths, device),
                    "audio2coeff": Audio2Coeff(paths, device),
                    "animate": animate,
                    "paths": paths,
                    "renderer": renderer,
                }
                log.info("  %dpx (%s) loaded.", size, renderer)
            except Exception as exc:
                log.error("Failed to load %dpx: %s", size, exc)
                self._models.pop(size, None)

        if not self._models:
            self._cleanup()
            raise LoadError("No SadTalker models loaded successfully")

        self._device = device
        log.info("SadTalker ready (%s). Sizes: %s", device, sorted(self._models.keys()))

    def unload(self) -> None:
        log.info("Unloading SadTalker.")
        self._cleanup()

    def _cleanup(self):
        for m in self._models.values():
            for key in ("preprocess", "audio2coeff", "animate"):
                obj = m.get(key)
                if obj is not None:
                    del obj
        self._models.clear()
        self._device = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def infer(
        self,
        params: dict,
        output_dir: Path,
        cancel_flag: threading.Event,
    ) -> dict:
        """Generate a talking-head video.

        params:
            image / image_file: portrait image
            audio / audio_file: driving audio
            size: 256 (pirender, fast) or 512 (facevid2vid, higher res). Default 256.
            expression_scale: float (default 1.0)
            preprocess: crop, resize, full (default crop)
        """
        if not self._models:
            raise InferenceError("SadTalker not loaded")

        self._check_cancel(cancel_flag)

        try:
            image_bytes = self._resolve_media(params, "image")
        except Exception as e:
            raise InferenceError(f"Failed to resolve image: {e}") from e
        try:
            audio_bytes = self._resolve_media(params, "audio")
        except Exception as e:
            raise InferenceError(f"Failed to resolve audio: {e}") from e

        size = int(params.get("size", 256))

        # Allow explicit facerender override, otherwise auto-select
        renderer = params.get("facerender")
        if renderer and renderer not in ("pirender", "facevid2vid"):
            raise InferenceError(f"Unknown facerender: {renderer}")

        if not renderer:
            renderer = SIZE_RENDERER.get(size, "pirender")

        # Find the right model set — match by size
        if size not in self._models:
            available = sorted(self._models.keys())
            raise InferenceError(f"Size {size} not loaded. Available: {available}")

        models = self._models[size]

        # Warn if renderer doesn't match what was loaded for this size
        if renderer != models["renderer"]:
            log.warning(
                "Requested facerender=%s but %dpx loaded with %s. "
                "Using loaded renderer.",
                renderer, size, models["renderer"],
            )
            renderer = models["renderer"]

        expression_scale = float(params.get("expression_scale", 1.0))
        preprocess = params.get("preprocess", "crop")
        still = params.get("still", False)
        batch_size = int(params.get("batch_size", 2))

        tmp_dir = tempfile.mkdtemp(prefix="sadtalker_")
        save_dir = os.path.join(tmp_dir, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)

        tmp_image = os.path.join(tmp_dir, "input.png")
        tmp_audio = os.path.join(tmp_dir, "input.wav")

        try:
            with open(tmp_image, "wb") as f:
                f.write(image_bytes)
            with open(tmp_audio, "wb") as f:
                f.write(audio_bytes)

            self._check_cancel(cancel_flag)

            _add_sadtalker_to_path()
            from src.generate_batch import get_data
            from src.generate_facerender_batch import get_facerender_data

            log.info("SadTalker: 3DMM extraction (size=%d)", size)
            first_frame_dir = os.path.join(save_dir, "first_frame_dir")
            os.makedirs(first_frame_dir, exist_ok=True)

            first_coeff_path, crop_pic_path, crop_info = models["preprocess"].generate(
                tmp_image, first_frame_dir, preprocess,
                source_image_flag=True, pic_size=size,
            )
            if first_coeff_path is None:
                raise InferenceError("3DMM extraction failed")

            self._check_cancel(cancel_flag)

            log.info("SadTalker: audio2coeff")
            batch = get_data(
                first_coeff_path, tmp_audio, self._device,
                ref_eyeblink_coeff_path=None, still=still,
            )
            coeff_path = models["audio2coeff"].generate(
                batch, save_dir, pose_style=0, ref_pose_coeff_path=None,
            )

            self._check_cancel(cancel_flag)

            log.info("SadTalker: rendering %dpx (%s)", size, renderer)
            data = get_facerender_data(
                coeff_path, crop_pic_path, first_coeff_path, tmp_audio,
                batch_size, expression_scale=expression_scale,
                still_mode=still, preprocess=preprocess,
                size=size, facemodel=renderer,
            )
            result_path = models["animate"].generate(
                data, save_dir, tmp_image, crop_info,
                enhancer=None, background_enhancer=None,
                preprocess=preprocess, img_size=size,
            )

            self._check_cancel(cancel_flag)

            if not result_path or not os.path.isfile(result_path):
                mp4s = list(Path(save_dir).rglob("*.mp4"))
                if mp4s:
                    result_path = str(mp4s[0])
                else:
                    raise InferenceError("SadTalker produced no output video")

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "result.mp4"
            shutil.copy2(result_path, str(out_path))

            width, height = self._probe_video_dimensions(str(out_path))

            return {
                "format": "mp4",
                "width": width,
                "height": height,
                "file": "result.mp4",
            }

        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(f"SadTalker inference failed: {e}") from e
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def estimate_time(self, params: dict) -> float:
        audio_file = params.get("audio_file", "")
        if audio_file and Path(audio_file).is_file():
            try:
                sz = Path(audio_file).stat().st_size
                duration = sz / 48000
                return duration * 800 + 3000
            except Exception:
                pass
        return 10000

    @staticmethod
    def _probe_video_dimensions(path: str) -> tuple[int, int]:
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
                capture_output=True, text=True, timeout=10,
            )
            parts = result.stdout.strip().split("x")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
        return 256, 256
