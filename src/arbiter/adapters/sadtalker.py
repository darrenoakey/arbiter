"""SadTalker talking-head video generation adapter (in-process).

Loads CropAndExtract, Audio2Coeff, and AnimateFromCoeff_PIRender once in
load() and keeps them on GPU. Inference runs the full pipeline in-process
without spawning subprocesses. This allows the arbiter to correctly track
GPU memory usage and enables concurrent weight sharing tests.
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

REQUIRED_CHECKPOINTS = [
    "SadTalker_V0.0.2_256.safetensors",
    "epoch_00190_iteration_000400000_checkpoint.pt",
    "mapping_00229-model.pth.tar",
]


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
    """Add SadTalker repo to sys.path and patch numpy compat."""
    _patch_numpy_compat()
    d = str(SADTALKER_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)


@register
class SadTalkerAdapter(ModelAdapter):
    model_id = "sadtalker"

    def __init__(self):
        self._preprocess_model = None
        self._audio_to_coeff = None
        self._animate_from_coeff = None
        self._device = None
        self._sadtalker_paths = None

    def load(self, device: str = "cuda") -> None:
        """Load all SadTalker sub-models onto GPU."""
        log.info("Loading SadTalker sub-models on %s ...", device)

        for ckpt in REQUIRED_CHECKPOINTS:
            p = SADTALKER_CHECKPOINTS / ckpt
            if not p.exists():
                raise LoadError(f"Missing checkpoint: {p}")

        _add_sadtalker_to_path()

        try:
            from src.utils.init_path import init_path
            from src.utils.preprocess import CropAndExtract
            from src.test_audio2coeff import Audio2Coeff
            from src.facerender.pirender_animate import AnimateFromCoeff_PIRender

            config_dir = str(SADTALKER_DIR / "src" / "config")
            self._sadtalker_paths = init_path(
                str(SADTALKER_CHECKPOINTS), config_dir,
                size=256, old_version=False, preprocess="crop",
            )

            self._preprocess_model = CropAndExtract(self._sadtalker_paths, device)
            self._audio_to_coeff = Audio2Coeff(self._sadtalker_paths, device)
            self._animate_from_coeff = AnimateFromCoeff_PIRender(self._sadtalker_paths, device)
            self._device = device

            log.info("SadTalker ready (%s). 3 sub-models loaded.", device)

        except Exception as exc:
            self._cleanup()
            raise LoadError(f"Failed to load SadTalker: {exc}") from exc

    def unload(self) -> None:
        """Release all SadTalker sub-models and free GPU memory."""
        log.info("Unloading SadTalker.")
        self._cleanup()

    def _cleanup(self):
        for attr in ("_preprocess_model", "_audio_to_coeff", "_animate_from_coeff"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        self._device = None
        self._sadtalker_paths = None
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
        """Generate a talking-head video using loaded SadTalker models.

        params:
            image / image_file: portrait image
            audio / audio_file: driving audio
            size: render size (256 or 512, default 256)
            facerender: renderer (pirender or facevid2vid, default pirender)
            expression_scale: float (default 1.0)
            preprocess: crop, resize, full (default crop)
        """
        if self._preprocess_model is None:
            raise InferenceError("SadTalker not loaded (call load first)")

        self._check_cancel(cancel_flag)

        # Resolve inputs
        try:
            image_bytes = self._resolve_media(params, "image")
        except Exception as e:
            raise InferenceError(f"Failed to resolve image: {e}") from e
        try:
            audio_bytes = self._resolve_media(params, "audio")
        except Exception as e:
            raise InferenceError(f"Failed to resolve audio: {e}") from e

        size = int(params.get("size", 256))
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

            # Step 1: 3DMM extraction
            log.info("SadTalker: 3DMM extraction (size=%d)", size)
            first_frame_dir = os.path.join(save_dir, "first_frame_dir")
            os.makedirs(first_frame_dir, exist_ok=True)

            first_coeff_path, crop_pic_path, crop_info = self._preprocess_model.generate(
                tmp_image, first_frame_dir, preprocess,
                source_image_flag=True, pic_size=size,
            )
            if first_coeff_path is None:
                raise InferenceError("3DMM extraction failed — can't get coefficients from input image")

            self._check_cancel(cancel_flag)

            # Step 2: Audio → coefficients
            log.info("SadTalker: audio2coeff")
            batch = get_data(
                first_coeff_path, tmp_audio, self._device,
                ref_eyeblink_coeff_path=None, still=still,
            )
            coeff_path = self._audio_to_coeff.generate(
                batch, save_dir, pose_style=0, ref_pose_coeff_path=None,
            )

            self._check_cancel(cancel_flag)

            # Step 3: Coefficients → video
            log.info("SadTalker: face rendering (size=%d, batch=%d)", size, batch_size)
            data = get_facerender_data(
                coeff_path, crop_pic_path, first_coeff_path, tmp_audio,
                batch_size, expression_scale=expression_scale,
                still_mode=still, preprocess=preprocess,
                size=size, facemodel="pirender",
            )
            result_path = self._animate_from_coeff.generate(
                data, save_dir, tmp_image, crop_info,
                enhancer=None, background_enhancer=None,
                preprocess=preprocess, img_size=size,
            )

            self._check_cancel(cancel_flag)

            if not result_path or not os.path.isfile(result_path):
                # Fallback: search for mp4 in save_dir
                mp4s = list(Path(save_dir).rglob("*.mp4"))
                if mp4s:
                    result_path = str(mp4s[0])
                else:
                    raise InferenceError("SadTalker produced no output video")

            # Copy to arbiter output dir
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "result.mp4"
            shutil.copy2(result_path, str(output_path))

            width, height = self._probe_video_dimensions(str(output_path))

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
        """Estimate inference time in milliseconds."""
        audio_file = params.get("audio_file", "")
        if audio_file and Path(audio_file).is_file():
            try:
                sz = Path(audio_file).stat().st_size
                duration = sz / 48000
                return duration * 800 + 3000  # ~0.8s per second of audio + 3s overhead
            except Exception:
                pass
        return 10000

    @staticmethod
    def _probe_video_dimensions(path: str) -> tuple[int, int]:
        """Use ffprobe to get video width and height."""
        import subprocess
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=p=0:s=x",
                    path,
                ],
                capture_output=True, text=True, timeout=10,
            )
            parts = result.stdout.strip().split("x")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
        return 256, 256
