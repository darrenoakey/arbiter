"""Face-restore adapter — CodeFormer per-frame video face restoration.

Alternative to the GFPGAN-based `face-restore` adapter. CodeFormer has a
`fidelity_weight` knob (0.0 = max restoration / least faithful, 1.0 = least
restoration / most faithful to original) that lets callers dial down the
effect on stylised/AI-generated content where aggressive restoration tends
to produce uncanny results.

Expected params:
    video_file       : str   — absolute spark path to input mp4 (must exist on disk)
    fidelity_weight  : float — CodeFormer w parameter, 0..1, default 0.9
    only_center_face : bool  — only restore the centre face, default False
    upscale          : int   — output upscale factor, default 1 (no change)

Output dict:
    {"file": "result.mp4", "format": "mp4", "frames": N, "faces_restored": M, ...}
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path

from arbiter.adapters.base import (
    CancelledException,
    InferenceError,
    LoadError,
    ModelAdapter,
)
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

CODEFORMER_DIR = Path("/home/darren/src/CodeFormer")


@register
class FaceRestoreCodeFormerAdapter(ModelAdapter):
    """CodeFormer face-restoration over video frames."""

    model_id = "face-restore-codeformer"

    def __init__(self):
        self._net = None
        self._face_helper = None
        self._device: str = "cuda"
        self._gpu_lock = threading.Lock()

    def load(self, device: str = "cuda") -> None:
        self._device = device

        # CodeFormer is not pip-installed; it's a cloned repo at CODEFORMER_DIR.
        # Put its own vendored `basicsr` + `facelib` on sys.path so imports work.
        cf_str = str(CODEFORMER_DIR)
        if cf_str not in sys.path:
            sys.path.insert(0, cf_str)

        try:
            import torch
            from basicsr.utils.registry import ARCH_REGISTRY
            from facelib.utils.face_restoration_helper import FaceRestoreHelper
            # Force-import the arch module so it registers with ARCH_REGISTRY
            import basicsr.archs.codeformer_arch  # noqa: F401
        except ImportError as e:
            raise LoadError(f"CodeFormer imports failed: {e}")

        try:
            net = ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(device)
            ckpt_path = CODEFORMER_DIR / "weights" / "CodeFormer" / "codeformer.pth"
            if not ckpt_path.exists():
                raise LoadError(
                    f"CodeFormer checkpoint not found at {ckpt_path}. Run "
                    f"`python scripts/download_pretrained_models.py CodeFormer` "
                    f"from {CODEFORMER_DIR}."
                )
            checkpoint = torch.load(str(ckpt_path))["params_ema"]
            net.load_state_dict(checkpoint)
            net.eval()
            self._net = net

            # FaceRestoreHelper for detection/alignment/pasteback
            self._face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                use_parse=True,
                device=device,
            )
            log.info("face-restore-codeformer: model + face helper loaded on %s", device)
        except Exception as e:
            self._net = None
            self._face_helper = None
            raise LoadError(f"CodeFormer init failed: {e}") from e

    def unload(self) -> None:
        log.info("Unloading face-restore-codeformer")
        if self._net is not None:
            del self._net
            self._net = None
        if self._face_helper is not None:
            del self._face_helper
            self._face_helper = None
        self._cleanup_gpu()

    def _restore_frame(self, bgr, fidelity_weight: float, only_center_face: bool):
        """Restore one BGR frame via CodeFormer. Returns the restored BGR frame."""
        import cv2  # noqa: F401  (imported for parity with inference_codeformer)
        import torch
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize

        self._face_helper.clean_all()
        self._face_helper.read_image(bgr)
        num = self._face_helper.get_face_landmarks_5(
            only_center_face=only_center_face,
            resize=640,
            eye_dist_threshold=5,
        )
        if num == 0:
            return bgr, 0
        self._face_helper.align_warp_face()

        restored_count = 0
        for cropped_face in self._face_helper.cropped_faces:
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self._device)
            try:
                with torch.no_grad():
                    output = self._net(cropped_face_t, w=fidelity_weight, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
            except Exception as e:
                log.warning("CodeFormer inference failed on crop, keeping original: %s", e)
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
            restored_face = restored_face.astype("uint8")
            self._face_helper.add_restored_face(restored_face, cropped_face)
            restored_count += 1

        self._face_helper.get_inverse_affine(None)
        restored_img = self._face_helper.paste_faces_to_input_image(upsample_img=None)
        return restored_img, restored_count

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        import numpy as np

        if self._net is None or self._face_helper is None:
            raise InferenceError("face-restore-codeformer not loaded")

        self._check_cancel(cancel_flag)

        video_file = params.get("video_file")
        if not video_file or not Path(video_file).exists():
            raise InferenceError(f"video_file missing or not found: {video_file}")

        fidelity_weight = float(params.get("fidelity_weight", 0.9))
        if not 0.0 <= fidelity_weight <= 1.0:
            raise InferenceError(f"fidelity_weight must be in [0, 1], got {fidelity_weight}")
        only_center_face = bool(params.get("only_center_face", False))

        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "result.mp4"

        # Probe dims + fps
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "csv=p=0",
                str(video_file),
            ],
            capture_output=True, text=True, check=True,
        )
        w_str, h_str, fps_str = probe.stdout.strip().split(",")
        width = int(w_str)
        height = int(h_str)
        fr_num, fr_den = fps_str.split("/")
        fps = float(fr_num) / float(fr_den) if float(fr_den) > 0 else 24.0
        frame_bytes = width * height * 3

        log.info(
            "face-restore-codeformer: input %dx%d @ %.2f fps, fidelity_weight=%.2f",
            width, height, fps, fidelity_weight,
        )

        dec = subprocess.Popen(
            [
                "ffmpeg", "-v", "error", "-i", str(video_file),
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        tmp_video = str(output_dir / "_restored_video.mp4")
        enc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps}",
                "-i", "-",
                "-c:v", "h264_nvenc", "-preset", "fast", "-pix_fmt", "yuv420p",
                tmp_video,
            ],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if dec.stdout is None or enc.stdin is None:
            raise InferenceError("ffmpeg pipes failed to open")

        frame_count = 0
        faces_found = 0
        try:
            while True:
                self._check_cancel(cancel_flag)
                chunk = dec.stdout.read(frame_bytes)
                if len(chunk) < frame_bytes:
                    break
                bgr = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3).copy()

                with self._gpu_lock:
                    restored_bgr, n_faces = self._restore_frame(
                        bgr, fidelity_weight, only_center_face,
                    )
                if n_faces > 0:
                    faces_found += 1
                enc.stdin.write(restored_bgr.tobytes())
                frame_count += 1
                if frame_count % 100 == 0:
                    log.info(
                        "face-restore-codeformer: processed %d frames (%d with faces)",
                        frame_count, faces_found,
                    )
        except CancelledException:
            dec.kill()
            enc.kill()
            raise
        except Exception as e:
            dec.kill()
            enc.kill()
            raise InferenceError(f"face-restore-codeformer inner loop failed: {e}") from e

        dec.stdout.close()
        enc.stdin.close()
        dec.wait()
        enc.wait()
        if enc.returncode != 0:
            err = enc.stderr.read().decode()[-500:] if enc.stderr else ""
            raise InferenceError(f"ffmpeg encode failed: {err}")

        mux = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", tmp_video,
                "-i", str(video_file),
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v", "-map", "1:a", "-shortest",
                str(result_path),
            ],
            capture_output=True,
        )
        Path(tmp_video).unlink(missing_ok=True)
        if mux.returncode != 0:
            err = mux.stderr.decode()[-500:]
            raise InferenceError(f"ffmpeg mux failed: {err}")

        log.info(
            "face-restore-codeformer done: %d frames, %d with faces (%.1f%%)",
            frame_count, faces_found,
            100.0 * faces_found / max(frame_count, 1),
        )
        return {
            "format": "mp4",
            "file": "result.mp4",
            "frames": frame_count,
            "faces_restored": faces_found,
            "width": width,
            "height": height,
            "fps": fps,
            "fidelity_weight": fidelity_weight,
        }

    def estimate_time(self, params: dict) -> float:
        return 900_000.0  # 15 min default
