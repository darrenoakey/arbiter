"""TRELLIS.2 image-to-3D adapter (microsoft/TRELLIS.2-4B).

Native capability is image-to-3D only. Text-to-3D is not a TRELLIS.2 API;
callers that start from text must produce a reference image through an
external pipeline and submit that image here.

Attention is selected through upstream module setters (dense SDPA). The
adapter never sets ATTN_BACKEND or other attention environment variables.
Sparse attention in upstream TRELLIS.2 has no SDPA backend (flash_attn /
xformers / flash_attn_3 only) and is left on the compiled default.

Background: RGBA with a real alpha channel uses that mask. RGB inputs use
the pipeline's configured briaai/RMBG-2.0 background remover.

GLB export requests PNG textures (extension_webp=False) so Blender can
open the file without the WebP glTF extension. An untextured STL is
optional geometry only.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from arbiter.adapters.base import HeapTrimGuard, InferenceError, ModelAdapter
from arbiter.adapters.registry import register

if TYPE_CHECKING:
    from PIL import Image

log = logging.getLogger(__name__)

HF_ID = "microsoft/TRELLIS.2-4B"
DEFAULT_SNAPSHOT = Path("/home/darren/local/models/trellis2/TRELLIS.2-4B")
ALLOWED_RESOLUTIONS = (512, 1024, 1536)
DEFAULT_RESOLUTION = 1024
DEFAULT_STEPS = 12
MAX_STEPS = 50
DEFAULT_SEED = 42
DEFAULT_TEXTURE_SIZE = 2048
ALLOWED_TEXTURE_SIZES = (1024, 2048, 4096)
DEFAULT_DECIMATION = 500_000
MIN_DECIMATION = 100_000
MAX_DECIMATION = 1_000_000
GLB_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
RESULT_GLB = "result.glb"
RESULT_STL = "geometry.stl"

_RESOLUTION_TO_PIPELINE = {
    512: "512",
    1024: "1024_cascade",
    1536: "1536_cascade",
}


def pipeline_type_for_resolution(resolution: int) -> str:
    if resolution not in _RESOLUTION_TO_PIPELINE:
        raise InferenceError(
            f"resolution must be one of {ALLOWED_RESOLUTIONS}, got {resolution}"
        )
    return _RESOLUTION_TO_PIPELINE[resolution]


def clamp_steps(value: object) -> int:
    return max(1, min(MAX_STEPS, int(value if value is not None else DEFAULT_STEPS)))


def clamp_texture_size(value: object) -> int:
    size = int(value if value is not None else DEFAULT_TEXTURE_SIZE)
    if size in ALLOWED_TEXTURE_SIZES:
        return size
    nearest = min(ALLOWED_TEXTURE_SIZES, key=lambda allowed: abs(allowed - size))
    return nearest


def clamp_decimation(value: object) -> int:
    target = int(value if value is not None else DEFAULT_DECIMATION)
    return max(MIN_DECIMATION, min(MAX_DECIMATION, target))


def clamp_resolution(value: object) -> int:
    resolution = int(value if value is not None else DEFAULT_RESOLUTION)
    if resolution not in ALLOWED_RESOLUTIONS:
        raise InferenceError(
            f"resolution must be one of {ALLOWED_RESOLUTIONS}, got {resolution}"
        )
    return resolution


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_weights_path(
    config_path: Path | None = None,
    snapshot_dir: Path | None = None,
) -> str:
    path = config_path if config_path is not None else repo_root() / "local" / "config.toml"
    if path.is_file():
        data = tomllib.loads(path.read_text())
        configured = data.get("trellis2", {}).get("weights_path")
        if configured:
            snapshot = Path(configured)
            if not snapshot.is_absolute() or not (snapshot / "pipeline.json").is_file():
                raise InferenceError(f"TRELLIS.2 configured snapshot is invalid: {snapshot}")
            return str(snapshot)
    snapshot = snapshot_dir if snapshot_dir is not None else DEFAULT_SNAPSHOT
    if not (snapshot / "pipeline.json").is_file():
        raise InferenceError(f"TRELLIS.2 local snapshot missing pipeline.json: {snapshot}")
    return str(snapshot)


def select_attention_sdpa() -> str:
    from trellis2.modules.attention import config as dense_attn
    from trellis2.modules.sparse import config as sparse_config

    dense_attn.set_backend("sdpa")
    sparse_config.set_attn_backend("flash_attn")
    sparse_config.set_conv_backend("flex_gemm")
    return str(dense_attn.BACKEND)


def sampler_params(steps: int) -> dict[str, int]:
    return {"steps": steps}


def load_job_image(params: dict) -> Image.Image:
    from PIL import Image, ImageOps

    try:
        raw = ModelAdapter._resolve_media(params, "image")
        image = Image.open(io.BytesIO(raw))
        image.load()
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError(
            f"bad input image (cannot decode — corrupt or unsupported format): {exc}"
        ) from exc
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        return image.convert("RGBA")
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def mesh_counts(mesh: Any) -> tuple[int, int]:
    vertices = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    n_vertices = int(vertices.shape[0]) if vertices is not None else 0
    n_faces = int(faces.shape[0]) if faces is not None else 0
    return n_vertices, n_faces


def fsync_file(path: Path) -> None:
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def cuda_peak_bytes() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except ImportError:
        return 0
    return 0


def reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        return


def export_pbr_glb(mesh: Any, glb_path: Path, texture_size: int, decimation: int) -> Any:
    import o_voxel

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=GLB_AABB,
        decimation_target=decimation,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    glb.export(str(glb_path), extension_webp=False)
    fsync_file(glb_path)
    return glb


def export_geometry_stl(mesh: Any, stl_path: Path) -> None:
    import numpy as np
    import trimesh

    vertices = mesh.vertices
    faces = mesh.faces
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()
    geometry = trimesh.Trimesh(
        vertices=np.asarray(vertices),
        faces=np.asarray(faces),
        process=False,
    )
    geometry.export(str(stl_path))
    fsync_file(stl_path)


@register
class Trellis2Adapter(ModelAdapter):
    model_id = "trellis2"

    def __init__(self) -> None:
        self._pipeline: Any = None
        self._device = "cuda"
        self._weights = HF_ID
        self._load_ms = 0.0
        self._load_cuda_peak_bytes = 0
        self._attention = "sdpa"

    def load(self, device: str = "cuda") -> None:
        started = time.perf_counter()
        reset_cuda_peak()
        self._attention = select_attention_sdpa()
        self._weights = resolve_weights_path()
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        log.info("Loading TRELLIS.2 from %s on %s (low_vram=True)", self._weights, device)
        with HeapTrimGuard():
            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self._weights)
            pipeline.low_vram = True
            pipeline.to(device)
        self._pipeline = pipeline
        self._device = device
        self._load_ms = (time.perf_counter() - started) * 1000.0
        self._load_cuda_peak_bytes = cuda_peak_bytes()
        log.info(
            "TRELLIS.2 ready load_ms=%.0f cuda_peak_gb=%.2f",
            self._load_ms,
            self._load_cuda_peak_bytes / (1024**3),
        )

    def unload(self) -> None:
        log.info("Unloading TRELLIS.2.")
        self._pipeline = None
        self._cleanup_gpu()

    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        self._check_cancel(cancel_flag)
        from arbiter.schemas import ImageTo3DParams

        image = load_job_image(params)
        params = ImageTo3DParams.model_validate(params).model_dump()
        if self._pipeline is None:
            raise InferenceError("trellis2 pipeline is not loaded")
        resolution = clamp_resolution(params.get("resolution"))
        pipeline_type = pipeline_type_for_resolution(resolution)
        steps = clamp_steps(params.get("steps"))
        seed = int(params.get("seed", DEFAULT_SEED))
        texture_size = clamp_texture_size(params.get("texture_size"))
        decimation = clamp_decimation(params.get("decimation"))
        include_stl = bool(params.get("include_stl", False))
        return self._run_image_to_3d(
            image=image,
            output_dir=output_dir,
            cancel_flag=cancel_flag,
            resolution=resolution,
            pipeline_type=pipeline_type,
            steps=steps,
            seed=seed,
            texture_size=texture_size,
            decimation=decimation,
            include_stl=include_stl,
        )

    def _run_image_to_3d(
        self,
        image: Image.Image,
        output_dir: Path,
        cancel_flag: threading.Event,
        resolution: int,
        pipeline_type: str,
        steps: int,
        seed: int,
        texture_size: int,
        decimation: int,
        include_stl: bool,
    ) -> dict:
        import torch

        self._check_cancel(cancel_flag)
        reset_cuda_peak()
        started = time.perf_counter()
        stage_params = sampler_params(steps)
        meshes = self._pipeline.run(
            image,
            num_samples=1,
            seed=seed,
            sparse_structure_sampler_params=stage_params,
            shape_slat_sampler_params=stage_params,
            tex_slat_sampler_params=stage_params,
            preprocess_image=True,
            pipeline_type=pipeline_type,
        )
        self._check_cancel(cancel_flag)
        mesh = meshes[0]
        mesh.simplify(16_777_216)  # nvdiffrast's upstream documented face limit
        self._check_cancel(cancel_flag)
        output_dir.mkdir(parents=True, exist_ok=True)
        glb_path = output_dir / RESULT_GLB
        glb = export_pbr_glb(mesh, glb_path, texture_size, decimation)
        self._check_cancel(cancel_flag)
        result = self._result_metadata(
            glb=glb,
            mesh=mesh,
            resolution=resolution,
            pipeline_type=pipeline_type,
            steps=steps,
            seed=seed,
            texture_size=texture_size,
            decimation=decimation,
            infer_ms=(time.perf_counter() - started) * 1000.0,
        )
        if include_stl:
            stl_path = output_dir / RESULT_STL
            export_geometry_stl(mesh, stl_path)
            result["stl_file"] = RESULT_STL
            result["stl_role"] = "geometry"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def _result_metadata(
        self,
        glb: Any,
        mesh: Any,
        resolution: int,
        pipeline_type: str,
        steps: int,
        seed: int,
        texture_size: int,
        decimation: int,
        infer_ms: float,
    ) -> dict:
        vertex_count, face_count = mesh_counts(glb)
        if vertex_count == 0 and face_count == 0:
            vertex_count, face_count = mesh_counts(mesh)
        infer_peak = cuda_peak_bytes()
        return {
            "file": RESULT_GLB,
            "format": "glb",
            "pbr": True,
            "texture_encoding": "png",
            "vertex_count": vertex_count,
            "face_count": face_count,
            "resolution": resolution,
            "pipeline_type": pipeline_type,
            "steps": steps,
            "seed": seed,
            "texture_size": texture_size,
            "decimation": decimation,
            "model": HF_ID,
            "weights": self._weights,
            "low_vram": True,
            "attention": self._attention,
            "load_ms": round(self._load_ms, 1),
            "infer_ms": round(infer_ms, 1),
            "cuda_peak_bytes": infer_peak,
            "cuda_peak_gb": round(infer_peak / (1024**3), 3),
            "load_cuda_peak_bytes": self._load_cuda_peak_bytes,
            "load_cuda_peak_gb": round(self._load_cuda_peak_bytes / (1024**3), 3),
        }

    def estimate_time(self, params: dict) -> float:
        try:
            resolution = clamp_resolution(params.get("resolution"))
        except InferenceError:
            resolution = DEFAULT_RESOLUTION
        steps = clamp_steps(params.get("steps"))
        texture_size = clamp_texture_size(params.get("texture_size"))
        bases = {512: 25_000.0, 1024: 90_000.0, 1536: 240_000.0}
        glb_ms = 20_000.0 + (texture_size / 1024.0) * 8_000.0
        return bases[resolution] * (steps / DEFAULT_STEPS) + glb_ms
