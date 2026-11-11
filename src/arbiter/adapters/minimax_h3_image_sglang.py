"""MiniMax-H3-Image through SGLang-Diffusion: MXFP8 DiT + encoder, FP8 attention.

Runs SGLang's DGX Spark recipe, which upstream measured fastest. The pin is PR
49599 head ``fd31ed2`` (``codex/h3-image-spark-performance``): ``--quantization
mxfp8 --component-quantizations.text_encoder mxfp8
--component-attention-backends.transformer fp8_fa_sm120``. All components stay
resident, AdaLN precompute is native, and execution is eager. The VAE and AdaLN
tables keep native precision. Both MXFP8 and FP8 attention change pixels
relative to BF16; upstream calls the recipe approximate.

Weights come from the same pinned ``MiniMaxAI/MiniMax-H3-Image`` revision as the
BF16 adapter, so the comparison runs on identical source weights. SGLang loads
the release's native ``transformer/``, ``vae/`` and PDD-head files, and reuses
the BF16 snapshot's verified ``text_encoder/`` through ``component_paths``.
Every file read is hash-verified against Hugging Face's download metadata
before SGLang starts. Shared tokenizer, processor and scheduler files must be
hard links to the verified BF16 files.

SGLang's generator runs a scheduler child process, which does not inherit
``torch.cuda.set_per_process_memory_fraction``. The venv's ``sitecustomize.py``
must apply the cap from ``ARBITER_MEMORY_GB``, and ``load()`` refuses to start
without it. Never ComfyUI.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from arbiter.adapters.base import InferenceError, LoadError, ModelAdapter
from arbiter.adapters.flux3_klein import verify_snapshot_file
from arbiter.adapters.minimax_h3_image import (
    DEFAULT_WEIGHTS_DIR,
    EXPECTED_SAMPLE_SIGMAS,
    EXPECTED_STEPS,
    MINIMAX_H3_IMAGE_HF_REPO,
    MINIMAX_H3_IMAGE_REVISION,
    StageProgressMixin,
    parse_request,
    verify_h3_image_snapshot,
)
from arbiter.adapters.qwen_image_21_sglang import _child_cuda_cap_installed
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

MODEL_ID = "minimax-h3-image-sglang-mxfp8"
SGLANG_COMMIT = "fd31ed2ecd5e4c6fc85718604e59a5406e38c460"
SGLANG_SOURCE = "https://github.com/sgl-project/sglang/pull/49599"
NATIVE_WEIGHTS_DIR = Path("/home/darren/nvme-models/minimax-h3-image-sglang/MiniMax-H3-Image")

#: Native-layout files downloaded beside the BF16 snapshot, hash-verified here.
NATIVE_DOWNLOADED = ("transformer", "vae", "pdd_config.json", "pdd_fused_heads.safetensors")
#: Small shared files, which must be hard links into the verified BF16 snapshot.
SHARED_WITH_BF16 = ("LICENSE", "model_index.json", "modular_model_index.json", "tokenizer", "processor", "scheduler")

#: The exact upstream Spark recipe (cookbook "DGX Spark with MXFP8 and FP8 attention").
RECIPE = {
    "quantization": "mxfp8",
    "component_quantizations": {"text_encoder": "mxfp8"},
    "component_attention_backends": {"transformer": "fp8_fa_sm120"},
    "component_residency": {"all": "resident"},
}


def _files(entry: Path, rel: str) -> list[str]:
    target = entry / rel
    if target.is_symlink() or not target.exists():
        raise LoadError(f"{MODEL_ID} snapshot is missing {target}")
    if target.is_file():
        return [rel]
    return sorted(
        str(p.relative_to(entry)) for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts
    )


def verify_native_snapshot(entry: Path, bf16_entry: Path, revision: str = MINIMAX_H3_IMAGE_REVISION) -> dict:
    """Fail closed unless every file SGLang reads is the pinned revision."""
    if entry.is_symlink() or not entry.is_dir():
        raise LoadError(f"{MODEL_ID} weights dir is not a real directory: {entry}")
    if (entry / "text_encoder").exists():
        raise LoadError(f"{MODEL_ID} snapshot must not carry its own text_encoder; it reuses {bf16_entry}")
    verified = 0
    for rel in NATIVE_DOWNLOADED:
        for relpath in _files(entry, rel):
            verify_snapshot_file(entry, relpath, revision)
            verified += 1
    for rel in SHARED_WITH_BF16:
        for relpath in _files(entry, rel):
            if not os.path.samefile(entry / relpath, bf16_entry / relpath):
                raise LoadError(f"{MODEL_ID} {relpath} is not the verified BF16 snapshot file")
    bf16 = verify_h3_image_snapshot(bf16_entry, revision)
    return {
        "hf_repo": MINIMAX_H3_IMAGE_HF_REPO,
        "revision": revision,
        "weights_dir": str(entry),
        "text_encoder_dir": str(bf16_entry / "text_encoder"),
        "verified_native_files": verified,
        "verified_bf16_files": bf16["verified_files"],
    }


def _process_tree_gpu_bytes() -> int:
    """Unified-memory bytes nvidia-smi attributes to this worker and its SGLang children."""
    me = os.getpid()
    tree = {me}
    try:
        ps = subprocess.run(["ps", "-e", "-o", "pid=,ppid="], capture_output=True, text=True, check=True).stdout
        parents = [tuple(int(x) for x in line.split()) for line in ps.splitlines() if line.strip()]
        grew = True
        while grew:
            grew = False
            for pid, ppid in parents:
                if ppid in tree and pid not in tree:
                    tree.add(pid)
                    grew = True
        smi = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError, ValueError):
        return -1
    total = 0
    for line in smi.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and int(parts[0]) in tree:
            total += int(parts[1]) * 1024 * 1024
    return total


@register
class MiniMaxH3ImageSglangAdapter(StageProgressMixin, ModelAdapter):
    """Resident SGLang MXFP8 + FP8-attention MiniMax-H3-Image; never ComfyUI."""

    model_id = MODEL_ID
    result_precision = {
        "transformer": "mxfp8 (online, block-scaled)",
        "text_encoder": "mxfp8 (online, block-scaled)",
        "attention": "fp8_fa_sm120 (E4M3 Q/K/V)",
        "adaln_tables": "native precompute",
        "vae": "native",
    }
    result_limitations = (
        "SGLang PR 49599 (unmerged) Spark recipe: MXFP8 weights and FP8 attention change pixels "
        "relative to BF16 at the same seed; upstream calls this approximate.",
    )

    def __init__(self, models_dir: Path | None = None, bf16_dir: Path | None = None) -> None:
        self.models_dir = Path(models_dir or NATIVE_WEIGHTS_DIR)
        self.bf16_dir = Path(bf16_dir or DEFAULT_WEIGHTS_DIR)
        self._gen: object | None = None
        self._load_info: dict | None = None
        self._renders = 0
        self._lock = threading.Lock()

    def generator_kwargs(self) -> dict:
        return {
            "model_path": str(self.models_dir),
            "num_gpus": 1,
            "component_paths": {"text_encoder": str(self.bf16_dir / "text_encoder")},
            **{k: (dict(v) if isinstance(v, dict) else v) for k, v in RECIPE.items()},
        }

    def load(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise LoadError(f"{MODEL_ID} requires cuda")
        declared = os.environ.get("ARBITER_MEMORY_GB")
        if not declared:
            raise LoadError(f"{MODEL_ID} requires Arbiter's CUDA memory cap")
        if not _child_cuda_cap_installed():
            raise LoadError("sglang venv is missing the CUDA memory cap sitecustomize; refusing an uncapped scheduler")
        self._start_progress("verify pinned H3-Image snapshot and start resident SGLang MXFP8 pipeline")
        started = time.perf_counter()
        try:
            weights = verify_native_snapshot(self.models_dir, self.bf16_dir)
            verified = time.perf_counter()
            from importlib import metadata

            try:
                sglang_version = metadata.version("sglang")
            except metadata.PackageNotFoundError as error:
                raise LoadError(f"{MODEL_ID} worker venv has no sglang install (expected {SGLANG_COMMIT[:12]})") from error
            entrypoint = importlib.import_module("sglang.multimodal_gen.runtime.entrypoints.diffusion_generator")
            self._gen = entrypoint.DiffGenerator.from_pretrained(**self.generator_kwargs())
            loaded = time.perf_counter()
        except BaseException:
            self.unload()
            raise
        finally:
            self._stop_progress()
        self._renders = 0
        self._load_info = {
            "weights": weights,
            "runtime": {
                "runtime": "sglang-diffusion",
                "sglang_source": SGLANG_SOURCE,
                "sglang_commit": SGLANG_COMMIT,
                "sglang_package_version": sglang_version,
                "recipe": RECIPE,
            },
            "declared_memory_gb": float(declared),
            "load_seconds": round(loaded - started, 3),
            "verify_seconds": round(verified - started, 3),
            "pipeline_load_seconds": round(loaded - verified, 3),
            "resident_process_tree_bytes": _process_tree_gpu_bytes(),
        }
        log.info("%s loaded: %s", MODEL_ID, self._load_info)

    def unload(self) -> None:
        gen, self._gen = self._gen, None
        self._load_info = None
        if gen is not None:
            shutdown = getattr(gen, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    log.exception("SGLang generator shutdown failed")
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        job_type = params.get("_job_type") or MODEL_ID
        if job_type != MODEL_ID:
            raise InferenceError(f"{MODEL_ID} serves only the {MODEL_ID!r} job type, not {job_type!r}")
        request = parse_request(params)
        self._check_cancel(cancel_flag)
        with self._lock:
            if self._gen is None or self._load_info is None:
                raise InferenceError(f"{MODEL_ID} is not loaded")
            self._start_progress("SGLang MXFP8 render (six fixed PDD steps)")
            try:
                return self._render(request, output_dir, cancel_flag)
            finally:
                self._stop_progress()

    def _render(self, request, output_dir: Path, cancel_flag: threading.Event) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        call = {
            "prompt": request.prompt,
            "seed": request.seed,
            "width": request.width,
            "height": request.height,
            "min_generation_area": request.min_generation_area,
            "save_output": True,
            "output_path": str(output_dir),
            "output_file_name": "result.png",
        }
        started = time.perf_counter()
        result = self._gen.generate(sampling_params_kwargs=call)  # type: ignore[attr-defined]
        finished = time.perf_counter()
        self._check_cancel(cancel_flag)
        if isinstance(result, list):
            if len(result) != 1:
                raise InferenceError(f"{MODEL_ID} returned {len(result)} results for one request")
            result = result[0]
        if result is None:
            raise InferenceError(f"{MODEL_ID} returned no result")
        out = Path(getattr(result, "output_file_path", None) or output_dir / "result.png")
        if out.resolve() != (output_dir / "result.png").resolve() or not out.is_file() or out.stat().st_size == 0:
            raise InferenceError(f"{MODEL_ID} did not write result.png (got {out})")
        from PIL import Image

        with Image.open(out) as saved:
            saved.load()
            delivered = saved.size
        if delivered != (request.width, request.height):
            raise InferenceError(
                f"{MODEL_ID} delivered {delivered[0]}x{delivered[1]}, expected {request.width}x{request.height}"
            )
        self._renders += 1
        info = self._load_info
        metrics = dict(getattr(result, "metrics", None) or {})
        return {
            "format": "png",
            "file": "result.png",
            "mode": "t2i",
            "model": MINIMAX_H3_IMAGE_HF_REPO,
            "model_id": MODEL_ID,
            "prompt": request.prompt,
            "seed": request.seed,
            "lane": request.lane,
            "steps": EXPECTED_STEPS,
            "sample_sigmas": list(EXPECTED_SAMPLE_SIGMAS),
            "guidance_scale": 1,
            "negative_prompt": None,
            "min_generation_area": request.min_generation_area,
            "sizes": {
                "requested": {"width": request.width, "height": request.height},
                "delivered": {"width": delivered[0], "height": delivered[1]},
            },
            "width": delivered[0],
            "height": delivered[1],
            "precision": dict(self.result_precision),
            "license": {"confirmed": True, "confirmation_ref": request.license_confirmation_ref},
            "provenance": {**info["weights"], **info["runtime"]},
            "limitations": list(self.result_limitations),
            "timings": {
                "render_index": self._renders,
                "total_seconds": round(finished - started, 3),
                "sglang_generation_seconds": round(float(getattr(result, "generation_time", 0.0) or 0.0), 3),
                "sglang_metrics": metrics,
                "load_seconds": info["load_seconds"],
                "verify_seconds": info["verify_seconds"],
                "pipeline_load_seconds": info["pipeline_load_seconds"],
                "component_swaps": 0,
            },
            "memory": {
                "declared_memory_gb": info["declared_memory_gb"],
                "resident_process_tree_bytes": info["resident_process_tree_bytes"],
                "sglang_peak_memory_mb": float(getattr(result, "peak_memory_mb", 0.0) or 0.0),
                "process_tree_bytes_after_render": _process_tree_gpu_bytes(),
            },
        }

    def estimate_time(self, params: dict) -> float:
        # Upstream GB10 medians: 2048^2 ~11 s, 4736x3552 ~56 s. Estimates only.
        return 90.0 if params.get("lane") == "recommended-4096" else 20.0
