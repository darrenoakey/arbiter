"""FLUX 3 Image Klein (distilled) — exact-variant Arbiter benchmark adapters.

Owner-directed benchmark exception (2026-10-16): the owner explicitly asked to
install FLUX 3 Image Klein and benchmark BF16, FP8 and NVFP4 head to head
against Qwen-Image-2.1. Each precision is its own exact model id and job type:

* ``flux3-klein-bf16``  — repository-root DiT (``model.safetensors``, bf16)
* ``flux3-klein-fp8``   — ``variants/fp8r`` (rowwise FP8 linears)
* ``flux3-klein-nvfp4`` — ``variants/nvfp4`` (NVFP4 linears via FlashInfer)

None of them is wired into ``generate_image`` / ``daz-agent-sdk`` / the Mac
mini Codex IGS route, and none is a fallback for IGS. Every other ``flux*`` id
stays denied by ``image_policy.py`` and the Go ``model_policy.go``.

Runtime contract
----------------
* The official reference package ``flux3`` (black-forest-labs/
  flux-3-image-klein at source commit ``FLUX3_SOURCE_COMMIT``) is used
  directly: ``flux3.pipeline.Pipeline`` with an explicit DiT dict, text-encoder
  directory and ``ae.safetensors`` path, then ``Pipeline.generate`` under
  ``torch.inference_mode``. ``flux3.generate.Models`` is deliberately not used:
  it imports vLLM for the prompt upsampler and loads the safety classifier,
  neither of which belongs in a matched-prompt benchmark. No ``WEIGHTS_DIR``
  environment variable is read or needed.
* Distilled flow-map model only: 16 steps by default, no CFG, no prompt
  upsampling, 1024x1024 by default, explicit seed.
* Weights root is the ``FLUX3_WEIGHTS_ROOT`` adapter param (Go allowlists it
  for these three model ids only), default ``DEFAULT_WEIGHTS_ROOT``. The
  entry ``flux-3-image-klein`` under it is a ``snapshot_download`` local dir
  pinned to ``FLUX3_REVISION``.
* Load is fail-closed: it refuses to run without the Arbiter worker CUDA cap
  (``ARBITER_MEMORY_GB``), verifies the installed ``flux3`` distribution was
  installed from the pinned source commit, verifies every file the pipeline
  reads against its Hugging Face snapshot metadata (revision + content hash),
  checks the DiT manifest quantization matches the requested variant, and
  after loading checks the DiT contains exactly the expected quantized linear
  class and count. A mislabelled path can never silently load another
  precision.
* Cancellation: checked before text encoding and before every DiT forward
  (one per denoising step) via a forward pre-hook, so a cancel stops within
  one step without re-implementing the reference sampler.
* Results carry measured timings, peak allocated/reserved CUDA memory, and
  prompt/seed/revision/precision/source provenance.

``torch`` and ``flux3`` are imported only inside ``load``/``infer`` so the
main-venv registry import stays clean. See ``requirements/flux3.txt`` and
``docs/flux3-klein.md``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from arbiter.adapters.base import (
    CancelledException,
    InferenceError,
    LoadError,
    ModelAdapter,
)
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

FLUX3_WEIGHTS_ROOT_PARAM = "FLUX3_WEIGHTS_ROOT"
DEFAULT_WEIGHTS_ROOT = Path("/home/darren/nvme-models/flux3")
FLUX3_ENTRY = "flux-3-image-klein"
FLUX3_HF_REPO = "black-forest-labs/flux-3-image-klein"
FLUX3_REVISION = "8454017f9ebaada30c998fccd9f67a80c7580f68"
FLUX3_SOURCE_REPO = "https://github.com/black-forest-labs/flux-3-image-klein"
FLUX3_SOURCE_COMMIT = "45e57c9fb06f9751cdced29a65c5d2171003be05"
FLUX3_DISTRIBUTION = "flux3"
MANIFEST_KIND = "f3_dit_export"

MODEL_KIND = "distilled"  # key of flux3.pipeline.MODELS (flow map, no CFG)
DEFAULT_STEPS = 16
SAMPLER_ALPHA = 6.93  # flux3.pipeline.Pipeline.generate default; pinned by config.json
DEFAULT_SIDE = 1024
MIN_SIDE = 64
MAX_SIDE = 4096
SIDE_MULTIPLE = 16
MAX_STEPS = 64
MAX_SEED = 2**63 - 1
MAX_PROMPT_CHARS = 16000

DIT_FILES = ("manifest.json", "config.json", "model.safetensors")
AUTOENCODER_FILE = "ae.safetensors"
TEXT_ENCODER_DIR = "text_encoder"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HASH_CHUNK = 16 * 1024 * 1024

#: Params the reference pipeline could use but this matched-prompt adapter
#: must never honour silently. A present non-default value fails the job.
_UNSUPPORTED_PARAMS = (
    "guidance",
    "guidance_scale",
    "cfg_scale",
    "true_cfg_scale",
    "negative_prompt",
    "pup",
    "upsample",
    "prompt_upsampling",
    "safety",
    "boxes",
    "image",
    "image_file",
    "images",
    "image_files",
    "references",
    "offload",
    "model_kind",
)


@dataclass(frozen=True)
class Flux3Variant:
    """One exact precision of the distilled DiT."""

    model_id: str
    name: str  # bf16 / fp8 / nvfp4 (result label)
    quantization: str | None  # manifest/config "quantization" value
    linear_class: str | None  # flux3 quantized nn.Module class name
    linear_count_key: str | None  # manifest key holding the expected count

    @property
    def dit_relpath(self) -> str:
        return f"variants/{self.quantization}" if self.quantization else ""


VARIANTS: dict[str, Flux3Variant] = {
    v.model_id: v
    for v in (
        Flux3Variant("flux3-klein-bf16", "bf16", None, None, None),
        Flux3Variant("flux3-klein-fp8", "fp8", "fp8r", "FP8Linear", "fp8_matrices"),
        Flux3Variant("flux3-klein-nvfp4", "nvfp4", "nvfp4", "NVFP4Linear", "nvfp4_matrices"),
    )
}


@dataclass(frozen=True)
class Flux3Request:
    prompt: str
    seed: int
    width: int
    height: int
    steps: int


def resolve_weights_root(environ: Mapping[str, str]) -> Path:
    """The adapter-param weights root: absolute, canonical, an existing real
    directory. Never a relative path, never a symlink."""
    raw = environ.get(FLUX3_WEIGHTS_ROOT_PARAM)
    value = str(DEFAULT_WEIGHTS_ROOT) if raw is None else raw
    if not value or not os.path.isabs(value) or os.path.normpath(value) != value:
        raise LoadError(f"{FLUX3_WEIGHTS_ROOT_PARAM} must be an absolute canonical path: {value!r}")
    path = Path(value)
    if path.is_symlink() or not path.is_dir():
        raise LoadError(f"{FLUX3_WEIGHTS_ROOT_PARAM} is not a real directory: {value}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot_file(entry: Path, relpath: str, revision: str) -> str:
    """Verify one ``snapshot_download(local_dir=...)`` file against the
    metadata Hugging Face wrote beside it: the commit must be ``revision`` and
    the content must hash to the recorded etag (sha256 for LFS objects, the
    git blob sha1 for regular files). Returns the verified sha256 for LFS
    files or the git blob sha1 otherwise."""
    target = entry / relpath
    if target.is_symlink() or not target.is_file():
        raise LoadError(f"flux3 weight file missing or not a regular file: {target}")
    metadata = entry / ".cache" / "huggingface" / "download" / f"{relpath}.metadata"
    try:
        lines = metadata.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise LoadError(f"flux3 snapshot metadata missing for {relpath}: {error}") from error
    if len(lines) < 2:
        raise LoadError(f"flux3 snapshot metadata malformed for {relpath}")
    commit, etag = lines[0].strip(), lines[1].strip().strip('"').lower()
    if commit != revision:
        raise LoadError(f"flux3 {relpath} is from revision {commit}, expected {revision}")
    if _HEX64.match(etag):
        actual = _sha256(target)
    elif _HEX40.match(etag):
        actual = _git_blob_sha1(target)
    else:
        raise LoadError(f"flux3 snapshot metadata for {relpath} has no content hash")
    if actual != etag:
        raise LoadError(f"flux3 {relpath} content hash {actual} does not match snapshot {etag}")
    return actual


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LoadError(f"flux3 cannot read {path}: {error}") from error
    if not isinstance(data, dict):
        raise LoadError(f"flux3 {path} is not a JSON object")
    return data


def verify_weights(entry: Path, variant: Flux3Variant, revision: str = FLUX3_REVISION) -> dict:
    """Fail-closed verification of everything the pipeline will read for
    ``variant``. Returns the weights provenance recorded in every result."""
    if entry.is_symlink() or not entry.is_dir():
        raise LoadError(f"flux3 weights entry is not a real directory: {entry}")
    dit_dir = entry / variant.dit_relpath if variant.dit_relpath else entry
    prefix = f"{variant.dit_relpath}/" if variant.dit_relpath else ""

    manifest = _read_json(dit_dir / "manifest.json")
    config = _read_json(dit_dir / "config.json")
    if manifest.get("kind") != MANIFEST_KIND or manifest.get("model") != FLUX3_ENTRY:
        raise LoadError(f"flux3 {dit_dir} manifest is not a {FLUX3_ENTRY} {MANIFEST_KIND}")
    for label, document in (("manifest", manifest), ("config", config)):
        if document.get("quantization") != variant.quantization:
            raise LoadError(
                f"flux3 {variant.model_id} expects quantization {variant.quantization!r}; "
                f"{label} at {dit_dir} declares {document.get('quantization')!r}"
            )
    if config.get("model") != FLUX3_ENTRY or (config.get("dit") or {}).get("flow_map") is not True:
        raise LoadError(f"flux3 {dit_dir} is not the distilled flow-map DiT")
    sampler = config.get("sampler") or {}
    if sampler.get("guidance") is not None or sampler.get("alpha") != SAMPLER_ALPHA:
        raise LoadError(f"flux3 {dit_dir} sampler {sampler} differs from the pinned distilled sampler")
    expected_count = None
    if variant.linear_count_key is not None:
        expected_count = manifest.get(variant.linear_count_key)
        if not isinstance(expected_count, int) or expected_count <= 0:
            raise LoadError(f"flux3 {dit_dir} manifest lacks {variant.linear_count_key}")

    hashes = manifest.get("sha256") or {}
    verified: dict[str, str] = {}
    for name in DIT_FILES:
        relpath = prefix + name
        digest = verify_snapshot_file(entry, relpath, revision)
        if name in ("config.json", "model.safetensors"):
            declared = hashes.get(name)
            if not isinstance(declared, str) or not _HEX64.match(declared):
                raise LoadError(f"flux3 {dit_dir} manifest has no sha256 for {name}")
            # LFS files were just verified by sha256; small git files by blob sha1.
            actual = digest if _HEX64.match(digest) else _sha256(dit_dir / name)
            if actual != declared:
                raise LoadError(f"flux3 {relpath} sha256 {actual} does not match manifest {declared}")
        verified[relpath] = digest

    verified[AUTOENCODER_FILE] = verify_snapshot_file(entry, AUTOENCODER_FILE, revision)
    text_dir = entry / TEXT_ENCODER_DIR
    if text_dir.is_symlink() or not text_dir.is_dir():
        raise LoadError(f"flux3 text encoder directory missing: {text_dir}")
    text_files = sorted(p.name for p in text_dir.iterdir())
    if not text_files:
        raise LoadError(f"flux3 text encoder directory is empty: {text_dir}")
    for name in text_files:
        relpath = f"{TEXT_ENCODER_DIR}/{name}"
        verified[relpath] = verify_snapshot_file(entry, relpath, revision)

    return {
        "hf_repo": FLUX3_HF_REPO,
        "revision": revision,
        "weights_entry": str(entry),
        "dit_path": str(dit_dir),
        "dit_sha256": hashes["model.safetensors"],
        "dit_source_sha256": manifest.get("source_sha256"),
        "dit_dtype": manifest.get("dtype"),
        "quantization": variant.quantization,
        "expected_quantized_linears": expected_count,
        "verified_files": len(verified),
    }


def source_provenance(distribution_name: str = FLUX3_DISTRIBUTION) -> dict:
    """The installed reference package must come from the pinned commit.
    ``pip install --no-deps 'flux3 @ git+<repo>@<commit>'`` (remote or
    ``git+file://``) records that commit in ``direct_url.json``."""
    from importlib import metadata

    try:
        dist = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError as error:
        raise LoadError(f"{distribution_name} is not installed in this worker venv") from error
    raw = dist.read_text("direct_url.json")
    if not raw:
        raise LoadError(f"{distribution_name} has no direct_url.json; install it from the pinned git commit")
    try:
        direct = json.loads(raw)
    except ValueError as error:
        raise LoadError(f"{distribution_name} direct_url.json is malformed") from error
    commit = (direct.get("vcs_info") or {}).get("commit_id")
    if commit != FLUX3_SOURCE_COMMIT:
        raise LoadError(
            f"{distribution_name} was installed from {direct.get('url')} at {commit!r}; "
            f"expected commit {FLUX3_SOURCE_COMMIT}"
        )
    return {
        "source_repo": FLUX3_SOURCE_REPO,
        "source_commit": commit,
        "source_url": direct.get("url"),
        "package_version": dist.version,
        "package_location": str(dist.locate_file("")),
    }


def parse_request(params: dict) -> Flux3Request:
    """Validate a job's params. Unsupported knobs fail instead of being
    silently ignored, so a benchmark row can never be mislabelled."""
    for key in _UNSUPPORTED_PARAMS:
        if params.get(key) not in (None, False):
            raise InferenceError(f"flux3 klein distilled does not support {key!r} (t2i, no CFG, no upsampler)")
    prompt = params.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InferenceError("prompt is required")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise InferenceError(f"prompt exceeds {MAX_PROMPT_CHARS} characters")

    def integer(key: str, default: int, low: int, high: int) -> int:
        value = params.get(key, default)
        if value is None:
            value = default
        if isinstance(value, bool) or not isinstance(value, int):
            raise InferenceError(f"{key} must be an integer")
        if not low <= value <= high:
            raise InferenceError(f"{key} must be between {low} and {high}")
        return value

    width = integer("width", DEFAULT_SIDE, MIN_SIDE, MAX_SIDE)
    height = integer("height", DEFAULT_SIDE, MIN_SIDE, MAX_SIDE)
    if width % SIDE_MULTIPLE or height % SIDE_MULTIPLE:
        raise InferenceError(f"width and height must be multiples of {SIDE_MULTIPLE}")
    return Flux3Request(
        prompt=prompt,
        seed=integer("seed", 42, 0, MAX_SEED),
        width=width,
        height=height,
        steps=integer("steps", DEFAULT_STEPS, 1, MAX_STEPS),
    )


def write_png_durably(path: Path, image) -> None:
    """fsync data and directory: queue eviction may SIGKILL the worker right
    after it returns, and the output share drops unsynced writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        image.save(handle, format="PNG")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        pass


class Flux3KleinAdapter(ModelAdapter):
    """Shared implementation; only the exact per-variant subclasses register."""

    variant: Flux3Variant

    def __init__(self) -> None:
        self._pipeline = None
        self._load_info: dict | None = None
        self._renders = 0
        self._lock = threading.Lock()

    def load(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise LoadError(f"{self.model_id} requires cuda")
        declared = os.environ.get("ARBITER_MEMORY_GB")
        if not declared:
            raise LoadError(
                f"{self.model_id} refuses to load without the Arbiter worker CUDA cap "
                "(ARBITER_MEMORY_GB); GPU work runs only through the Arbiter queue"
            )
        started = time.perf_counter()
        entry = resolve_weights_root(os.environ) / FLUX3_ENTRY
        source = source_provenance()
        weights = verify_weights(entry, self.variant)
        verified = time.perf_counter()

        import torch
        from flux3 import checkpoint as flux3_checkpoint
        from flux3 import pipeline as flux3_pipeline

        module_path = Path(flux3_pipeline.__file__).resolve()
        if not module_path.is_relative_to(Path(source["package_location"]).resolve()):
            raise LoadError(f"flux3 imported from {module_path}, not the pinned distribution")
        if flux3_pipeline.MODELS[MODEL_KIND]["guidance"] is not None:
            raise LoadError("flux3 reference MODELS['distilled'] is no longer a flow map")
        dit_dir = weights["dit_path"]
        if not flux3_checkpoint.is_flow_map(dit_dir):
            raise LoadError(f"{dit_dir} is not a flow-map (distilled) checkpoint")

        try:
            pipeline = flux3_pipeline.Pipeline(
                {MODEL_KIND: dit_dir},
                str(entry / TEXT_ENCODER_DIR),
                str(entry / AUTOENCODER_FILE),
                device=device,
                offload=False,
            )
        except SystemExit as error:  # the reference signals mismatches this way
            raise LoadError(f"flux3 pipeline refused {dit_dir}: {error}") from error
        torch.cuda.synchronize()
        loaded = time.perf_counter()

        dit = pipeline.dits[MODEL_KIND]
        counts = {
            name: sum(type(module).__name__ == name for module in dit.modules())
            for name in ("FP8Linear", "NVFP4Linear")
        }
        expected = {name: 0 for name in counts}
        if self.variant.linear_class is not None:
            expected[self.variant.linear_class] = weights["expected_quantized_linears"]
        if counts != expected:
            pipeline = None
            self._cleanup_gpu()
            raise LoadError(f"{self.model_id} DiT quantized linears {counts} != expected {expected}")

        runtime = {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(),
        }
        if self.variant.quantization == "nvfp4":
            import flashinfer

            runtime["flashinfer_version"] = getattr(flashinfer, "__version__", "unknown")
        self._pipeline = pipeline
        self._renders = 0
        self._load_info = {
            "source": source,
            "weights": weights,
            "runtime": runtime,
            "quantized_linears": counts,
            "declared_memory_gb": float(declared),
            "verify_seconds": round(verified - started, 3),
            "pipeline_load_seconds": round(loaded - verified, 3),
            "load_seconds": round(loaded - started, 3),
            "resident_allocated_bytes": torch.cuda.memory_allocated(),
            "resident_reserved_bytes": torch.cuda.memory_reserved(),
        }
        log.info("%s loaded: %s", self.model_id, self._load_info)

    def unload(self) -> None:
        self._pipeline = None
        self._load_info = None
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        job_type = params.get("_job_type") or self.model_id
        if job_type != self.model_id:
            raise InferenceError(f"{self.model_id} serves only the {self.model_id!r} job type, not {job_type!r}")
        request = parse_request(params)
        self._check_cancel(cancel_flag)
        with self._lock:
            if self._pipeline is None or self._load_info is None:
                raise InferenceError(f"{self.model_id} is not loaded")
            return self._render(request, output_dir, cancel_flag)

    def _render(self, request: Flux3Request, output_dir: Path, cancel_flag: threading.Event) -> dict:
        import torch

        pipeline = self._pipeline
        dit = pipeline.dits[MODEL_KIND]
        forwards = 0

        def cancel_between_steps(_module, _args, _kwargs):
            nonlocal forwards
            if cancel_flag.is_set():
                raise CancelledException(f"Job cancelled for model {self.model_id}")
            forwards += 1

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        handle = dit.register_forward_pre_hook(cancel_between_steps, with_kwargs=True)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                image = pipeline.generate(
                    MODEL_KIND,
                    request.prompt,
                    [],
                    request.width,
                    request.height,
                    request.steps,
                    None,
                    request.seed,
                )
            torch.cuda.synchronize()
        finally:
            handle.remove()
        generated = time.perf_counter()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        self._check_cancel(cancel_flag)

        write_png_durably(output_dir / "result.png", image)
        written = time.perf_counter()
        self._renders += 1
        info = self._load_info
        gib = float(1024**3)
        return {
            "format": "png",
            "file": "result.png",
            "mode": "t2i",
            "model": FLUX3_HF_REPO,
            "model_id": self.model_id,
            "model_kind": MODEL_KIND,
            "variant": self.variant.name,
            "prompt": request.prompt,
            "seed": request.seed,
            "width": image.width,
            "height": image.height,
            "steps": request.steps,
            "guidance": None,
            "prompt_upsampling": False,
            "sampler_alpha": SAMPLER_ALPHA,
            "precision": {
                "dit": info["weights"]["dit_dtype"],
                "quantization": self.variant.quantization,
                "quantized_linears": info["quantized_linears"],
                "text_encoder": "bfloat16",
                "autoencoder": "float32",
            },
            "provenance": {**info["weights"], **info["source"], **info["runtime"]},
            "timings": {
                "render_index": self._renders,
                "generate_seconds": round(generated - started, 3),
                "png_write_seconds": round(written - generated, 3),
                "total_seconds": round(written - started, 3),
                "dit_forwards": forwards,
                "load_seconds": info["load_seconds"],
                "verify_seconds": info["verify_seconds"],
                "pipeline_load_seconds": info["pipeline_load_seconds"],
            },
            "memory": {
                "declared_memory_gb": info["declared_memory_gb"],
                "resident_allocated_bytes": info["resident_allocated_bytes"],
                "resident_reserved_bytes": info["resident_reserved_bytes"],
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "peak_allocated_gb": round(peak_allocated / gib, 3),
                "peak_reserved_gb": round(peak_reserved / gib, 3),
            },
        }

    def estimate_time(self, params: dict) -> float:
        steps = params.get("steps", DEFAULT_STEPS)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            steps = DEFAULT_STEPS
        return 10000.0 + 3000.0 * steps


@register
class Flux3KleinBF16Adapter(Flux3KleinAdapter):
    model_id = "flux3-klein-bf16"
    variant = VARIANTS["flux3-klein-bf16"]


@register
class Flux3KleinFP8Adapter(Flux3KleinAdapter):
    model_id = "flux3-klein-fp8"
    variant = VARIANTS["flux3-klein-fp8"]


@register
class Flux3KleinNVFP4Adapter(Flux3KleinAdapter):
    model_id = "flux3-klein-nvfp4"
    variant = VARIANTS["flux3-klein-nvfp4"]
