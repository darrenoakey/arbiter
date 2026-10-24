"""QLabs noise-model training, run in-process inside the Arbiter CUDA worker.

The qlabs-noise-spark project owns the experiment code; this adapter is the
only sanctioned way to run it on the GB10. The worker imports the pinned
project file in-process (never a subprocess), so training runs under the
worker's audited ``ARBITER_MEMORY_GB`` CUDA cap instead of the unbounded
side-channel runs that livelock the unified-memory host.

The architecture is fixed — the original 37.7M-parameter model (D512, L8,
H8, vocab 4096) — so runs stay comparable. Callers may only choose
steps/batch/sequence/seed, and the kernel grids require the sequence length
to be a multiple of 64 (64/128/256).
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import sys
import threading
import time
from pathlib import Path

from arbiter.adapters.base import InferenceError, LoadError, ModelAdapter
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

# Pinned deploy location of the parent's isolated project. The experiment
# file is imported in-process; it must come from exactly here.
PROJECT_DIR = Path("/home/darren/src/qlabs-noise-spark")
EXPERIMENT_FILE = PROJECT_DIR / "src" / "experiment.py"
_MODULE_NAME = "arbiter_qlabs_noise_experiment"

# The only caller-tunable surface. Everything else about the run — data,
# architecture, kernels, optimizer — is owned by the pinned experiment file.
# ``force`` is the Arbiter dedup-protocol flag; it must be a bool and is
# accepted only to be ignored (dedup is the queue's job, not the trainer's).
_ALLOWLISTED_PARAMS = {"steps", "batch", "sequence", "seed", "force"}
_DEFAULTS = {"steps": 100, "batch": 2, "sequence": 64, "seed": 0}
_SEQUENCE_CHOICES = (64, 128, 256)
_RANGES = {"steps": (1, 500), "batch": (1, 4), "seed": (0, 9999)}
# Fixed architecture, reported for comparability; never caller-configurable.
ARCHITECTURE = {"dim": 512, "layers": 8, "heads": 8, "vocab": 4096, "params_m": 37.7}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_module_from_file(path: Path, name: str):
    """Import a Python file in-process and return the module."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LoadError("cannot build an import spec for " + str(path))
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__];
    # an unregistered module makes the file fail at import time.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_experiment_module(path: Path = EXPERIMENT_FILE):
    """Import the pinned qlabs experiment file and validate its port surface."""
    try:
        resolved = path.resolve()
        resolved.relative_to(PROJECT_DIR.resolve())
    except ValueError as exc:
        raise LoadError("qlabs experiment file must live under the pinned project " + str(PROJECT_DIR)) from exc
    if not resolved.is_file():
        raise LoadError("qlabs experiment file is missing: " + str(resolved))
    module = import_module_from_file(resolved, _MODULE_NAME)
    if not callable(getattr(module, "run_experiment", None)):
        sys.modules.pop(_MODULE_NAME, None)
        raise LoadError("qlabs experiment file does not define a callable run_experiment")
    return module


def validate_params(params: dict) -> dict:
    """Reduce caller params to the fixed, bounded run surface.

    The worker protocol injects ``_job_type``; every other key outside the
    allowlist is rejected so no arbitrary experiment knob leaks through.
    """
    unknown = sorted(set(params) - _ALLOWLISTED_PARAMS - {"_job_type"})
    if unknown:
        raise InferenceError("unsupported qlabs-noise-train params: " + ", ".join(unknown))
    cleaned = dict(_DEFAULTS)
    for key in ("steps", "batch", "sequence", "seed"):
        if key not in params or params[key] is None:
            continue
        try:
            value = int(params[key])
        except (TypeError, ValueError):
            raise InferenceError(f"{key} must be an integer")
        if key == "sequence":
            if value not in _SEQUENCE_CHOICES:
                raise InferenceError(
                    "sequence must be one of 64, 128, 256 (kernel grids require multiples of 64)"
                )
        else:
            low, high = _RANGES[key]
            if not low <= value <= high:
                raise InferenceError(f"{key} must be between {low} and {high}")
        cleaned[key] = value
    if "force" in params and params["force"] is not None and not isinstance(params["force"], bool):
        raise InferenceError("force must be a boolean")
    return cleaned


@register
class QlabsNoiseTrainAdapter(ModelAdapter):
    """Run the pinned qlabs noise-training experiment under the worker CUDA cap."""

    model_id = "qlabs-noise-train"

    def __init__(self) -> None:
        self._module = None
        self._experiment_sha256 = ""

    def load(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise LoadError("qlabs-noise-train requires the Arbiter CUDA worker")
        self._module = load_experiment_module()
        self._experiment_sha256 = file_sha256(EXPERIMENT_FILE)
        log.info(
            "qlabs-noise-train experiment %s sha256=%s",
            EXPERIMENT_FILE,
            self._experiment_sha256[:16],
        )

    def unload(self) -> None:
        self._module = None
        sys.modules.pop(_MODULE_NAME, None)
        self._cleanup_gpu()

    def estimate_time(self, params: dict) -> float:
        return 3_600_000.0

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        if self._module is None:
            raise InferenceError("qlabs-noise-train is not loaded")
        self._check_cancel(cancel_flag)
        fixed = validate_params(params)
        log.info(
            "qlabs-noise-train steps=%(steps)d batch=%(batch)d sequence=%(sequence)d seed=%(seed)d",
            fixed,
        )
        started = time.perf_counter()
        metadata = self._module.run_experiment(fixed, output_dir, cancel_flag)
        self._check_cancel(cancel_flag)
        if not isinstance(metadata, dict):
            raise InferenceError("qlabs experiment did not return a JSON metadata object")
        report = output_dir / "report.json"
        if not report.is_file():
            raise InferenceError("qlabs experiment did not write report.json to the output dir")
        result = dict(metadata)
        result.setdefault("format", "json")
        result.setdefault("file", "report.json")
        result["params"] = fixed
        result["architecture"] = ARCHITECTURE
        result["experiment_sha256"] = self._experiment_sha256
        result["experiment_file"] = str(EXPERIMENT_FILE)
        # Wall time around the whole experiment call (compile + eval included),
        # named so it cannot shadow the experiment's own train-only timings.
        result["adapter_wall_seconds"] = time.perf_counter() - started
        return result
