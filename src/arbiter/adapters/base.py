"""Abstract base classes for model adapters."""

from __future__ import annotations

import base64
import ctypes
import gc
import io
import logging
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

_base_log = logging.getLogger(__name__)


class HeapTrimGuard:
    """Return freed glibc heap pages while a unified-memory model loads."""

    def __init__(self, interval_seconds: float = 0.25):
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.trim_count = 0

    def __enter__(self) -> "HeapTrimGuard":
        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if trim is None:
            return self
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int

        def trim_loop() -> None:
            while not self._stop.wait(self._interval_seconds):
                trim(0)
                self.trim_count += 1

        self._thread = threading.Thread(
            target=trim_loop,
            name="arbiter-heap-trim",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_seconds * 4))
        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if trim is not None:
            trim(0)
            self.trim_count += 1


class ModelState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    EVICTING = "evicting"
    ERROR = "error"


class AdapterError(Exception):
    """Base error for adapter failures."""

    pass


class LoadError(AdapterError):
    """Model failed to load."""

    pass


class InferenceError(AdapterError):
    """Inference failed."""

    pass


class CancelledException(AdapterError):
    """Inference was cancelled."""

    pass


class ModelAdapter(ABC):
    """Abstract base for a single model adapter.

    Subclasses must set model_id as a class attribute and implement
    load(), unload(), infer(), and estimate_time().
    """

    model_id: str = ""

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        """Load model weights onto the device. Called from a worker thread."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release all GPU memory. Called from a worker thread."""
        ...

    @abstractmethod
    def infer(
        self, params: dict, output_dir: Path, cancel_flag: threading.Event
    ) -> dict:
        """Run inference.

        Args:
            params: Validated job parameters.
            output_dir: Directory to write result files (e.g., output/jobs/{job_id}/).
            cancel_flag: Check cancel_flag.is_set() at natural breakpoints.
                         Raise CancelledException if set.

        Returns:
            Result metadata dict (e.g., {"format": "png", "width": 1024, "height": 1024}).
        """
        ...

    @abstractmethod
    def estimate_time(self, params: dict) -> float:
        """Estimate inference time in milliseconds for SJF scoring."""
        ...

    def _check_cancel(self, cancel_flag: threading.Event):
        """Helper: raise CancelledException if flag is set."""
        if cancel_flag.is_set():
            raise CancelledException(f"Job cancelled for model {self.model_id}")

    @staticmethod
    def _resolve_media(
        params: dict, key: str = "image", file_key: str | None = None
    ) -> bytes:
        """Resolve media bytes from either a file path or base64 data.

        Checks for a file path first (key + "_file" or explicit file_key),
        then falls back to base64 decoding of the standard key.
        This is the single place all adapters should get their media from.
        """
        if file_key is None:
            file_key = f"{key}_file"
        file_path = params.get(file_key)
        if file_path and file_path.startswith("ref:"):
            file_path = str(Path("output/refs") / file_path[4:])
        if file_path:
            p = Path(file_path)
            # Reject DEFINITIVELY-bad inputs instantly — never burn the 10s
            # CIFS-lag poll (or a model load + inference) on a file that can
            # never be valid media. A macOS AppleDouble resource fork (._*) or
            # a 0-byte file is junk that will fail no matter what; fail fast
            # here as a client error (see isClientError in the Go scheduler) so
            # the worker stays free and the model's circuit breaker is spared.
            if p.name.startswith("._"):
                raise InferenceError(
                    f"bad input file (macOS resource fork '._', not real media): {p}"
                )
            try:
                if p.is_file() and p.stat().st_size == 0:
                    raise InferenceError(f"bad input file (empty, 0 bytes): {p}")
            except OSError:
                pass
            # The shared inbox is a CIFS mount; the client writes the file on
            # the Mac side then immediately submits the job, but spark's CIFS
            # attribute cache can lag briefly. Poll for up to 10s before
            # giving up — dropping to base64 input (which isn't sent by
            # most clients) was previously swallowing every transient miss
            # as "No image or image_file provided".
            deadline = time.monotonic() + 10.0
            while not p.is_file() and time.monotonic() < deadline:
                threading.Event().wait(0.2)
            if p.is_file():
                _base_log.debug("Reading %s from file: %s", key, p)
                return p.read_bytes()
            _base_log.warning(
                "%s file not found after wait, falling back to base64: %s", key, p
            )
        # Fall back to base64
        b64_data = params.get(key) or params.get(f"{key}_url", "")
        if not b64_data:
            raise InferenceError(f"No {key} or {file_key} provided")
        if b64_data.startswith("data:"):
            _, b64_data = b64_data.split(",", 1)
        return base64.b64decode(b64_data)

    @staticmethod
    def _resolve_image(params: dict) -> "Image.Image":
        """Convenience: resolve image media and return a PIL Image."""
        from PIL import Image

        try:
            raw = ModelAdapter._resolve_media(params, "image")
            img = Image.open(io.BytesIO(raw))
            img.load()  # force decode now so a corrupt/truncated file fails here
            return img.convert("RGB")
        except Exception as e:
            if isinstance(e, InferenceError):
                raise
            # Present-but-undecodable (corrupt, truncated, or unsupported format
            # like HEIC without the plugin) — a client/input problem, not a model
            # fault. Phrase it so isClientError classifies it as such.
            raise InferenceError(
                f"bad input image (cannot decode — corrupt or unsupported format): {e}"
            )

    @staticmethod
    def _cleanup_gpu():
        """Free GPU caches. Call after unload or failed inference."""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    @staticmethod
    def _pipe_to_cuda_cloned(pipe, device: str = "cuda") -> None:
        """Move a diffusers Pipeline to `device`, bypassing the GB10
        mmap→cuda pathological slow path.

        Background: on GB10 with torch 2.10.0+cu130 (cuda cap 12.1), moving
        a safetensors-mmap-backed CPU tensor directly via ``.to("cuda")``
        triggers per-4KB-page fault-in during cudaMemcpy, collapsing host→device
        bandwidth to ~170 MB/s. Cloning the tensor first (which does a fast
        sequential memcpy out of mmap into a heap-allocated tensor) lets the
        subsequent ``.to("cuda")`` run at normal DRAM bandwidth (~8-16 GB/s).

        This function walks every nn.Module attribute of the pipe, clones
        each parameter and buffer individually, then moves it to the target
        device — so peak extra memory is the size of one tensor, not the
        whole state dict. Measured speedup on 2026-04-15: z-image-turbo
        46s→4s (10.9x), flux-schnell 184s→9s (21.1x).

        Reference: performance.md (LTX-2 Encode Load Performance Investigation)
        where this same mmap→cuda bug was diagnosed for the ltx2 pipeline
        and fixed inside ltx-core's sft_loader.py with .clone().
        """
        import torch
        import torch.nn as nn

        seen = set()
        for attr in dir(pipe):
            if attr.startswith("_"):
                continue
            try:
                obj = getattr(pipe, attr, None)
            except Exception:
                continue
            if not isinstance(obj, nn.Module):
                continue
            if id(obj) in seen:
                continue
            seen.add(id(obj))

            for p in obj.parameters(recurse=True):
                if p.data.device.type != device:
                    p.data = p.data.clone().to(device, non_blocking=False)

            # Buffers must be replaced via setattr on their parent module
            for name, b in list(obj.named_buffers(recurse=True)):
                if b.device.type != device:
                    new_b = b.clone().to(device, non_blocking=False)
                    parts = name.split(".")
                    mod = obj
                    for pn in parts[:-1]:
                        mod = getattr(mod, pn)
                    setattr(mod, parts[-1], new_b)

        # Sanity call — after the per-tensor move above this is a no-op,
        # but it lets the pipeline update any internal device-tracking state
        # that diffusers relies on.
        try:
            pipe.to(device)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class GroupAdapter(ModelAdapter):
    """Adapter for co-dependent sub-models loaded/unloaded atomically.

    Subclasses should load all sub-models in load() and unload all in unload().
    If load() fails partway through, it must clean up already-loaded sub-models.
    """

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        """Load ALL sub-models atomically."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload ALL sub-models, gc.collect(), empty_cache()."""
        ...
