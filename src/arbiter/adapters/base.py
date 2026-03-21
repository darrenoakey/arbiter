"""Abstract base classes for model adapters."""
from __future__ import annotations

import base64
import gc
import io
import logging
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Optional

_base_log = logging.getLogger(__name__)


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
    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
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
    def _resolve_media(params: dict, key: str = "image", file_key: str | None = None) -> bytes:
        """Resolve media bytes from either a file path or base64 data.

        Checks for a file path first (key + "_file" or explicit file_key),
        then falls back to base64 decoding of the standard key.
        This is the single place all adapters should get their media from.
        """
        if file_key is None:
            file_key = f"{key}_file"
        file_path = params.get(file_key)
        if file_path:
            p = Path(file_path)
            if p.is_file():
                _base_log.debug("Reading %s from file: %s", key, p)
                return p.read_bytes()
            _base_log.warning("%s file not found, falling back to base64: %s", key, p)
        # Fall back to base64
        b64_data = params.get(key) or params.get(f"{key}_url", "")
        if not b64_data:
            from arbiter.adapters.base import InferenceError
            raise InferenceError(f"No {key} or {file_key} provided")
        if b64_data.startswith("data:"):
            _, b64_data = b64_data.split(",", 1)
        return base64.b64decode(b64_data)

    @staticmethod
    def _resolve_image(params: dict) -> "Image.Image":
        """Convenience: resolve image media and return a PIL Image."""
        from PIL import Image
        from arbiter.adapters.base import ModelAdapter, InferenceError
        try:
            raw = ModelAdapter._resolve_media(params, "image")
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            if isinstance(e, InferenceError):
                raise
            raise InferenceError(f"Failed to decode image: {e}")

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
