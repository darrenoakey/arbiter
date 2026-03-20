"""Abstract base classes for model adapters."""
from __future__ import annotations

import gc
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Optional


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
