"""VRAM measurement utilities."""
from __future__ import annotations


def get_gpu_info() -> dict:
    """Get GPU hardware info."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpu": "none", "cuda_available": False}
        return {
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1),
            "cuda_version": torch.version.cuda or "unknown",
        }
    except Exception as e:
        return {"gpu": "unknown", "error": str(e)}


def measure_vram() -> dict:
    """Measure current VRAM usage."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return {
            "allocated_gb": round(allocated, 3),
            "reserved_gb": round(reserved, 3),
            "free_gb": round(total - allocated, 3),
            "total_gb": round(total, 1),
        }
    except Exception:
        return {"allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}
