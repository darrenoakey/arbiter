"""Model adapter registry."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ModelAdapter

_REGISTRY: dict[str, type[ModelAdapter]] = {}


def register(adapter_cls: type[ModelAdapter]) -> type[ModelAdapter]:
    """Decorator to register an adapter class by its model_id."""
    model_id = adapter_cls.model_id
    if not model_id:
        raise ValueError(f"Adapter {adapter_cls.__name__} has no model_id")
    _REGISTRY[model_id] = adapter_cls
    return adapter_cls


def get_adapter_class(model_id: str) -> type[ModelAdapter]:
    """Get adapter class by model_id. Raises KeyError if not found."""
    if model_id not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown model: {model_id}. Available: {available}")
    return _REGISTRY[model_id]


def list_registered() -> list[str]:
    """List all registered model IDs."""
    return sorted(_REGISTRY.keys())
