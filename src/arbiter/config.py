"""Configuration loader for Arbiter."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# Resolve paths relative to project root (two levels up from this file: src/arbiter/config.py -> arbiter/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ModelConfig(BaseModel):
    memory_gb: float
    max_concurrent: int = 1
    keep_alive_seconds: int = 300
    avg_inference_ms: float = 5000
    load_ms: float = 10000
    auto_download: Optional[str] = None
    model_path: Optional[str] = None
    group: bool = False


class ArbiterConfig(BaseModel):
    vram_budget_gb: float = 100
    host: str = "0.0.0.0"
    port: int = 8400
    default_keep_alive_seconds: int = 300
    models: dict[str, ModelConfig] = Field(default_factory=dict)


def load_config(project_root: Path | None = None) -> ArbiterConfig:
    """Load config from local/config.json, falling back to local/config.default.json."""
    root = project_root or _PROJECT_ROOT
    config_path = root / "local" / "config.json"
    default_path = root / "local" / "config.default.json"

    if config_path.exists():
        path = config_path
    elif default_path.exists():
        path = default_path
    else:
        return ArbiterConfig()

    with open(path) as f:
        data = json.load(f)

    # Environment variable overrides
    if env_budget := os.environ.get("ARBITER_VRAM_BUDGET_GB"):
        data["vram_budget_gb"] = float(env_budget)
    if env_port := os.environ.get("ARBITER_PORT"):
        data["port"] = int(env_port)
    if env_host := os.environ.get("ARBITER_HOST"):
        data["host"] = env_host

    return ArbiterConfig(**data)
