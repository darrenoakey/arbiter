"""Shared test fixtures for Arbiter tests."""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_config_dict():
    """Minimal valid config dict for unit tests."""
    return {
        "vram_budget_gb": 24,
        "host": "127.0.0.1",
        "port": 18400,
        "default_keep_alive_seconds": 60,
        "models": {
            "model-a": {
                "memory_gb": 4.0,
                "max_concurrent": 2,
                "keep_alive_seconds": 60,
                "avg_inference_ms": 500,
                "load_ms": 2000,
            },
            "model-b": {
                "memory_gb": 8.0,
                "max_concurrent": 1,
                "keep_alive_seconds": 30,
                "avg_inference_ms": 2000,
                "load_ms": 5000,
            },
            "model-c": {
                "memory_gb": 1.0,
                "max_concurrent": 4,
                "keep_alive_seconds": 120,
                "avg_inference_ms": 100,
                "load_ms": 500,
            },
        },
    }


@pytest.fixture
def sample_config(sample_config_dict):
    """ArbiterConfig instance from sample dict."""
    from arbiter.config import ArbiterConfig
    return ArbiterConfig(**sample_config_dict)


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def store(tmp_db):
    """JobStore instance with temp database."""
    from arbiter.store import JobStore
    s = JobStore(tmp_db)
    yield s
    s.close()


@pytest.fixture
def mock_adapter():
    """Mock adapter that simulates load/unload/infer."""
    adapter = MagicMock()
    adapter.model_id = "mock-model"
    adapter.load = MagicMock()  # sync
    adapter.unload = MagicMock()  # sync
    adapter.infer = MagicMock(return_value={"format": "json"})
    adapter.estimate_time = MagicMock(return_value=1000.0)
    return adapter


@pytest.fixture
def mock_logger():
    """Mock event logger."""
    logger = MagicMock()
    logger.log = MagicMock()
    return logger
