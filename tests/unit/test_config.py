"""Tests for config loading and validation."""
import json
import os
from pathlib import Path

import pytest

from arbiter.config import ArbiterConfig, ModelConfig, load_config


class TestModelConfig:
    def test_defaults(self):
        mc = ModelConfig(memory_gb=4.0)
        assert mc.max_concurrent == 1
        assert mc.keep_alive_seconds == 300
        assert mc.avg_inference_ms == 5000
        assert mc.load_ms == 10000
        assert mc.auto_download is None
        assert mc.group is False

    def test_all_fields(self):
        mc = ModelConfig(
            memory_gb=12, max_concurrent=2, keep_alive_seconds=600,
            avg_inference_ms=2000, load_ms=15000,
            auto_download="org/model", model_path="/tmp/m", group=True,
        )
        assert mc.memory_gb == 12
        assert mc.group is True


class TestArbiterConfig:
    def test_defaults(self):
        cfg = ArbiterConfig()
        assert cfg.vram_budget_gb == 100
        assert cfg.port == 8400
        assert cfg.models == {}

    def test_from_dict(self, sample_config_dict):
        cfg = ArbiterConfig(**sample_config_dict)
        assert cfg.vram_budget_gb == 24
        assert len(cfg.models) == 3
        assert cfg.models["model-a"].memory_gb == 4.0
        assert cfg.models["model-a"].max_concurrent == 2

    def test_unknown_fields_ignored(self):
        cfg = ArbiterConfig(vram_budget_gb=50, unknown_field="x")
        assert cfg.vram_budget_gb == 50


class TestLoadConfig:
    def test_load_from_default(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        default = local / "config.default.json"
        default.write_text(json.dumps({"vram_budget_gb": 80, "models": {}}))
        cfg = load_config(tmp_path)
        assert cfg.vram_budget_gb == 80

    def test_config_overrides_default(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        (local / "config.default.json").write_text(json.dumps({"vram_budget_gb": 80, "models": {}}))
        (local / "config.json").write_text(json.dumps({"vram_budget_gb": 90, "models": {}}))
        cfg = load_config(tmp_path)
        assert cfg.vram_budget_gb == 90

    def test_env_override(self, tmp_path, monkeypatch):
        local = tmp_path / "local"
        local.mkdir()
        (local / "config.default.json").write_text(json.dumps({"vram_budget_gb": 80, "models": {}}))
        monkeypatch.setenv("ARBITER_VRAM_BUDGET_GB", "50")
        cfg = load_config(tmp_path)
        assert cfg.vram_budget_gb == 50

    def test_no_config_returns_default(self, tmp_path):
        cfg = load_config(tmp_path)
        assert cfg.vram_budget_gb == 100
