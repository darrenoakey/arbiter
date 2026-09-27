"""Registration tests for the routing-decide trainer. No model is loaded."""

from __future__ import annotations

import json
from pathlib import Path

from arbiter.adapters.routing_decide import RoutingDecideAdapter, _percentile
from arbiter.schemas import JOB_TYPE_PARAMS, JOB_TYPE_TO_MODEL, RoutingDecideTrainParams


def test_routing_decide_job_type_is_registered():
    assert JOB_TYPE_TO_MODEL["routing-decide-train"] == "routing-decide"
    params = RoutingDecideTrainParams(dataset_file="/mnt/arbiter-store/inbox/example.json")
    assert params.max_len == 256
    assert params.num_epochs == 1
    assert params.batch_size == 4
    assert JOB_TYPE_PARAMS["routing-decide-train"] is RoutingDecideTrainParams


def test_routing_decide_latency_percentiles_use_linear_interpolation():
    assert _percentile([0.1, 0.4, 0.2, 0.3], 0.5) == 0.25
    assert _percentile([0.1, 0.4, 0.2, 0.3], 0.95) == 0.385


def test_routing_decide_adapter_does_not_load_a_model():
    adapter = RoutingDecideAdapter()
    assert adapter.model_id == "routing-decide"
    assert adapter.estimate_time({}) == 3_600_000.0


def test_routing_decide_spark_config_uses_the_sanctioned_venv():
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "config/spark/routing-decide.model.json").read_text())
    assert config["memory_gb"] == 24
    assert config["max_runtime_seconds"] == 14400
    assert config["worker_cmd"] == [
        "/home/darren/src/arbiter/venvs/routing-decide/bin/python",
        "-m",
        "arbiter.worker_main",
        "routing-decide",
    ]
