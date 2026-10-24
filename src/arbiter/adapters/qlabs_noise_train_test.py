"""Registration and validation tests for the qlabs-noise-train adapter.

No model is loaded and no GPU is touched: these tests cover the schema
surface, the bounded param validation, the pinned project import path, and
the sanctioned spark worker config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arbiter.adapters.base import LoadError
from arbiter.adapters.qlabs_noise_train import (
    ARCHITECTURE,
    EXPERIMENT_FILE,
    PROJECT_DIR,
    QlabsNoiseTrainAdapter,
    import_module_from_file,
    validate_params,
)
from arbiter.schemas import (
    JOB_TYPE_PARAMS,
    JOB_TYPE_TO_MODEL,
    JobType,
    QlabsNoiseTrainParams,
)


def test_qlabs_noise_train_job_type_is_registered():
    assert JobType.QLABS_NOISE_TRAIN == "qlabs-noise-train"
    assert JOB_TYPE_TO_MODEL["qlabs-noise-train"] == "qlabs-noise-train"
    assert JOB_TYPE_PARAMS["qlabs-noise-train"] is QlabsNoiseTrainParams


def test_qlabs_noise_train_params_defaults_and_bounds():
    params = QlabsNoiseTrainParams()
    assert params.steps == 100
    assert params.batch == 2
    assert params.sequence == 64
    assert params.seed == 0
    full = QlabsNoiseTrainParams(steps=500, batch=4, sequence=256, seed=9999)
    assert full.sequence == 256
    with pytest.raises(Exception):
        QlabsNoiseTrainParams(steps=0)
    with pytest.raises(Exception):
        QlabsNoiseTrainParams(steps=501)
    with pytest.raises(Exception):
        QlabsNoiseTrainParams(batch=5)
    with pytest.raises(Exception):
        QlabsNoiseTrainParams(sequence=32)
    with pytest.raises(Exception):
        QlabsNoiseTrainParams(sequence=96)
    with pytest.raises(Exception):
        QlabsNoiseTrainParams(seed=10000)


def test_qlabs_noise_train_validate_params_only_fixed_surface():
    assert validate_params({}) == {"steps": 100, "batch": 2, "sequence": 64, "seed": 0}
    assert validate_params({"_job_type": "qlabs-noise-train"}) == {
        "steps": 100,
        "batch": 2,
        "sequence": 64,
        "seed": 0,
    }
    assert validate_params({"steps": 2, "batch": 1, "sequence": 64}) == {
        "steps": 2,
        "batch": 1,
        "sequence": 64,
        "seed": 0,
    }
    # the Arbiter dedup-protocol flag is accepted as a bool and ignored
    assert validate_params({"force": True}) == {
        "steps": 100,
        "batch": 2,
        "sequence": 64,
        "seed": 0,
    }
    assert validate_params({"force": False})["steps"] == 100
    with pytest.raises(Exception, match="force"):
        validate_params({"force": "yes"})
    with pytest.raises(Exception, match="unsupported"):
        validate_params({"lr": 0.001})
    with pytest.raises(Exception, match="unsupported"):
        validate_params({"architecture": "D1024"})
    with pytest.raises(Exception, match="sequence"):
        validate_params({"sequence": 100})
    with pytest.raises(Exception, match="steps"):
        validate_params({"steps": 0})
    with pytest.raises(Exception, match="seed"):
        validate_params({"seed": 10000})


def test_qlabs_noise_train_imports_experiment_file_in_process(tmp_path):
    experiment = tmp_path / "experiment.py"
    experiment.write_text(
        "def run_experiment(params, output_dir, cancel_flag):\n"
        "    return {'ok': True, 'params': params}\n"
    )
    module = import_module_from_file(experiment, "arbiter_qlabs_noise_test_experiment")
    assert callable(module.run_experiment)
    result = module.run_experiment({"steps": 2}, tmp_path, None)
    assert result == {"ok": True, "params": {"steps": 2}}


def test_qlabs_noise_train_load_requires_the_pinned_project():
    # The pinned spark path does not exist on a dev laptop; the load must
    # fail closed with LoadError rather than import anything else.
    assert EXPERIMENT_FILE == PROJECT_DIR / "src" / "experiment.py"
    adapter = QlabsNoiseTrainAdapter()
    with pytest.raises(LoadError):
        adapter.load()


def test_qlabs_noise_train_adapter_identity():
    adapter = QlabsNoiseTrainAdapter()
    assert adapter.model_id == "qlabs-noise-train"
    assert adapter.estimate_time({}) == 3_600_000.0
    assert ARCHITECTURE == {"dim": 512, "layers": 8, "heads": 8, "vocab": 4096, "params_m": 37.7}


def test_qlabs_noise_train_spark_config_uses_the_sanctioned_venv():
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "config/spark/qlabs-noise-train.model.json").read_text())
    assert config["memory_gb"] == 8
    assert config["max_concurrent"] == 1
    assert config["max_instances"] == 1
    assert config["worker_cmd"] == [
        "/home/darren/src/arbiter/venvs/qlabsnoise/bin/python",
        "-m",
        "arbiter.worker_main",
        "qlabs-noise-train",
    ]
