"""Tests for the mini-AGI chunk adapter. No model is loaded and no GPU is used."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from arbiter.adapters import mini_agi
from arbiter.adapters.base import InferenceError
from arbiter.adapters.mini_agi import MiniAgiAdapter, _StopMinutes, chunk_minutes, progress, run_dir
from arbiter.schemas import JOB_TYPE_PARAMS, JOB_TYPE_TO_MODEL, MiniAgiReadParams


def test_mini_agi_job_type_is_registered():
    assert JOB_TYPE_TO_MODEL["mini-agi-read"] == "mini-agi"
    assert JOB_TYPE_PARAMS["mini-agi-read"] is MiniAgiReadParams
    params = MiniAgiReadParams()
    assert params.run == "mnemnos" and params.minutes == 26.0 and params.samples is True
    with pytest.raises(ValueError):
        MiniAgiReadParams(minutes=31)
    with pytest.raises(ValueError):
        MiniAgiReadParams(run="../etc")


def test_run_names_cannot_escape_the_training_root():
    assert run_dir("mnemnos") == mini_agi.TRAINING_ROOT / "mnemnos"
    for bad in ("../x", "/abs", "UPPER", "", "a/b", ".hidden"):
        with pytest.raises(InferenceError):
            run_dir(bad)


def test_chunks_are_capped_at_half_an_hour():
    assert chunk_minutes({}) == 26.0
    assert chunk_minutes({"minutes": 30}) == 30.0
    for bad in (0, -1, 30.5, "soon"):
        with pytest.raises(InferenceError):
            chunk_minutes({"minutes": bad})
    adapter = MiniAgiAdapter()
    assert adapter.estimate_time({"minutes": 10}) == 10 * 60_000 + 120_000


def test_stop_minutes_ends_the_loop_on_time_or_cancel():
    cancel = threading.Event()
    budget = _StopMinutes(26.0, cancel)
    # train.py's exact test: (elapsed seconds / 60) >= args.minutes
    assert not (5.0 >= budget)
    assert 26.0 >= budget
    assert budget  # train.py also checks `args.minutes and ...`
    cancel.set()
    assert 0.5 >= budget


def _run(tmp_path: Path, read_chars: int) -> Path:
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "manifest.json").write_text(json.dumps({"read_chars": read_chars}))
    lane = tmp_path / "data" / "lane"
    lane.mkdir(parents=True)
    (lane / "a.txt").write_text("x" * 600)
    (lane / "b.txt").write_text("y" * 400)
    (lane / "skip.bin").write_bytes(b"\0" * 999)
    return tmp_path


def test_progress_counts_the_pass_from_where_it_started(tmp_path):
    root = _run(tmp_path, 5_000)
    first = progress(root)
    assert first == {
        "read_chars": 5_000,
        "pass_read_chars": 0,
        "pass_corpus_chars": 1_000,
        "pass_fraction": 0.0,
        "done": False,
    }
    (root / "weights" / "manifest.json").write_text(json.dumps({"read_chars": 5_400}))
    assert progress(root)["pass_fraction"] == 0.4
    (root / "weights" / "manifest.json").write_text(json.dumps({"read_chars": 6_000}))
    assert progress(root)["done"] is True


def test_spark_config_declares_a_bounded_chunk():
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "config/spark/mini-agi.model.json").read_text())
    assert config["max_runtime_seconds"] <= 2100
    assert config["max_instances"] == 1 and config["max_concurrent"] == 1
    assert "worker_cmd" not in config  # the main venv; torch is all it needs
