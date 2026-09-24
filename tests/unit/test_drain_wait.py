"""Fail-closed tests for the deploy drain protocol parser."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "drain_wait.py"
DEPLOY_SCRIPT = Path(__file__).resolve().parents[2] / "deploy-to-spark.sh"

spec = importlib.util.spec_from_file_location("drain_wait", SCRIPT)
assert spec is not None and spec.loader is not None
drain_wait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drain_wait)


def recorded_ps(*, draining: bool = True, active: int = 0) -> dict:
    """Return the production /v1/ps shape recorded from Spark."""
    return {
        "vram_budget_gb": 100.0,
        "vram_used_gb": 37.0,
        "draining": draining,
        "active_jobs": active,
        "models": [
            {
                "id": "ltx25-denoise1",
                "state": "active" if active else "loaded",
                "memory_gb": 36.0,
                "active_jobs": active,
                "queued_jobs": 2,
                "instances": [
                    {
                        "instance_id": "ltx25-denoise1-0",
                        "state": "active" if active else "loaded",
                        "active_jobs": active,
                        "host": "spark",
                    }
                ],
            },
            {
                "id": "local-titler",
                "state": "loaded",
                "memory_gb": 1.0,
                "active_jobs": 0,
                "queued_jobs": 0,
                "instances": [],
            },
        ],
        "queue": {
            "queued": 2,
            "scheduled": active,
            "completed": 57,
            "failed": 2,
            "cancelled": 0,
        },
    }


def run_parser(kind: str, payload: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), kind],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )


def run_recovery_abort(kind: str, detail: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(DEPLOY_SCRIPT), "--test-recovery-abort", kind, detail],
        capture_output=True,
        text=True,
        check=False,
    )


def test_recorded_ps_proves_drained_with_queued_work_preserved():
    snapshot = recorded_ps(active=0)
    assert snapshot["queue"]["queued"] == 2
    assert drain_wait.ps_state(snapshot) == (True, 0)


def test_recorded_ps_reports_active_work_without_calling_it_drained():
    # Read-only GET /v1/ps on Spark, 2026-09-25. The queue aggregate is DB
    # cached and can differ from the in-memory active count; deployment safety
    # therefore uses root active_jobs cross-checked against models, not queue.
    live = {
        "draining": False,
        "active_jobs": 3,
        "models": [
            {"id": "llm:qwen3-vl-8b-fp8", "active_jobs": 2},
            {"id": "moondream", "active_jobs": 1},
        ],
        "queue": {"cancelled": 61, "completed": 27623, "failed": 171, "running": 2},
    }
    assert drain_wait.ps_state(live) == (False, 3)


def test_wait_decision_never_proceeds_with_active_jobs_at_deadline():
    active = recorded_ps(active=1)
    assert drain_wait.wait_decision(active, deadline_reached=False) == ("wait", 1)
    assert drain_wait.wait_decision(active, deadline_reached=True) == ("abort", 1)
    assert drain_wait.wait_decision(recorded_ps(active=0), deadline_reached=True) == (
        "drained",
        0,
    )


def test_wait_decision_rejects_a_lost_drain_lease():
    with pytest.raises(drain_wait.DrainProtocolError, match="draining=true"):
        drain_wait.wait_decision(recorded_ps(draining=False), deadline_reached=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("active_jobs"), "active_jobs is missing"),
        (lambda value: value.__setitem__("active_jobs", False), "non-negative integer"),
        (lambda value: value.__setitem__("active_jobs", -1), "non-negative integer"),
        (lambda value: value.pop("models"), "models must be an array"),
        (lambda value: value.__setitem__("models", {}), "models must be an array"),
        (lambda value: value["models"][0].pop("active_jobs"), "active_jobs is missing"),
        (lambda value: value["models"][0].__setitem__("active_jobs", 1), "does not equal"),
        (lambda value: value.pop("queue"), "queue must be an object"),
        (lambda value: value["queue"].__setitem__("queued", -1), "non-negative integer"),
        (lambda value: value.pop("draining"), "draining must be a boolean"),
    ],
)
def test_ps_rejects_missing_malformed_or_inconsistent_counts(mutation, message):
    snapshot = recorded_ps(active=0)
    mutation(snapshot)
    with pytest.raises(drain_wait.DrainProtocolError, match=message):
        drain_wait.ps_state(snapshot)


def test_drain_and_resume_acknowledgements_are_explicit():
    drain_wait.drain_response(
        {"draining": True, "active_jobs": 0, "lease_seconds": 300},
        resumed=False,
        expected_lease=300,
    )
    drain_wait.drain_response({"draining": False}, resumed=True)


@pytest.mark.parametrize(
    ("payload", "resumed"),
    [
        ({"draining": True, "active_jobs": 0}, False),
        ({"draining": True, "active_jobs": "0", "lease_seconds": 300}, False),
        ({"draining": False, "active_jobs": 0, "lease_seconds": 300}, False),
        ({"draining": True, "active_jobs": 0, "lease_seconds": 1}, False),
        ({"draining": True}, True),
    ],
)
def test_drain_acknowledgement_rejects_unproved_state(payload, resumed):
    with pytest.raises(drain_wait.DrainProtocolError):
        drain_wait.drain_response(payload, resumed=resumed, expected_lease=300)


@pytest.mark.parametrize("payload", ("", "not-json", "[]", "{}"))
def test_cli_fails_closed_for_empty_malformed_or_incomplete_ps(payload):
    result = run_parser("ps", payload)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "invalid Arbiter ps response" in result.stderr


def test_cli_emits_only_validated_ps_state():
    result = run_parser("ps", json.dumps(recorded_ps(draining=False, active=0)))
    assert result.returncode == 0
    assert result.stdout == "0\t0\n"


def test_closed_port_curl_failure_cannot_feed_a_successful_parse():
    curl = subprocess.run(
        ["curl", "-fsS", "--max-time", "1", "http://127.0.0.1:1/v1/ps"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert curl.returncode != 0
    parsed = run_parser("ps", curl.stdout)
    assert parsed.returncode == 2


def test_replacement_listener_aborts_without_recovery_termination():
    result = run_recovery_abort("replacement-listener", "48123")
    assert result.returncode != 0
    assert "pid 48123" in result.stderr
    assert "aborted without stopping or signaling it" in result.stderr


def test_already_running_start_aborts_without_recovery_termination():
    result = run_recovery_abort("already-running")
    assert result.returncode != 0
    assert "already running" in result.stderr
    assert "aborted without stopping or signaling the replacement" in result.stderr


def test_deploy_has_only_the_drained_stop_and_normal_start():
    script = DEPLOY_SCRIPT.read_text()
    assert script.count('auto/run stop arbiter') == 1
    assert script.count('auto/run start arbiter') == 1
    assert "auto/run restart arbiter" not in script
    assert "kill -TERM" not in script
    assert "pkill" not in script
