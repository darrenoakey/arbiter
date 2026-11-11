from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("deploy_readiness", ROOT / "scripts/deploy_readiness.py")
assert SPEC and SPEC.loader
readiness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readiness)


def snapshot(*, active: int = 0, queued: int = 0, draining: bool = False) -> dict:
    return {
        "draining": draining,
        "active_jobs": active,
        "models": [{"active_jobs": active, "queued_jobs": queued}],
        "queue": {"queued": queued, "scheduled": 0, "running": active,
                  "completed": 200, "failed": 3},
    }


def test_config_default_valid_and_rejects_invalid(tmp_path: Path):
    config = tmp_path / "config.toml"
    assert readiness.configured_wait(config) == 0
    config.write_text("[deploy]\npre_drain_wait_seconds = 180\n")
    assert readiness.configured_wait(config) == 180
    for raw in ("true", "-1", "181", '"2"', "1.5"):
        config.write_text(f"[deploy]\npre_drain_wait_seconds = {raw}\n")
        with pytest.raises(ValueError):
            readiness.configured_wait(config)
    config.write_text("[deploy\n")
    with pytest.raises(ValueError):
        readiness.configured_wait(config)


def test_ready_requires_undrained_coherent_strict_zero():
    assert readiness.ready(snapshot())
    assert readiness.ready(snapshot(queued=700))  # Backlog is not in-flight work.
    assert not readiness.ready(snapshot(active=1, queued=0))
    with pytest.raises(readiness.DrainProtocolError, match="ownership is external"):
        readiness.ready(snapshot(draining=True))
    invalid = snapshot()
    invalid["models"][0]["active_jobs"] = 1
    with pytest.raises(readiness.DrainProtocolError, match="does not equal"):
        readiness.ready(invalid)


def test_deploy_calls_readiness_after_noop_before_ownership_and_preserves_drain():
    source = (ROOT / "deploy-to-spark.sh").read_text()
    readiness_call = source.index("scripts/deploy_readiness.py")
    no_op = source.index("skipping drain, build, sync, and bounce")
    initial_ps = source.index('if ! initial_state="$(ps_state)"')
    drain_request = source.index("if ! drain_request; then", initial_ps)
    assert no_op < readiness_call < initial_ps < drain_request
    assert "DEPLOY_DRAIN_TIMEOUT=\"${DEPLOY_DRAIN_TIMEOUT:-30}\"" in source
    assert "DEPLOY_DRAIN_LEASE=\"${DEPLOY_DRAIN_LEASE:-300}\"" in source
    assert "DRAIN_PROVEN=0" in source


def test_deadline_is_assigned_once_before_the_poll_loop():
    import ast
    import inspect

    function = ast.parse(inspect.getsource(readiness.wait)).body[0]
    deadlines = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "deadline"
                         for target in node.targets)]
    assert len(deadlines) == 1
    loop_index = next(index for index, node in enumerate(function.body) if isinstance(node, ast.While))
    assert function.body.index(deadlines[0]) < loop_index


def test_pre_drain_preparation_overlaps_readiness_and_rejoins_before_drain_wait():
    source = (ROOT / "deploy-to-spark.sh").read_text()
    no_op = source.index("skipping drain, build, sync, and bounce")
    prep_start = source.index('PREP_LOG="$(mktemp -t arbiter-deploy-prep)"')
    prep_end = source.index(') >"$PREP_LOG" 2>&1 &')
    readiness_call = source.index("scripts/deploy_readiness.py")
    drain_requested = source.index('stage "Drain requested')
    rejoin = source.index('wait "$PREP_PID" || prep_rc=$?')
    drain_wait = source.index('stage "Draining arbiter')
    stop = source.index('stage "Stopping arbiter on spark..."')
    assert no_op < prep_start < prep_end < readiness_call < drain_requested < rejoin < drain_wait < stop
    prep = source[prep_start:prep_end]
    for step in ("arbiter.adapters import registry", "pip install", "build_ltx25_runtime.py' verify",
                 "build_binary arbiter-linux-arm64", "arbiter-go.new"):
        assert step in prep
    # The job may only stage: nothing on the live path is stopped or replaced.
    for live in ("run stop arbiter", "run start arbiter", "mv -f", "/v1/drain"):
        assert live not in prep
    trap = source.index("trap resume_if_aborted EXIT")
    assert prep_end < trap < readiness_call
    resume = source[source.index("resume_if_aborted() {"):source.index("\n}\n", source.index("resume_if_aborted() {"))]
    assert 'wait "$PREP_PID"' in resume
    assert not any(line.strip().startswith(("kill", "pkill")) for line in resume.splitlines())
