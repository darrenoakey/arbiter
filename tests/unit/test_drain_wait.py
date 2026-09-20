"""Real unit tests for scripts/drain_wait.py (deploy drain-wait verdict).

Feeds /v1/ps snapshots with the exact field names and shapes the Go API
marshals (cmd/arbiter/api.go updatePSCache over store.Job rows): unix-float
started_at, non-terminal states queued/scheduled/running/following — shapes
recorded live from spark 2026-09-20. Asserts the per-job patience verdict.
No network, no mocks — pure function over recorded shapes.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "drain_wait.py"

spec = importlib.util.spec_from_file_location("drain_wait", SCRIPT)
drain_wait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drain_wait)

NOW = 1_800_000_000.0


def started(seconds_ago: float) -> float:
    # Live wire format: unix-seconds float (store.Job StartedAt *float64).
    return NOW - seconds_ago


def ps(active: int, jobs: list[dict] | None = None) -> dict:
    snap = {"active_jobs": active, "draining": True}
    if jobs is not None:
        snap["active_jobs_detail"] = jobs
    return snap


def running(started_at: float | str | None, job_id: str = "job-1",
            state: str = "scheduled") -> dict:
    # "scheduled" is the observed in-flight state: dispatched with started_at
    # set (recorded live: started_at lands ~60ms after created_at).
    job = {"job_id": job_id, "type": "ltx25", "model": "ltx25", "state": state}
    if started_at is not None:
        job["started_at"] = started_at
    return job


def test_no_active_jobs_is_drained():
    assert drain_wait.verdict(ps(0, []), 120.0, now=NOW) == (0, 0)


def test_young_running_job_is_waitable():
    # Started 10s ago inside a 120s window: deserves the remaining patience.
    snap = ps(1, [running(started(10))])
    assert drain_wait.verdict(snap, 120.0, now=NOW) == (1, 1)


def test_job_that_outlived_the_window_is_not_waitable():
    # Started 400s ago; a 120s patience window is long spent.
    snap = ps(2, [running(started(400), "a"), running(started(900), "b")])
    assert drain_wait.verdict(snap, 120.0, now=NOW) == (2, 0)


def test_mixed_young_and_old_waits_for_the_young_one():
    snap = ps(2, [running(started(400), "old"), running(started(30), "young")])
    assert drain_wait.verdict(snap, 120.0, now=NOW) == (2, 1)


def test_boundary_age_just_under_window_is_waitable():
    # Age 119s of a 120s window (+5s skew grace) — still protected.
    snap = ps(1, [running(started(119))])
    assert drain_wait.verdict(snap, 120.0, now=NOW) == (1, 1)


def test_age_slightly_past_window_plus_grace_is_not_waitable():
    snap = ps(1, [running(started(130))])
    assert drain_wait.verdict(snap, 120.0, now=NOW) == (1, 0)


def test_missing_started_at_is_waitable():
    # Cannot prove the job is long — protect it (old behavior).
    snap = ps(1, [running(None)])
    assert drain_wait.verdict(snap, 120.0, now=NOW) == (1, 1)


def test_rfc3339_string_started_at_is_tolerated():
    # Compatibility path: a future panel marshaling time.Time would emit
    # RFC3339 strings; parsing must still age the job, not silently protect
    # it forever (that bug shipped briefly on 2026-09-20 and cost the fix).
    iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(NOW - 400))
    assert drain_wait.verdict(ps(1, [running(iso)]), 120.0, now=NOW) == (1, 0)


def test_recorded_live_snapshot_shapes_drive_the_verdict():
    # Byte-shape recorded from GET /v1/ps on spark 2026-09-20: chat-completion
    # jobs in state "scheduled" with float started_at ~60ms after float
    # created_at. Young at snapshot time — one full patience window ahead.
    live = {
        "active_jobs": 2,
        "draining": True,
        "active_jobs_detail": [
            {"job_id": "000f9a6290c4", "type": "chat-completion", "model": "local-titler",
             "state": "scheduled", "created_at": NOW - 10.0, "started_at": NOW - 9.94},
            {"job_id": "a0f98b1ac7f6", "type": "chat-completion", "model": "local-titler",
             "state": "scheduled", "created_at": NOW - 9.8, "started_at": NOW - 9.75},
        ],
    }
    assert drain_wait.verdict(live, 120.0, now=NOW) == (2, 1)
    # Same jobs 400s later: both outlived the window — cut the wait.
    assert drain_wait.verdict(live, 120.0, now=NOW + 400) == (2, 0)


def test_followers_and_queued_do_not_extend_the_originals_patience():
    # "following" rows are dedup followers (cmd/arbiter/dedup.go) piggybacking
    # on an original job's result — the ORIGINAL's running row holds the
    # worker, so its age alone governs patience; queued work never starts
    # under drain and is requeued wholesale by the shutdown path.
    jobs = [
        running(started(400), "orig"),
        {"job_id": "f", "type": "following", "state": "following",
         "started_at": started(50)},
        {"job_id": "q", "type": "llm-chat", "state": "queued"},
    ]
    snap = ps(1, jobs)
    assert drain_wait.verdict(snap, 120.0, now=NOW) == (1, 0)


def test_followers_without_a_running_original_wait():
    # active>0 but no running row visible: cannot prove the worker is idle of
    # long jobs (schema drift) — stay conservative and keep waiting.
    jobs = [{"job_id": "f", "type": "following", "state": "following",
             "started_at": started(900)}]
    assert drain_wait.verdict(ps(1, jobs), 120.0, now=NOW) == (1, 1)


def test_missing_detail_panel_falls_back_to_waiting():
    # Older server without active_jobs_detail — never cut the wait short.
    assert drain_wait.verdict(ps(3), 120.0, now=NOW) == (3, 1)


def test_truncated_detail_panel_falls_back_to_waiting():
    # Panel at the 200-entry cap drops newest jobs first, so the visible ages
    # cannot prove anything — stay conservative.
    jobs = [running(started(999), f"old-{i}") for i in range(200)]
    snap = ps(200, jobs)
    assert drain_wait.verdict(snap, 120.0, now=NOW) == (200, 1)


def test_unparseable_snapshot_reports_drained_like_the_old_loop():
    assert drain_wait.verdict(None, 120.0, now=NOW) == (0, 0)
    assert drain_wait.verdict("not-a-dict", 120.0, now=NOW) == (0, 0)


def test_cli_prints_tab_separated_verdict():
    # started_at is computed against the REAL clock the CLI subprocess uses,
    # in the live unix-float wire format.
    real = time.time()
    jobs = [
        {"job_id": "a", "type": "lora-train", "state": "running",
         "started_at": real - 400},
        {"job_id": "b", "type": "lora-train", "state": "scheduled",
         "started_at": real - 900},
    ]
    payload = json.dumps(ps(2, jobs))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "120"],
        input=payload, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "2\t0"


def test_cli_on_empty_input_matches_legacy_zero_semantics():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "120"],
        input="", capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "0\t0"
