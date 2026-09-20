#!/usr/bin/env python3
"""Drain-wait verdict for deploy-to-spark.sh.

Reads one /v1/ps snapshot (JSON on stdin) plus the drain window length in
seconds (argv[1]) and prints two tab-separated fields:

    <active_jobs> <waitable>

active_jobs — the snapshot's worker-level in-flight count, verbatim.

waitable — 1 while at least one RUNNING job still deserves drain patience.
The patience policy is per-job, measured from EACH JOB'S OWN start: a job is
protected until it has run for one full drain window. A job that has already
outlived the window without finishing is empirically not a short job, and
waiting for it inside a bounded window whose total equals that same window is
pure release-time waste (measured 2026-09-20: 2 long training jobs burned ~80s
of a 209.6s release, then were requeued anyway).

This policy is never less protective than a wall-clock-from-request deadline:
a job that starts at the request gets the same full window; a job that started
earlier has already consumed part of its own window, exactly as before. The
overall deadline in deploy-to-spark.sh still bounds the total wait.

Clock skew between this Mac and spark biases the verdict toward "wait"
(a 5s grace is added to every age), so skew can only cost wait time, never
job protection. Anything unparseable or truncated also falls back to "wait",
which is the pre-2026-09-20 behavior.
"""

from __future__ import annotations

import datetime
import json
import sys
import time

# Matches cmd/arbiter/api.go activeJobsDetailLimit: above this the panel is
# truncated (newest entries dropped first by created_at ASC ordering), so the
# ages we can see are incomplete and the verdict must stay conservative.
DETAIL_PANEL_LIMIT = 200

# LAN NTP skew allowance, biased toward protecting jobs (treat as younger).
CLOCK_SKEW_GRACE_S = 5.0


def _started_epoch(value: object) -> float | None:
    """Parse started_at to a unix epoch, or None when unparseable.

    The live wire format is a unix-seconds FLOAT (store.Job marshals
    StartedAt *float64 — recorded 2026-09-20 from GET /v1/ps on spark); an
    RFC3339 string is tolerated in case a future panel marshals time.Time.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


def verdict(snapshot: object, window_s: float, now: float | None = None) -> tuple[int, int]:
    """Return (active_jobs, waitable) for one /v1/ps snapshot."""
    if now is None:
        now = time.time()
    if not isinstance(snapshot, dict):
        return 0, 0  # unparseable snapshot: pre-existing "drained" semantics
    active = snapshot.get("active_jobs", 0)
    if not isinstance(active, int):
        return 0, 0
    if active == 0:
        return 0, 0

    detail = snapshot.get("active_jobs_detail")
    if not isinstance(detail, list) or len(detail) >= DETAIL_PANEL_LIMIT:
        return active, 1  # can't see job ages — keep waiting (old behavior)

    # Worker-holding states: "running" executes; "scheduled" is dispatched
    # with started_at already set (recorded live 2026-09-20: both in-flight
    # chat-completion jobs sat in state "scheduled" with started_at ~60ms
    # after created_at). "following" rows are dedup shadows of an original
    # and "queued" rows never start under drain — neither holds patience.
    holding_states = {"scheduled", "running"}
    holding_started = [
        _started_epoch(j.get("started_at"))
        for j in detail
        if isinstance(j, dict) and j.get("state") in holding_states
    ]
    if not holding_started:
        return active, 1  # no worker-holding row visible — keep waiting

    waitable = any(
        started is None or (now - started) < window_s + CLOCK_SKEW_GRACE_S
        for started in holding_started
    )
    return active, int(waitable)


def main() -> int:
    window_s = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    try:
        snapshot = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        snapshot = None  # curl failure / empty body → historical "0 0" output
    active, waitable = verdict(snapshot, window_s)
    print(f"{active}\t{waitable}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
