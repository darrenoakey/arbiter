#!/usr/bin/env python3
"""Optional bounded, read-only readiness check before requesting a deploy drain."""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from drain_wait import DrainProtocolError, ps_state

MAX_WAIT_SECONDS = 180
MAX_BODY_BYTES = 1_000_000
HTTP_TIMEOUT_SECONDS = 3
POLL_SECONDS = 2
# Short-work readiness. A strict-zero wait on a busy fleet waits on queue churn,
# not on any job: each finishing chat completion is replaced by the next one,
# so a zero instant may never occur (2026-11-11: 120s of a 224.3s release).
# Work that the deploy's own 30s owned drain will outlast needs no read-only
# wait, because the drain stops dispatch and still independently proves zero
# before any stop. A job qualifies only once Arbiter marks it "running" (set
# after its model load succeeded) on a model whose recorded mean execution is
# a third of that drain deadline, over enough samples to mean something.
# Anything else (scheduled/loading, long or unmeasured models, an incomplete
# job panel) keeps the strict-zero wait.
SHORT_JOB_MAX_MEAN_SECONDS = 10.0
SHORT_JOB_MIN_SAMPLES = 20


def configured_wait(path: Path) -> int:
    try:
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    except FileNotFoundError:
        return 0
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot parse {path}: {error}") from error
    section = config.get("deploy", {})
    if not isinstance(section, dict):
        raise ValueError("[deploy] must be a table")
    value = section.get("pre_drain_wait_seconds", 0)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_WAIT_SECONDS:
        raise ValueError(f"[deploy].pre_drain_wait_seconds must be an integer from 0 to {MAX_WAIT_SECONDS}")
    return value


def only_short_running_work(snapshot: dict, active: int) -> bool:
    """True only when every active job is provably short, running, resident work."""
    stats = {}
    for model in snapshot["models"]:
        mean = model.get("avg_execution_seconds")
        samples = model.get("completed_jobs")
        if (isinstance(model.get("id"), str) and isinstance(mean, (int, float))
                and not isinstance(mean, bool) and isinstance(samples, int)
                and not isinstance(samples, bool)):
            stats[model["id"]] = (float(mean), samples)
    detail = snapshot.get("active_jobs_detail")
    if not isinstance(detail, list):
        return False
    in_flight = [job for job in detail
                 if not isinstance(job, dict) or job.get("state") != "queued"]
    # The panel must account for exactly the validated active count; a capped,
    # stale, or partial panel never qualifies.
    if len(in_flight) != active:
        return False
    for job in in_flight:
        if not isinstance(job, dict) or job.get("state") != "running":
            return False
        mean, samples = stats.get(job.get("model"), (None, 0))
        if mean is None or samples < SHORT_JOB_MIN_SAMPLES or not 0 <= mean <= SHORT_JOB_MAX_MEAN_SECONDS:
            return False
    return True


def ready(snapshot: object) -> bool:
    """An undrained, validated snapshot is ready at strict zero or with only short running work."""
    draining, active = ps_state(snapshot)
    if draining:
        raise DrainProtocolError("/v1/ps reports draining=true; ownership is external")
    # Queued jobs remain persisted; only active/running/scheduled work matters.
    if active == 0:
        return True
    assert isinstance(snapshot, dict)  # ps_state validated the shape.
    return only_short_running_work(snapshot, active)


def fetch(url: str) -> object:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        body = response.read(MAX_BODY_BYTES + 1)
    if len(body) > MAX_BODY_BYTES:
        raise ValueError("/v1/ps response exceeds size limit")
    return json.loads(body)


def wait(url: str, seconds: int) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        snapshot = fetch(url)
        if time.monotonic() > deadline:
            return False
        if ready(snapshot):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(POLL_SECONDS, remaining))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        seconds = configured_wait(args.config)
        if seconds == 0:
            return 0
        if wait(args.url, seconds):
            print("==> Read-only readiness: validated undrained snapshot with zero or only short running work")
            return 0
        print("ERROR: pre-drain readiness deadline expired; ordinary drain will not be requested", file=sys.stderr)
        return 1
    except (ValueError, OSError, urllib.error.URLError, json.JSONDecodeError, DrainProtocolError) as error:
        print(f"ERROR: pre-drain readiness failed closed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
