#!/usr/bin/env python3
"""Validate the Arbiter drain protocol responses used by deploy-to-spark.sh."""

from __future__ import annotations

import argparse
import json
import sys
from typing import NoReturn


class DrainProtocolError(ValueError):
    """The server response cannot prove the requested drain state."""


def _fail(message: str) -> NoReturn:
    raise DrainProtocolError(message)


def _count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{field} must be a non-negative integer")
    return value


def ps_state(snapshot: object) -> tuple[bool, int]:
    """Return the validated global drain state and active-job count."""
    if not isinstance(snapshot, dict):
        _fail("/v1/ps response must be an object")

    draining = snapshot.get("draining")
    if not isinstance(draining, bool):
        _fail("/v1/ps draining must be a boolean")

    if "active_jobs" not in snapshot:
        _fail("/v1/ps active_jobs is missing")
    active = _count(snapshot["active_jobs"], "/v1/ps active_jobs")

    models = snapshot.get("models")
    if not isinstance(models, list):
        _fail("/v1/ps models must be an array")
    model_active = 0
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            _fail(f"/v1/ps models[{index}] must be an object")
        if "active_jobs" not in model:
            _fail(f"/v1/ps models[{index}].active_jobs is missing")
        model_active += _count(
            model["active_jobs"], f"/v1/ps models[{index}].active_jobs"
        )
    if model_active != active:
        _fail(
            "/v1/ps active_jobs does not equal the sum of models[].active_jobs"
        )

    queue = snapshot.get("queue")
    if not isinstance(queue, dict):
        _fail("/v1/ps queue must be an object")
    for state, count in queue.items():
        if not isinstance(state, str) or not state:
            _fail("/v1/ps queue keys must be non-empty strings")
        _count(count, f"/v1/ps queue.{state}")

    return draining, active


def wait_decision(snapshot: object, *, deadline_reached: bool) -> tuple[str, int]:
    """Decide whether a validated owned drain is safe, waiting, or timed out."""
    draining, active = ps_state(snapshot)
    if not draining:
        _fail("/v1/ps no longer confirms draining=true")
    if active == 0:
        return "drained", active
    if deadline_reached:
        return "abort", active
    return "wait", active


def drain_response(
    response: object, *, resumed: bool, expected_lease: int | None = None
) -> None:
    """Validate the documented POST /v1/drain acknowledgement."""
    if not isinstance(response, dict):
        _fail("/v1/drain response must be an object")
    expected = not resumed
    if response.get("draining") is not expected:
        _fail(f"/v1/drain response must contain draining={str(expected).lower()}")
    if not resumed:
        if "active_jobs" not in response:
            _fail("/v1/drain active_jobs is missing")
        _count(response["active_jobs"], "/v1/drain active_jobs")
        if "lease_seconds" not in response:
            _fail("/v1/drain lease_seconds is missing")
        lease = _count(response["lease_seconds"], "/v1/drain lease_seconds")
        if lease == 0:
            _fail("/v1/drain lease_seconds must be positive")
        if expected_lease is not None and lease != expected_lease:
            _fail(
                f"/v1/drain lease_seconds is {lease}, expected {expected_lease}"
            )


def _read_json() -> object:
    try:
        return json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DrainProtocolError(f"invalid JSON: {error.msg}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("ps", "wait", "drain", "resume"))
    parser.add_argument("lease_seconds", nargs="?", type=int)
    arguments = parser.parse_args()
    try:
        payload = _read_json()
        if arguments.kind == "ps":
            draining, active = ps_state(payload)
            print(f"{int(draining)}\t{active}")
        elif arguments.kind == "wait":
            if arguments.lease_seconds not in (0, 1):
                _fail("wait requires deadline_reached as 0 or 1")
            decision, active = wait_decision(
                payload, deadline_reached=bool(arguments.lease_seconds)
            )
            print(f"{decision}\t{active}")
        else:
            if arguments.kind == "drain" and arguments.lease_seconds is None:
                _fail("expected lease_seconds argument is missing")
            drain_response(
                payload,
                resumed=arguments.kind == "resume",
                expected_lease=arguments.lease_seconds,
            )
    except DrainProtocolError as error:
        print(f"invalid Arbiter {arguments.kind} response: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
