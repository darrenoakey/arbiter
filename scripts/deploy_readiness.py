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


def ready(snapshot: object) -> bool:
    """Only an undrained, validated snapshot with coherent strict-zero counts is ready."""
    draining, active = ps_state(snapshot)
    if draining:
        raise DrainProtocolError("/v1/ps reports draining=true; ownership is external")
    # Queued jobs remain persisted; only active/running/scheduled work must be zero.
    return active == 0


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
            print("==> Read-only readiness: validated undrained strict-zero snapshot")
            return 0
        print("ERROR: pre-drain readiness deadline expired; ordinary drain will not be requested", file=sys.stderr)
        return 1
    except (ValueError, OSError, urllib.error.URLError, json.JSONDecodeError, DrainProtocolError) as error:
        print(f"ERROR: pre-drain readiness failed closed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
