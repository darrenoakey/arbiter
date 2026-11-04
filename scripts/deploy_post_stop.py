#!/usr/bin/env python3
"""Validate and optionally run the local deploy post-stop maintenance hook."""
import argparse
import os
import signal
import subprocess
import sys
import tomllib
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "local" / "config.toml"
TIMEOUT_SECONDS = 120


def command_from_config(config_path=CONFIG):
    if not config_path.exists():
        return None
    try:
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid {config_path}: {exc}") from exc
    deploy = config.get("deploy", {})
    if not isinstance(deploy, dict):
        raise ValueError("[deploy] must be a TOML table")
    if "post_stop_command" not in deploy:
        return None
    command = deploy["post_stop_command"]
    if (not isinstance(command, list) or not command or
            any(not isinstance(arg, str) or not arg for arg in command)):
        raise ValueError("deploy.post_stop_command must be a non-empty argv string list")
    executable = command[0]
    if (not os.path.isabs(executable) or not os.path.isfile(executable)
            or not os.access(executable, os.X_OK)):
        raise ValueError("deploy.post_stop_command argv[0] must be an absolute executable file")
    return command


def run_command(command, timeout=TIMEOUT_SECONDS):
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, shell=False,
                               start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # start_new_session makes this child PID the unique process-group ID.
        # Kill only that group, then reap the leader before reporting timeout.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return subprocess.CompletedProcess(command, returncode)


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    try:
        command = command_from_config(args.config)
        if args.execute and command:
            run_command(command)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"post-stop hook failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
