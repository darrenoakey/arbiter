import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/deploy_post_stop.py"
SPEC = importlib.util.spec_from_file_location("deploy_post_stop", SCRIPT)
hook = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hook)


def write_config(path, command):
    import json

    path.write_text("[deploy]\npost_stop_command = " + json.dumps(command) + "\n")
    return path


def cli(config, mode):
    return subprocess.run([sys.executable, str(SCRIPT), mode, "--config", str(config)],
                          stdin=subprocess.DEVNULL, capture_output=True, text=True)


def test_missing_file_and_missing_key_are_noop(tmp_path):
    assert hook.command_from_config(tmp_path / "absent.toml") is None
    config = tmp_path / "config.toml"
    config.write_text("[deploy]\n")
    assert hook.command_from_config(config) is None


def test_run_command_executes_and_propagates_nonzero_and_timeout(tmp_path):
    marker = tmp_path / "invoked"
    good = tmp_path / "good"
    good.write_text(f"#!/bin/sh\nread value || true\nprintf '%s' invoked > '{marker}'\n")
    good.chmod(0o755)
    hook.run_command([str(good)])
    assert marker.read_text() == "invoked"
    bad = tmp_path / "bad"
    bad.write_text("#!/bin/sh\nexit 7\n")
    bad.chmod(0o755)
    with pytest.raises(subprocess.CalledProcessError) as error:
        hook.run_command([str(bad)])
    assert error.value.returncode == 7
    # The leader exits on TERM, but its child deliberately ignores TERM. The
    # process-group cleanup must still kill that child and reap the leader.
    import os
    import time

    marker = tmp_path / "orphan-marker"
    child_code = (
        "import signal,time,pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(.35); pathlib.Path(%r).write_text('orphaned')" % str(marker)
    )
    parent_code = (
        "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); "
        "time.sleep(5)" % child_code
    )
    with pytest.raises(subprocess.TimeoutExpired):
        hook.run_command([sys.executable, "-c", parent_code], timeout=0.1)
    time.sleep(0.45)
    assert not marker.exists()
    assert os.name == "posix"


def test_cli_validate_is_noop_execute_invokes_and_failures_exit_one(tmp_path):
    marker = tmp_path / "marker"
    program = tmp_path / "hook"
    program.write_text(f"#!/bin/sh\nprintf invoked > '{marker}'\n")
    program.chmod(0o755)
    config = write_config(tmp_path / "config.toml", [str(program)])
    assert cli(config, "--validate").returncode == 0
    assert not marker.exists()
    assert cli(config, "--execute").returncode == 0
    assert marker.read_text() == "invoked"

    malformed = tmp_path / "malformed.toml"
    malformed.write_text("[deploy]\npost_stop_command = [\n")
    assert cli(malformed, "--validate").returncode == 1
    invalid_command = write_config(tmp_path / "invalid.toml", ["relative-command"])
    assert cli(invalid_command, "--validate").returncode == 1
    failing = tmp_path / "failing"
    failing.write_text("#!/bin/sh\nexit 9\n")
    failing.chmod(0o755)
    assert cli(write_config(tmp_path / "failure.toml", [str(failing)]), "--execute").returncode == 1


def test_deploy_hook_order_is_after_release_proof_before_start():
    source = (ROOT / "deploy-to-spark.sh").read_text()
    validation = source.index("scripts/deploy_post_stop.py --validate")
    drain = source.index("drain_request")
    release = source.index('echo "    port 8400 released"')
    installed = source.index("mv -f $REMOTE/arbiter-go.new $REMOTE/arbiter-go")
    execute = source.index("scripts/deploy_post_stop.py --execute")
    start = source.index('"/home/darren/local/auto/run start arbiter"')
    assert validation < drain
    assert release < installed < execute < start
