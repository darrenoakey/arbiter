"""Real CPU children exercise LatentSync's worker output boundary."""

import subprocess
import sys

import pytest

from arbiter.adapters import latentsync
from arbiter.adapters.base import InferenceError


def test_stage_routes_both_child_streams_to_worker_stderr(tmp_path, capfd):
    latentsync._run_stage(
        [sys.executable, "-u", "-c", "import sys; print('frame 10/20'); print('diagnostic', file=sys.stderr)"],
        cwd=tmp_path, timeout=5, stage="inference",
    )
    output = capfd.readouterr()
    assert output.out == ""  # stdout belongs exclusively to worker JSON replies.
    assert "frame 10/20" in output.err
    assert "diagnostic" in output.err


def test_stage_failure_retains_child_diagnostic_in_worker_log(tmp_path, capfd):
    with pytest.raises(InferenceError, match="mouth guard.*7"):
        latentsync._run_stage(
            [sys.executable, "-c", "import sys; print('decode failed', file=sys.stderr); sys.exit(7)"],
            cwd=tmp_path, timeout=5, stage="mouth guard",
        )
    assert "decode failed" in capfd.readouterr().err


def test_stage_timeout_names_the_stage_and_reaps_child(tmp_path):
    with pytest.raises(InferenceError, match="mouth guard.*timed out.*0.1"):
        latentsync._run_stage(
            [sys.executable, "-c", "import threading; threading.Event().wait(30)"],
            cwd=tmp_path, timeout=0.1, stage="mouth guard",
        )


def test_progress_reaches_parent_before_child_finishes(tmp_path):
    # A FIFO blocks the real child after its first progress line. If the
    # adapter captures output until exit, this read cannot complete.
    import os
    import selectors

    release = tmp_path / "release"
    os.mkfifo(release)
    child = "import pathlib; print('landmarks 10/131', flush=True); pathlib.Path(%r).read_bytes()" % str(release)
    driver = (
        "import sys; from pathlib import Path; "
        "from arbiter.adapters.latentsync import _run_stage; "
        f"_run_stage([sys.executable, '-u', '-c', {child!r}], "
        f"cwd=Path({str(tmp_path)!r}), timeout=5, stage='mouth guard')"
    )
    with subprocess.Popen([sys.executable, "-c", driver], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stderr, selectors.EVENT_READ)
                assert selector.select(timeout=4), "progress was hidden until subprocess exit"
                assert process.stderr.readline() == b"landmarks 10/131\n"
            assert process.poll() is None
            with release.open("wb") as stream:
                stream.write(b"finish")
            stdout, stderr = process.communicate(timeout=5)
            assert process.returncode == 0, stderr.decode()
            assert stdout == b""
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
