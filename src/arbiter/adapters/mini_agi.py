"""Continual-learning chunks for mini-AGI (github.com/volotat/mini-AGI).

mini-AGI is a byte-level MoE language model that keeps learning from whatever
it reads. One job is ONE bounded reading session: load the run's weights
directory, read its corpus for at most `minutes`, checkpoint, return. A long
read is a chain of these jobs submitted one after another by the caller, so
the GPU is never held for more than one chunk and other work is scheduled
between chunks.

Layout on spark (all fixed here; a job only names a run):

    /home/darren/src/mini-AGI                 pinned upstream checkout (code)
    /home/darren/training/mini-agi/<run>/
        weights/      the model - mini-AGI's weights directory
        data/         the corpus, one top-level folder per subject
        runs/         sample log, expert history, corpus index
        pass.json     {"start_chars", "corpus_chars"}: the pass being read

The job returns how far through the pass the run is and `done` once a whole
pass has been read, so the caller knows when to stop submitting.

Training runs in the worker thread through train.py's own `read` command, so
the worker's CUDA memory cap applies to it. train.py prints its progress to
stdout, which is the worker's protocol channel, so stdout is redirected to
stderr for the duration.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

from arbiter.adapters.base import CancelledException, InferenceError, ModelAdapter
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

CODE_ROOT = Path("/home/darren/src/mini-AGI")
TRAINING_ROOT = Path("/home/darren/training/mini-agi")
RUN_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MAX_MINUTES = 30.0
DEFAULT_MINUTES = 26.0
SAMPLE_TAIL_BYTES = 6000


class _StopMinutes(float):
    """A minute budget that also ends early when the job is cancelled.

    train.py stops its reading loop when `(now - t0) / 60 >= args.minutes`.
    Because this is a float subclass that overrides the reflected comparison,
    Python asks it first, so a cancel ends the loop at the next chunk boundary
    and train.py writes its normal final checkpoint instead of being killed
    between checkpoints.
    """

    def __new__(cls, minutes: float, cancel_flag: threading.Event):
        value = super().__new__(cls, minutes)
        value.cancel_flag = cancel_flag
        return value

    def __le__(self, elapsed):
        return self.cancel_flag.is_set() or float(self) <= elapsed


def run_dir(run: str) -> Path:
    if not isinstance(run, str) or not RUN_NAME.match(run):
        raise InferenceError("run must be a lowercase slug (a-z, 0-9, '-', '_')")
    return TRAINING_ROOT / run


def chunk_minutes(params: dict) -> float:
    try:
        minutes = float(params.get("minutes", DEFAULT_MINUTES))
    except (TypeError, ValueError) as exc:
        raise InferenceError("minutes must be a number") from exc
    if not 0 < minutes <= MAX_MINUTES:
        raise InferenceError(f"minutes must be in (0, {MAX_MINUTES:g}]")
    return minutes


def corpus_chars(data: Path) -> int:
    return sum(
        (Path(root) / name).stat().st_size
        for root, _dirs, files in os.walk(data)
        for name in files
        if name.endswith(".txt")
    )


def read_chars(weights: Path) -> int:
    manifest = json.loads((weights / "manifest.json").read_text())
    return int(manifest.get("read_chars") or 0)


def pass_state(root: Path) -> dict:
    """The pass this run is reading; started at the current position if new."""
    path = root / "pass.json"
    if path.is_file():
        return json.loads(path.read_text())
    state = {
        "start_chars": read_chars(root / "weights"),
        "corpus_chars": corpus_chars(root / "data"),
        "started_at": time.time(),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)
    return state


def progress(root: Path) -> dict:
    state = pass_state(root)
    now = read_chars(root / "weights")
    read = max(0, now - int(state["start_chars"]))
    total = int(state["corpus_chars"])
    return {
        "read_chars": now,
        "pass_read_chars": read,
        "pass_corpus_chars": total,
        "pass_fraction": round(read / total, 6) if total else 0.0,
        "done": total > 0 and read >= total,
    }


def sample_tail(log_path: Path, since: int) -> str:
    if not log_path.is_file():
        return ""
    with log_path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        start = max(since, size - SAMPLE_TAIL_BYTES)
        handle.seek(start)
        return handle.read().decode("utf-8", errors="replace")


def _load_train_module():
    path = CODE_ROOT / "train.py"
    if not path.is_file():
        raise InferenceError(f"mini-AGI checkout missing: {path}")
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    spec = importlib.util.spec_from_file_location("mini_agi_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_args(train, root: Path, minutes: float, samples: bool, cancel_flag) -> object:
    """train.py's own `read` arguments, so its defaults stay the source of truth."""
    argv = [
        "--device", "cuda",
        "read", str(root / "data"),
        "--weights-dir", str(root / "weights"),
        "--save",
        "--held-out", "",
        "--no-plots",
        "--sample-log", str(root / "runs" / "samples.txt"),
        "--history", str(root / "runs" / "expert_history.jsonl"),
        # Mid-chunk samples never fire; one round is written at the end.
        "--sample-every", str(MAX_MINUTES * 10 if samples else 0),
    ]
    parser = _parser(train)
    args = parser.parse_args(argv)
    args.minutes = _StopMinutes(minutes, cancel_flag)
    return args


def _parser(train):
    """Capture the parser train.main() builds, without running anything."""
    import argparse

    captured = {}
    original = argparse.ArgumentParser.parse_args

    def capture(self, *a, **k):
        captured["parser"] = self
        raise _Captured

    argparse.ArgumentParser.parse_args = capture
    try:
        train.main()
    except _Captured:
        pass
    finally:
        argparse.ArgumentParser.parse_args = original
    parser = captured.get("parser")
    if parser is None:
        raise InferenceError("could not build train.py's argument parser")

    class _Parser:
        def parse_args(self, argv):
            return original(parser, argv)

    return _Parser()


class _Captured(Exception):
    pass


@register
class MiniAgiAdapter(ModelAdapter):
    """Read one bounded chunk of a mini-AGI run's corpus and checkpoint."""

    model_id = "mini-agi"

    def __init__(self):
        self._train = None

    def load(self, device: str = "cuda") -> None:
        # Nothing stays resident between chunks: each job loads the weights
        # directory, trains, saves and frees, so an idle worker holds no GPU.
        self._train = _load_train_module()

    def unload(self) -> None:
        self._train = None
        self._cleanup_gpu()

    def estimate_time(self, params: dict) -> float:
        try:
            return chunk_minutes(params) * 60_000.0 + 120_000.0
        except InferenceError:
            return DEFAULT_MINUTES * 60_000.0

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        root = run_dir(params.get("run", "mnemnos"))
        minutes = chunk_minutes(params)
        samples = bool(params.get("samples", True))
        for need in ("weights/manifest.json", "data"):
            if not (root / need).exists():
                raise InferenceError(f"run is not set up: {root / need} is missing")
        (root / "runs").mkdir(exist_ok=True)
        self._check_cancel(cancel_flag)

        before = progress(root)
        if before["done"]:
            return {**before, "chunk_chars": 0, "minutes": 0.0, "samples": ""}
        train = self._train or _load_train_module()
        args = read_args(train, root, minutes, samples, cancel_flag)
        sample_log = root / "runs" / "samples.txt"
        sample_start = sample_log.stat().st_size if sample_log.is_file() else 0

        log.info("mini-agi: reading %s for up to %.1f min from %d chars", root, minutes, before["read_chars"])
        started = time.time()
        cwd = os.getcwd()
        try:
            os.chdir(root)  # train.py keeps its corpus index under ./runs
            with contextlib.redirect_stdout(sys.stderr):
                code = train.cmd_read(args)
        finally:
            os.chdir(cwd)
            self._cleanup_gpu()
        if code:
            raise InferenceError(f"train.py read exited {code}")

        after = progress(root)
        result = {
            **after,
            "chunk_chars": after["read_chars"] - before["read_chars"],
            "minutes": round((time.time() - started) / 60.0, 2),
            "samples": sample_tail(sample_log, sample_start) if samples else "",
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2))
        if cancel_flag.is_set():
            raise CancelledException("mini-agi chunk cancelled after checkpoint")
        return result
