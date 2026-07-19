"""JSONL structured event logger for Arbiter."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


class EventLogger:
    """Append-only JSONL logger. One file per day, thread-safe."""

    def __init__(self, log_dir: str | Path):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current_date: str = ""
        self._file: TextIO | None = None

    def _ensure_file(self) -> TextIO:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._file:
                self._file.close()
            self._current_date = today
            path = self._log_dir / f"arbiter-{today}.jsonl"
            self._file = open(path, "a", encoding="utf-8")
        assert self._file is not None
        return self._file

    def log(self, event: str, **kwargs: Any):
        """Log a structured event."""
        entry = {"ts": time.time(), "event": event, **kwargs}
        line = json.dumps(entry, default=str) + "\n"
        with self._lock:
            log_file = self._ensure_file()
            log_file.write(line)
            log_file.flush()

    def close(self):
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None
