"""JSONL structured event logger for Arbiter."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLogger:
    """Append-only JSONL logger. One file per day, thread-safe."""

    def __init__(self, log_dir: str | Path):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current_date: str = ""
        self._file = None

    def _ensure_file(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._file:
                self._file.close()
            self._current_date = today
            path = self._log_dir / f"arbiter-{today}.jsonl"
            self._file = open(path, "a", encoding="utf-8")

    def log(self, event: str, **kwargs: Any):
        """Log a structured event."""
        entry = {"ts": time.time(), "event": event, **kwargs}
        line = json.dumps(entry, default=str) + "\n"
        with self._lock:
            self._ensure_file()
            self._file.write(line)
            self._file.flush()

    def close(self):
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None
