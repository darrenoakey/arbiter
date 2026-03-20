"""Persistent job store backed by SQLite."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Job:
    id: str
    model_id: str
    job_type: str
    state: str
    priority: float
    payload: dict
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'queued',
    priority REAL NOT NULL DEFAULT 0,
    payload TEXT NOT NULL DEFAULT '{}',
    result TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority) WHERE state = 'queued';
CREATE INDEX IF NOT EXISTS idx_jobs_model ON jobs(model_id);
"""


class JobStore:
    """Thread-safe SQLite-backed job persistence."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = self._connect()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            model_id=row["model_id"],
            job_type=row["job_type"],
            state=row["state"],
            priority=row["priority"],
            payload=json.loads(row["payload"]),
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def create_job(self, model_id: str, job_type: str, payload: dict, priority: float = 0.0) -> Job:
        """Create a new job in 'queued' state. Returns the Job."""
        job_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, model_id, job_type, state, priority, payload, created_at) "
                "VALUES (?, ?, ?, 'queued', ?, ?, ?)",
                (job_id, model_id, job_type, priority, json.dumps(payload), now),
            )
            self._conn.commit()
        return Job(
            id=job_id, model_id=model_id, job_type=job_type,
            state="queued", priority=priority, payload=payload, created_at=now,
        )

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self, state: Optional[str] = None, model_id: Optional[str] = None, limit: int = 100) -> list[Job]:
        """List jobs, optionally filtered by state and/or model_id."""
        query = "SELECT * FROM jobs WHERE 1=1"
        params: list = []
        if state:
            # Support comma-separated states
            states = [s.strip() for s in state.split(",")]
            placeholders = ",".join("?" for _ in states)
            query += f" AND state IN ({placeholders})"
            params.extend(states)
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_job(r) for r in rows]

    def pick_next_job(self, exclude_models: Optional[set[str]] = None) -> Optional[Job]:
        """Pick the highest-priority (lowest score) queued job, optionally excluding certain models."""
        query = "SELECT * FROM jobs WHERE state = 'queued'"
        params: list = []
        if exclude_models:
            placeholders = ",".join("?" for _ in exclude_models)
            query += f" AND model_id NOT IN ({placeholders})"
            params.extend(exclude_models)
        query += " ORDER BY priority ASC, created_at ASC LIMIT 1"
        with self._lock:
            row = self._conn.execute(query, params).fetchone()
        return self._row_to_job(row) if row else None

    def update_state(self, job_id: str, state: str, **kwargs) -> bool:
        """Update job state. Optional kwargs: started_at, finished_at, result, error."""
        sets = ["state = ?"]
        params: list = [state]
        if "started_at" in kwargs:
            sets.append("started_at = ?")
            params.append(kwargs["started_at"])
        if "finished_at" in kwargs:
            sets.append("finished_at = ?")
            params.append(kwargs["finished_at"])
        if "result" in kwargs:
            sets.append("result = ?")
            params.append(json.dumps(kwargs["result"]) if kwargs["result"] else None)
        if "error" in kwargs:
            sets.append("error = ?")
            params.append(kwargs["error"])
        params.append(job_id)
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def update_priority(self, model_id: str, new_priority: float) -> int:
        """Re-score all queued jobs for a given model. Returns count updated."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE jobs SET priority = ? WHERE model_id = ? AND state = 'queued'",
                (new_priority, model_id),
            )
            self._conn.commit()
            return cursor.rowcount

    def count_by_state(self, model_id: Optional[str] = None) -> dict[str, int]:
        """Count jobs grouped by state."""
        if model_id:
            query = "SELECT state, COUNT(*) as cnt FROM jobs WHERE model_id = ? GROUP BY state"
            params = (model_id,)
        else:
            query = "SELECT state, COUNT(*) as cnt FROM jobs GROUP BY state"
            params = ()
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return {row["state"]: row["cnt"] for row in rows}

    def count_running(self, model_id: str) -> int:
        """Count running jobs for a specific model."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM jobs WHERE model_id = ? AND state = 'running'",
                (model_id,),
            ).fetchone()
        return row["cnt"] if row else 0

    def recover_from_crash(self) -> int:
        """Reset scheduled/running jobs to queued. Call on startup. Returns count recovered."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE jobs SET state = 'queued', started_at = NULL "
                "WHERE state IN ('scheduled', 'running')"
            )
            self._conn.commit()
            return cursor.rowcount

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's queued or scheduled. Returns True if cancelled."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE jobs SET state = 'cancelled', finished_at = ? "
                "WHERE id = ? AND state IN ('queued', 'scheduled')",
                (time.time(), job_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def close(self):
        """Close the database connection."""
        with self._lock:
            self._conn.close()
