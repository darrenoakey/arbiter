"""Worker pool for running inference jobs in threads."""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import ArbiterConfig
    from .log import EventLogger
    from .memory import MemoryManager
    from .store import Job, JobStore

logger = logging.getLogger(__name__)


class WorkerPool:
    """Manages inference job execution in a thread pool."""

    def __init__(
        self,
        config: ArbiterConfig,
        store: JobStore,
        memory: MemoryManager,
        executor: ThreadPoolExecutor,
        output_dir: Path,
        event_logger: Optional[EventLogger] = None,
        on_job_done: Optional[callable] = None,
    ):
        self._config = config
        self._store = store
        self._memory = memory
        self._executor = executor
        self._output_dir = output_dir
        self._logger = event_logger
        self._on_job_done = on_job_done  # callback for scheduler cleanup

    async def run_job(self, job: Job, cancel_flag: asyncio.Event, instance_id: Optional[str] = None):
        """Run a job's inference in the thread pool. Handles completion/failure.

        instance_id: specific instance slot to use (for multi-instance models).
                     Falls back to job.model_id for single-instance models.
        """
        resolved_id = instance_id or job.model_id
        job_dir = self._output_dir / "jobs" / job.id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Convert asyncio.Event to threading.Event for the adapter
        import threading
        thread_cancel = threading.Event()

        # Bridge: if asyncio cancel_flag is set, set the threading one too
        async def _bridge_cancel():
            await cancel_flag.wait()
            thread_cancel.set()

        bridge_task = asyncio.create_task(_bridge_cancel())

        loop = asyncio.get_event_loop()
        start_time = time.time()

        try:
            slot = self._memory.get_slot(resolved_id)
            if slot is None:
                raise RuntimeError(f"No slot for instance {resolved_id}")

            # Inject job_type into params so adapters can dispatch
            infer_params = {**job.payload, "_job_type": job.job_type}
            result = await loop.run_in_executor(
                self._executor,
                slot.adapter.infer,
                infer_params,
                job_dir,
                thread_cancel,
            )

            elapsed = time.time() - start_time

            if thread_cancel.is_set():
                self._store.update_state(
                    job.id, "cancelled", finished_at=time.time()
                )
                self._log(
                    "job.cancelled", job_id=job.id, model_id=job.model_id,
                    was_running=True,
                )
            else:
                self._store.update_state(
                    job.id, "completed",
                    result=result,
                    finished_at=time.time(),
                )
                self._log(
                    "job.completed", job_id=job.id, model_id=job.model_id,
                    inference_seconds=round(elapsed, 2),
                )

        except Exception as e:
            from .adapters.base import CancelledException
            if isinstance(e, CancelledException):
                self._store.update_state(
                    job.id, "cancelled", finished_at=time.time()
                )
                self._log(
                    "job.cancelled", job_id=job.id, model_id=job.model_id,
                    was_running=True,
                )
            else:
                elapsed = time.time() - start_time
                error_msg = f"{type(e).__name__}: {e}"
                self._store.update_state(
                    job.id, "failed",
                    error=error_msg,
                    finished_at=time.time(),
                )
                self._log(
                    "job.failed", job_id=job.id, model_id=job.model_id,
                    error=error_msg, inference_seconds=round(elapsed, 2),
                )
                logger.error("Job %s failed: %s", job.id, e, exc_info=True)

        finally:
            bridge_task.cancel()
            # Release the specific instance slot
            self._memory.release(resolved_id)
            # Rescore since model may now have capacity
            if self._on_job_done:
                self._on_job_done(job)

    def _log(self, event: str, **kwargs):
        if self._logger:
            self._logger.log(event, **kwargs)
