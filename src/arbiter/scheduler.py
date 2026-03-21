"""Scheduler loop for Arbiter — SJF with memory-aware dispatch."""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import ArbiterConfig
    from .log import EventLogger
    from .memory import MemoryManager
    from .store import Job, JobStore
    from .worker import WorkerPool

logger = logging.getLogger(__name__)


class Scheduler:
    """Shortest-job-first scheduler with memory-aware model loading.

    The scheduler continuously:
    1. Picks the highest-priority queued job
    2. Picks the best instance for that model (multi-instance aware)
    3. Ensures the instance is loaded (may evict others)
    4. Dispatches to a worker thread
    5. Speculatively pre-loads the next job's model
    """

    def __init__(
        self,
        config: ArbiterConfig,
        store: JobStore,
        memory: MemoryManager,
        worker_pool: WorkerPool,
        event_logger: Optional[EventLogger] = None,
    ):
        self._config = config
        self._store = store
        self._memory = memory
        self._worker = worker_pool
        self._logger = event_logger
        self._task: Optional[asyncio.Task] = None
        self._cancel_flags: dict[str, asyncio.Event] = {}  # job_id -> cancel event
        self._running = False

    def compute_priority(self, model_id: str) -> float:
        """Compute SJF priority score for a model.

        Lower = run sooner.
        Score = avg_inference_ms + (load_ms if model not loaded else 0)
        """
        model_cfg = self._config.models.get(model_id)
        if not model_cfg:
            return float("inf")

        inference_ms = model_cfg.avg_inference_ms
        load_ms = 0.0
        if not self._memory.is_loaded(model_id):
            load_ms = model_cfg.load_ms
        return inference_ms + load_ms

    def rescore_model(self, model_id: str):
        """Re-score all queued jobs for a model (call when load state changes)."""
        new_priority = self.compute_priority(model_id)
        count = self._store.update_priority(model_id, new_priority)
        if count > 0:
            logger.debug("Rescored %d jobs for %s -> priority %.0f", count, model_id, new_priority)

    def rescore_all(self):
        """Re-score all queued jobs for all models."""
        for model_id in self._config.models:
            self.rescore_model(model_id)

    def _get_full_models(self) -> set[str]:
        """Get model IDs that are at their total capacity.

        A model is full when in-flight jobs (scheduled + running) >= max_instances * max_concurrent.
        Uses count_active so that jobs being loaded don't cause over-dispatch.
        """
        full = set()
        for model_id, model_cfg in self._config.models.items():
            active = self._store.count_active(model_id)
            total_capacity = model_cfg.max_instances * model_cfg.max_concurrent
            if active >= total_capacity:
                full.add(model_id)
        return full

    async def _dispatch_job(self, job: Job):
        """Pick an instance, load it, and dispatch the job to a worker."""
        from .memory import EvictionImpossible

        model_cfg = self._config.models.get(job.model_id)
        max_concurrent = model_cfg.max_concurrent if model_cfg else 1

        self._store.update_state(job.id, "scheduled")
        self._log("job.scheduled", job_id=job.id, model_id=job.model_id)

        # Pick the best instance for this model
        instance_id = self._memory.pick_instance(job.model_id, max_concurrent)
        if instance_id is None:
            self._store.update_state(job.id, "queued")
            logger.debug("No instance available for %s, requeueing job %s", job.model_id, job.id)
            return

        try:
            await self._memory.ensure_loaded(instance_id)
        except EvictionImpossible:
            # Can't load right now — put it back
            self._store.update_state(job.id, "queued")
            logger.debug("Can't load %s, requeueing job %s", instance_id, job.id)
            return
        except Exception as e:
            self._store.update_state(
                job.id, "failed",
                error=f"Model load failed: {e}",
                finished_at=time.time(),
            )
            self._log("job.failed", job_id=job.id, model_id=job.model_id, error=str(e))
            return

        # Rescore since this model is now loaded
        self.rescore_model(job.model_id)

        # Mark running and dispatch
        self._store.update_state(job.id, "running", started_at=time.time())
        self._log("job.started", job_id=job.id, model_id=job.model_id, instance_id=instance_id)

        # Create cancel flag
        cancel_flag = asyncio.Event()
        self._cancel_flags[job.id] = cancel_flag

        # Dispatch to worker with the specific instance_id
        asyncio.create_task(self._worker.run_job(job, cancel_flag, instance_id=instance_id))

    async def _try_preload(self):
        """Speculatively pre-load the next job's model if budget allows."""
        full_models = self._get_full_models()
        next_job = self._store.pick_next_job(exclude_models=full_models)
        if next_job is None:
            return

        model_cfg = self._config.models.get(next_job.model_id)
        max_concurrent = model_cfg.max_concurrent if model_cfg else 1

        instance_id = self._memory.pick_instance(next_job.model_id, max_concurrent)
        if instance_id is None:
            return

        slot = self._memory.get_slot(instance_id)
        if slot is None:
            return

        from .memory import ModelState
        if slot.state in (ModelState.LOADED, ModelState.LOADING):
            return

        # Check if we can fit alongside current usage
        if self._memory.free_gb >= slot.memory_gb:
            logger.debug("Speculatively pre-loading %s", instance_id)
            try:
                await self._memory.ensure_loaded(instance_id)
                # Release immediately — we're just warming it up
                self._memory.release(instance_id)
                self.rescore_model(next_job.model_id)
            except Exception:
                pass  # Pre-load failures are non-fatal

    async def run(self):
        """Main scheduler loop.

        Each iteration: pick a job, load instance, dispatch, then kick off
        a background preload of the next instance so it's warming up while
        the current job runs.
        """
        self._running = True
        logger.info("Scheduler started")

        while self._running:
            try:
                full_models = self._get_full_models()
                job = self._store.pick_next_job(exclude_models=full_models if full_models else None)

                if job is None:
                    await asyncio.sleep(0.1)
                    continue

                await self._dispatch_job(job)

                # After dispatching, kick off background preload of the next
                # instance so it's ready when the current job finishes
                asyncio.create_task(self._try_preload())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduler error: %s", e, exc_info=True)
                await asyncio.sleep(1.0)

        logger.info("Scheduler stopped")

    def start(self):
        """Start the scheduler as a background task."""
        self._task = asyncio.create_task(self.run())

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    def request_cancel(self, job_id: str) -> bool:
        """Request cancellation of a running job."""
        if job_id in self._cancel_flags:
            self._cancel_flags[job_id].set()
            return True
        # Try to cancel in store (queued/scheduled)
        return self._store.cancel_job(job_id)

    def remove_cancel_flag(self, job_id: str):
        """Clean up cancel flag after job completes."""
        self._cancel_flags.pop(job_id, None)

    def _log(self, event: str, **kwargs):
        if self._logger:
            self._logger.log(event, **kwargs)
