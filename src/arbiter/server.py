"""Arbiter FastAPI server — unified GPU model serving."""
from __future__ import annotations

import asyncio
import base64
import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import ArbiterConfig, load_config, save_model_config_field
from .log import EventLogger
from .memory import MemoryManager
from .scheduler import Scheduler
from .schemas import (
    HealthResponse,
    JOB_TYPE_PARAMS,
    JOB_TYPE_TO_MODEL,
    JobState,
    JobSubmitRequest,
    JobSubmitResponse,
    JobStatusResponse,
    SystemStatus,
)
from .store import JobStore
from .worker import WorkerPool

logger = logging.getLogger(__name__)

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Global state (set during lifespan)
_config: ArbiterConfig = None
_store: JobStore = None
_memory: MemoryManager = None
_scheduler: Scheduler = None
_worker: WorkerPool = None
_event_logger: EventLogger = None
_executor: ThreadPoolExecutor = None
_io_executor: ThreadPoolExecutor = None  # dedicated pool for API handlers (never blocked by inference)
_ps_cache: dict = None  # cached /v1/ps snapshot, updated by background task
_start_time: float = 0


async def _run_io(fn, *args):
    """Run a blocking function in the IO executor so it never blocks the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_io_executor, functools.partial(fn, *args))


def _setup_adapters(config: ArbiterConfig, memory: MemoryManager):
    """Register all configured models with the memory manager.

    For models with max_instances > 1, creates separate adapter instances
    for each slot (e.g. moondream#0, moondream#1).  Each instance gets its
    own copy of model weights in VRAM.
    """
    from .adapters import registry

    for model_id, model_cfg in config.models.items():
        try:
            adapter_cls = registry.get_adapter_class(model_id)
        except KeyError:
            logger.warning("No adapter registered for model: %s (will fail on load)", model_id)
            continue

        n = model_cfg.max_instances
        for i in range(n):
            adapter = adapter_cls()
            instance_id = f"{model_id}#{i}" if n > 1 else None
            memory.register(
                model_id=model_id,
                adapter=adapter,
                memory_gb=model_cfg.memory_gb,
                keep_alive_s=model_cfg.keep_alive_seconds,
                max_concurrent=model_cfg.max_concurrent,
                instance_id=instance_id,
            )

        if n > 1:
            logger.info(
                "Registered %d instances for %s (%.1fGB each, %.1fGB total max)",
                n, model_id, model_cfg.memory_gb, n * model_cfg.memory_gb,
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _store, _memory, _scheduler, _worker, _event_logger, _executor, _io_executor, _start_time

    _start_time = time.time()
    _config = load_config(_PROJECT_ROOT)

    # Output dirs
    output_dir = _PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "jobs").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    # Logger
    _event_logger = EventLogger(output_dir / "logs")

    # Store
    db_path = output_dir / "arbiter.db"
    _store = JobStore(db_path)
    recovered = _store.recover_from_crash()
    if recovered:
        logger.info("Recovered %d jobs from crash", recovered)

    # Thread pools: inference pool for GPU work, IO pool for API handlers
    _executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="arbiter")
    _io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="arbiter-io")

    # Memory manager
    _memory = MemoryManager(
        budget_gb=_config.vram_budget_gb,
        executor=_executor,
        event_logger=_event_logger,
    )
    _setup_adapters(_config, _memory)

    # Worker pool
    def _on_job_done(job):
        _scheduler.remove_cancel_flag(job.id)
        _scheduler.rescore_model(job.model_id)

    _worker = WorkerPool(
        config=_config,
        store=_store,
        memory=_memory,
        executor=_executor,
        output_dir=output_dir,
        event_logger=_event_logger,
        on_job_done=_on_job_done,
    )

    # Scheduler
    _scheduler = Scheduler(
        config=_config,
        store=_store,
        memory=_memory,
        worker_pool=_worker,
        event_logger=_event_logger,
    )
    _scheduler.rescore_all()
    _scheduler.start()
    _memory.start_keepalive()

    _event_logger.log("server.start", vram_budget_gb=_config.vram_budget_gb, recovered_jobs=recovered)
    logger.info("Arbiter started on %s:%d (VRAM budget: %.0fGB)", _config.host, _config.port, _config.vram_budget_gb)

    # Background snapshot: updates _ps_cache every second in the IO pool
    # so /v1/ps always returns instantly without blocking
    async def _ps_update_loop():
        global _ps_cache
        log_counter = 0
        while True:
            try:
                snap = await _run_io(_build_ps_snapshot)
                _ps_cache = snap
                # Log to event log every 60 iterations (~60s)
                log_counter += 1
                if log_counter >= 60:
                    _event_logger.log("memory.snapshot", **snap)
                    log_counter = 0
            except Exception as e:
                logger.debug("PS snapshot update failed: %s", e)
            await asyncio.sleep(1.0)

    snapshot_task = asyncio.create_task(_ps_update_loop())

    yield

    # Shutdown
    snapshot_task.cancel()
    _scheduler.stop()
    _memory.stop_keepalive()
    _event_logger.log("server.stop", uptime_seconds=round(time.time() - _start_time, 1))
    _event_logger.close()
    _store.close()
    _executor.shutdown(wait=False)
    _io_executor.shutdown(wait=False)


app = FastAPI(title="Arbiter", version="0.1.0", lifespan=lifespan)


# --- Request schemas for management endpoints ---

class ModelUpdateRequest(BaseModel):
    max_instances: int


# --- Job endpoints ---

@app.post("/v1/jobs", status_code=202)
async def submit_job(req: JobSubmitRequest) -> JobSubmitResponse:
    job_type = req.type.value
    model_id = JOB_TYPE_TO_MODEL.get(job_type)
    if not model_id:
        raise HTTPException(400, f"Unknown job type: {job_type}")

    if model_id not in _config.models:
        raise HTTPException(400, f"Model not configured: {model_id}")

    # Validate params
    param_schema = JOB_TYPE_PARAMS.get(job_type)
    if param_schema:
        try:
            param_schema(**req.params)
        except Exception as e:
            raise HTTPException(400, f"Invalid params: {e}")

    priority = _scheduler.compute_priority(model_id)
    job = await _run_io(
        _store.create_job,
        model_id, job_type, req.params, priority,
    )

    model_cfg = _config.models[model_id]
    estimated = (model_cfg.avg_inference_ms + (model_cfg.load_ms if not _memory.is_loaded(model_id) else 0)) / 1000

    _event_logger.log("job.submitted", job_id=job.id, model_id=model_id, job_type=job_type, priority=priority)

    return JobSubmitResponse(
        job_id=job.id,
        status="queued",
        model=model_id,
        estimated_seconds=round(estimated, 1),
    )


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str) -> JobStatusResponse:
    job = await _run_io(_store.get_job, job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    result = job.result
    if job.state == "completed" and result:
        # Try to read result file and base64 encode it
        job_dir = _PROJECT_ROOT / "output" / "jobs" / job.id
        fmt = result.get("format", "")
        result_file = job_dir / f"result.{fmt}" if fmt else None
        if result_file and result_file.exists():
            data = result_file.read_bytes()
            result = {**result, "data": base64.b64encode(data).decode()}

    return JobStatusResponse(
        job_id=job.id,
        status=JobState(job.state),
        model=job.model_id,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        result=result,
    )


@app.delete("/v1/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = await _run_io(_store.get_job, job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    if job.state in ("completed", "failed", "cancelled"):
        return {"job_id": job_id, "status": job.state, "message": "Job already finished"}

    cancelled = _scheduler.request_cancel(job_id)
    if cancelled:
        return {"job_id": job_id, "status": "cancelled"}
    else:
        raise HTTPException(409, "Could not cancel job")


@app.get("/v1/jobs")
async def list_jobs(state: str | None = None, model: str | None = None, limit: int = 100):
    jobs = await _run_io(_store.list_jobs, state, model, limit)
    return [
        {
            "job_id": j.id,
            "type": j.job_type,
            "model": j.model_id,
            "status": j.state,
            "created_at": j.created_at,
            "started_at": j.started_at,
            "finished_at": j.finished_at,
        }
        for j in jobs
    ]


# --- Model management endpoints ---

@app.patch("/v1/models/{model_id}")
async def update_model(model_id: str, req: ModelUpdateRequest):
    """Update model configuration at runtime.

    Currently supports changing max_instances:
    - Scaling up adds new unloaded instances (loaded on demand).
    - Scaling down evicts idle instances immediately; active instances
      finish their current jobs then auto-evict.
    - Setting to 0 disables dispatch (queue waits until scaled back up).
    - Changes are persisted to config.json.
    """
    if model_id not in _config.models:
        raise HTTPException(404, f"Model not configured: {model_id}")

    if req.max_instances < 0:
        raise HTTPException(400, "max_instances must be >= 0")

    model_cfg = _config.models[model_id]
    old_max = model_cfg.max_instances
    new_max = req.max_instances

    if new_max == old_max:
        return {
            "model_id": model_id,
            "max_instances": new_max,
            "previous_max_instances": old_max,
            "added": 0,
            "removed": 0,
            "condemned": 0,
        }

    # Get adapter class for creating new instances (only needed for scale-up)
    from .adapters import registry
    try:
        adapter_cls = registry.get_adapter_class(model_id)
    except KeyError:
        raise HTTPException(500, f"No adapter registered for model: {model_id}")

    # Update in-memory config FIRST so scheduler sees new capacity immediately
    model_cfg.max_instances = new_max

    # Scale instances in memory manager
    scale_result = await _memory.scale_model(
        model_id=model_id,
        new_count=new_max,
        adapter_cls=adapter_cls,
        memory_gb=model_cfg.memory_gb,
        keep_alive_s=model_cfg.keep_alive_seconds,
        max_concurrent=model_cfg.max_concurrent,
    )

    # Persist to config file
    await _run_io(save_model_config_field, model_id, "max_instances", new_max, _PROJECT_ROOT)

    # Rescore priorities
    _scheduler.rescore_model(model_id)

    _event_logger.log(
        "model.scaled",
        model_id=model_id,
        old_max_instances=old_max,
        new_max_instances=new_max,
        **scale_result,
    )

    return {
        "model_id": model_id,
        "max_instances": new_max,
        "previous_max_instances": old_max,
        **scale_result,
    }


@app.delete("/v1/models/{model_id}/queue")
async def clear_model_queue(model_id: str):
    """Cancel all queued jobs for a specific model."""
    if model_id not in _config.models:
        raise HTTPException(404, f"Model not configured: {model_id}")

    cancelled = await _run_io(_store.cancel_queued_for_model, model_id)

    _event_logger.log("queue.cleared", model_id=model_id, cancelled=cancelled)

    return {
        "model_id": model_id,
        "cancelled": cancelled,
    }


# --- System status endpoints ---

def _build_ps_snapshot() -> dict:
    """Build the full /v1/ps response. Called from IO thread pool."""
    snap = _memory.snapshot()
    counts = _store.count_by_state()
    for model_info in snap["models"]:
        mid = model_info["id"]
        queued = _store.count_by_state(model_id=mid)
        model_info["queued_jobs"] = queued.get("queued", 0)
        # Add configured max_instances so API consumers see target vs actual
        model_cfg = _config.models.get(mid)
        if model_cfg:
            model_info["max_instances"] = model_cfg.max_instances
    snap["queue"] = counts
    return snap


@app.get("/v1/ps")
async def system_status() -> dict:
    if _ps_cache is not None:
        return _ps_cache
    # Fallback before first background update
    return await _run_io(_build_ps_snapshot)


@app.get("/v1/health")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


def main():
    """Entry point for `python -m arbiter.server`."""
    config = load_config(_PROJECT_ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(
        "arbiter.server:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
