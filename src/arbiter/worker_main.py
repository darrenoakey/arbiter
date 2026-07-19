#!/usr/bin/env python3
"""Adapter subprocess worker for Arbiter.

Reads JSON commands from stdin, writes JSON responses to stdout.
Logs go to stderr. Infer commands run in a thread pool so the main
thread stays free to read cancel signals.

The worker has no concept of concurrency configuration — it just
processes whatever arrives. The Go server decides how many infer
commands to send concurrently.

Cancel: parent sends SIGUSR1 to set cancel_flag during inference.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("arbiter.worker")

# Thread-safe stdout writing
_write_lock = threading.Lock()


def respond(obj: dict):
    """Write a JSON response to stdout (thread-safe)."""
    line = json.dumps(obj, default=str) + "\n"
    with _write_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m arbiter.worker_main <model_id>", file=sys.stderr)
        sys.exit(1)

    model_id = sys.argv[1]
    log.info("Worker starting for model: %s", model_id)

    from arbiter.image_policy import require_still_image_disabled

    # Run before the CUDA cap and adapter import so a direct invocation cannot
    # import a pipeline or touch disabled model weights.
    require_still_image_disabled(model_id)

    _apply_cuda_memory_cap()

    from arbiter.adapters import registry
    from arbiter.adapters.base import CancelledException

    adapter_cls = registry.get_adapter_class(model_id)
    adapter = adapter_cls()

    cancel_flag = threading.Event()
    signal.signal(
        signal.SIGUSR1,
        lambda *_: (cancel_flag.set(), log.info("Cancel signal received")),
    )

    executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="infer")

    def do_infer(msg):
        cancel_flag.clear()
        params = msg.get("params", {})
        if isinstance(params, str):
            params = json.loads(params)
        output_dir = Path(msg.get("output_dir", "/tmp"))
        req_id = msg.get("req_id", "")
        params["_job_type"] = msg.get("job_type", "")

        try:
            result = adapter.infer(params, output_dir, cancel_flag)
            if cancel_flag.is_set():
                respond({"status": "cancelled", "req_id": req_id})
            else:
                respond({"status": "ok", "req_id": req_id, "result": result})
        except CancelledException:
            respond({"status": "cancelled", "req_id": req_id})
        except Exception as e:
            log.exception("Infer failed")
            respond(
                {
                    "status": "error",
                    "req_id": req_id,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    # Main loop: read stdin, dispatch commands
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            log.warning("Invalid JSON: %s", line)
            continue

        cmd = msg.get("cmd")

        if cmd == "load":
            try:
                adapter.load(msg.get("device", "cuda"))
                respond({"status": "ok", "vram_bytes": _get_vram_bytes()})
            except Exception as e:
                log.exception("Load failed")
                respond({"status": "error", "error": str(e)})

        elif cmd == "infer":
            executor.submit(do_infer, msg)

        elif cmd == "unload":
            try:
                adapter.unload()
                respond({"status": "ok"})
            except Exception as e:
                log.exception("Unload failed")
                respond({"status": "error", "error": str(e)})

        elif cmd == "shutdown":
            respond({"status": "ok"})
            break

        elif cmd == "ping":
            respond({"status": "ok"})

        else:
            respond({"status": "error", "error": f"unknown command: {cmd}"})

    executor.shutdown(wait=False)
    log.info("Worker shutting down")


def _apply_cuda_memory_cap():
    """Cap this worker's CUDA allocator to its declared footprint.

    On the GB10 unified-memory host, GPU allocations bypass cgroup accounting,
    so an adapter that overshoots its budget exhausts physical RAM and livelocks
    the whole machine (no OOM-killer record, requires a physical reset). The Go
    server passes the model's declared budget in ARBITER_MEMORY_GB; we set
    torch.cuda.set_per_process_memory_fraction so an overshoot raises a catchable
    ``CUDA out of memory`` and fails just this job instead of taking down the host.

    A 15% pad over the declared reservation is allowed so legitimate slight
    overshoot does not turn into a false OOM — the cap exists to stop the
    catastrophic runaway, not to enforce the budget to the byte. The fraction is
    hard-ceilinged at 0.92 so the host always keeps a reserve.
    """
    import os

    mem_gb = os.environ.get("ARBITER_MEMORY_GB")
    if not mem_gb:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        total = torch.cuda.get_device_properties(0).total_memory
        want = float(mem_gb) * (1024**3) * 1.15
        fraction = max(0.01, min(0.92, want / total))
        torch.cuda.set_per_process_memory_fraction(fraction)
        log.info(
            "CUDA memory cap applied: declared=%.1f GB -> fraction %.3f (%.1f GB of %.1f GB)",
            float(mem_gb),
            fraction,
            fraction * total / (1024**3),
            total / (1024**3),
        )
    except Exception:
        log.exception("Failed to apply CUDA memory cap (continuing uncapped)")


def _get_vram_bytes() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    main()
