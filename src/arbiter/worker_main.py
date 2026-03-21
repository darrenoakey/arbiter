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

    from arbiter.adapters import registry
    from arbiter.adapters.base import CancelledException

    adapter_cls = registry.get_adapter_class(model_id)
    adapter = adapter_cls()

    cancel_flag = threading.Event()
    signal.signal(signal.SIGUSR1, lambda *_: (cancel_flag.set(), log.info("Cancel signal received")))

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
            respond({"status": "error", "req_id": req_id, "error": f"{type(e).__name__}: {e}"})

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
