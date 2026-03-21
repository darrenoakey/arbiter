#!/usr/bin/env python3
"""Adapter subprocess worker for Arbiter Go server.

Communicates with the Go parent process via newline-delimited JSON:
  stdin:  commands from Go
  stdout: responses to Go (JSON only)
  stderr: logging (free-form, captured by Go)

Commands:
  {"cmd":"load","device":"cuda"}
    -> {"status":"ok","vram_bytes":N}
  {"cmd":"infer","params":{...},"output_dir":"/path","job_id":"abc","job_type":"caption"}
    -> {"status":"ok","result":{...}}
    -> {"status":"error","error":"msg"}
    -> {"status":"cancelled"}
  {"cmd":"infer","req_id":"abc","params":{...},...}  (concurrent mode)
    -> {"status":"ok","req_id":"abc","result":{...}}
  {"cmd":"unload"}
    -> {"status":"ok"}
  {"cmd":"shutdown"}
    -> {"status":"ok"} then exit

Cancel: parent sends SIGUSR1 to set cancel_flag during inference.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path

# Redirect all logging to stderr so stdout stays clean for protocol
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("arbiter.worker")


def respond(obj: dict):
    """Write a JSON response to stdout."""
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m arbiter.worker_main <model_id>", file=sys.stderr)
        sys.exit(1)

    model_id = sys.argv[1]
    log.info("Worker starting for model: %s", model_id)

    # Import adapter registry and create adapter
    from arbiter.adapters import registry
    adapter_cls = registry.get_adapter_class(model_id)
    adapter = adapter_cls()

    # Cancel flag — set by SIGUSR1
    cancel_flag = threading.Event()

    def handle_cancel(signum, frame):
        cancel_flag.set()
        log.info("Cancel signal received")

    signal.signal(signal.SIGUSR1, handle_cancel)

    # For concurrent mode: read stdin in a background thread
    # so cancel and new infer commands can arrive during inference.
    max_concurrent = int(os.environ.get("ARBITER_MAX_CONCURRENT", "1"))
    if max_concurrent > 1:
        _run_concurrent(adapter, cancel_flag, max_concurrent)
    else:
        _run_single(adapter, cancel_flag)


def _run_single(adapter, cancel_flag: threading.Event):
    """Single-job mode: read command, execute, respond, repeat."""
    from arbiter.adapters.base import CancelledException

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
            device = msg.get("device", "cuda")
            try:
                adapter.load(device)
                # Report VRAM usage
                vram_bytes = _get_vram_bytes()
                respond({"status": "ok", "vram_bytes": vram_bytes})
            except Exception as e:
                log.exception("Load failed")
                respond({"status": "error", "error": str(e)})

        elif cmd == "infer":
            cancel_flag.clear()
            params = msg.get("params", {})
            output_dir = Path(msg.get("output_dir", "/tmp"))
            job_type = msg.get("job_type", "")
            req_id = msg.get("req_id", "")

            # Inject _job_type so adapters can dispatch
            if isinstance(params, dict):
                params["_job_type"] = job_type
            elif isinstance(params, str):
                params = json.loads(params)
                params["_job_type"] = job_type

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

    log.info("Worker shutting down")


def _run_concurrent(adapter, cancel_flag: threading.Event, max_concurrent: int):
    """Concurrent mode: multiple infers can run simultaneously via thread pool."""
    import queue
    from concurrent.futures import ThreadPoolExecutor
    from arbiter.adapters.base import CancelledException

    executor = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="infer")
    cmd_queue = queue.Queue()

    # Background thread reads stdin
    def _reader():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Cancel commands are handled immediately
            if msg.get("cmd") == "cancel":
                cancel_flag.set()
                continue

            cmd_queue.put(msg)
        cmd_queue.put(None)  # sentinel

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    def _do_infer(msg):
        params = msg.get("params", {})
        if isinstance(params, str):
            params = json.loads(params)
        output_dir = Path(msg.get("output_dir", "/tmp"))
        job_type = msg.get("job_type", "")
        req_id = msg.get("req_id", "")
        params["_job_type"] = job_type

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

    while True:
        msg = cmd_queue.get()
        if msg is None:
            break

        cmd = msg.get("cmd")
        if cmd == "load":
            try:
                adapter.load(msg.get("device", "cuda"))
                respond({"status": "ok", "vram_bytes": _get_vram_bytes()})
            except Exception as e:
                respond({"status": "error", "error": str(e)})
        elif cmd == "infer":
            executor.submit(_do_infer, msg)
        elif cmd == "unload":
            try:
                adapter.unload()
                respond({"status": "ok"})
            except Exception as e:
                respond({"status": "error", "error": str(e)})
        elif cmd == "shutdown":
            respond({"status": "ok"})
            break
        elif cmd == "ping":
            respond({"status": "ok"})

    executor.shutdown(wait=False)
    log.info("Worker shutting down")


def _get_vram_bytes() -> int:
    """Get current CUDA VRAM usage for this process."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    main()
