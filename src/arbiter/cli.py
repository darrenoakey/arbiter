"""Arbiter CLI — communicate with the running server."""
from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_URL = "http://localhost:8400"


def _server_url() -> str:
    return os.environ.get("ARBITER_URL", DEFAULT_URL)


def _request(method: str, path: str, data: dict | None = None) -> dict:
    """Make an HTTP request to the Arbiter server."""
    url = f"{_server_url()}{path}"
    body = json.dumps(data).encode() if data else None
    req = Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except URLError as e:
        print(f"Error: Could not connect to Arbiter at {_server_url()}", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ps(args):
    """Show loaded models and VRAM usage."""
    data = _request("GET", "/v1/ps")

    print(f"VRAM: {data['vram_used_gb']:.1f} / {data['vram_budget_gb']:.0f} GB")
    print()

    models = data.get("models", [])
    if not models:
        print("No models registered.")
        return

    # Header
    print(f"{'MODEL':<20} {'STATE':<10} {'VRAM':<8} {'ACTIVE':<8} {'QUEUED':<8} {'IDLE':<10}")
    print("-" * 74)
    for m in models:
        idle = f"{m['idle_seconds']:.0f}s" if m.get("idle_seconds") is not None else "-"
        print(
            f"{m['id']:<20} {m['state']:<10} {m['memory_gb']:<8.1f} "
            f"{m.get('active_jobs', 0):<8} {m.get('queued_jobs', 0):<8} {idle:<10}"
        )

    print()
    queue = data.get("queue", {})
    parts = [f"{k}: {v}" for k, v in sorted(queue.items())]
    print(f"Queue: {', '.join(parts) if parts else 'empty'}")


def cmd_jobs(args):
    """List jobs."""
    params = []
    if args.state:
        params.append(f"state={args.state}")
    if args.model:
        params.append(f"model={args.model}")
    if args.limit:
        params.append(f"limit={args.limit}")

    query = "&".join(params)
    path = f"/v1/jobs?{query}" if query else "/v1/jobs"
    jobs = _request("GET", path)

    if not jobs:
        print("No jobs found.")
        return

    print(f"{'JOB ID':<14} {'TYPE':<18} {'MODEL':<16} {'STATUS':<12} {'CREATED':<12}")
    print("-" * 72)
    for j in jobs:
        from datetime import datetime, timezone
        created = datetime.fromtimestamp(j["created_at"], tz=timezone.utc).strftime("%H:%M:%S")
        print(
            f"{j['job_id']:<14} {j['type']:<18} {j['model']:<16} "
            f"{j['status']:<12} {created:<12}"
        )


def cmd_submit(args):
    """Submit a job."""
    try:
        params = json.loads(args.params) if args.params else {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON params: {e}", file=sys.stderr)
        sys.exit(1)

    data = {"type": args.type, "params": params}
    result = _request("POST", "/v1/jobs", data)
    print(f"Job submitted: {result['job_id']}")
    print(f"  Model: {result['model']}")
    print(f"  Status: {result['status']}")
    if result.get("estimated_seconds"):
        print(f"  Estimated: {result['estimated_seconds']:.1f}s")


def cmd_status(args):
    """Get job status."""
    data = _request("GET", f"/v1/jobs/{args.job_id}")
    print(f"Job: {data['job_id']}")
    print(f"  Model: {data['model']}")
    print(f"  Status: {data['status']}")
    if data.get("error"):
        print(f"  Error: {data['error']}")
    if data.get("started_at"):
        print(f"  Started: {data['started_at']}")
    if data.get("finished_at"):
        print(f"  Finished: {data['finished_at']}")
    if data.get("result"):
        result = data["result"]
        # Don't print base64 data
        display = {k: v for k, v in result.items() if k != "data"}
        if display:
            print(f"  Result: {json.dumps(display, indent=2)}")
        if "data" in result:
            print(f"  Data: ({len(result['data'])} bytes base64)")


def cmd_cancel(args):
    """Cancel a job."""
    data = _request("DELETE", f"/v1/jobs/{args.job_id}")
    print(f"Job {args.job_id}: {data.get('status', data.get('message', 'cancelled'))}")


def cmd_health(args):
    """Check server health."""
    data = _request("GET", "/v1/health")
    print(f"Status: {data['status']}")
    print(f"Uptime: {data['uptime_seconds']:.0f}s")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="arbiter", description="Arbiter CLI")
    parser.add_argument("--server", help="Server URL (default: $ARBITER_URL or http://localhost:8400)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("ps", help="Show loaded models and VRAM")
    sub.add_parser("health", help="Check server health")

    p_jobs = sub.add_parser("jobs", help="List jobs")
    p_jobs.add_argument("--state", help="Filter by state (comma-separated)")
    p_jobs.add_argument("--model", help="Filter by model")
    p_jobs.add_argument("--limit", type=int, default=50)

    p_submit = sub.add_parser("submit", help="Submit a job")
    p_submit.add_argument("type", help="Job type (e.g., image-generate, transcribe)")
    p_submit.add_argument("params", nargs="?", default="{}", help="JSON params")

    p_status = sub.add_parser("status", help="Get job status")
    p_status.add_argument("job_id", help="Job ID")

    p_cancel = sub.add_parser("cancel", help="Cancel a job")
    p_cancel.add_argument("job_id", help="Job ID")

    args = parser.parse_args(argv)

    if args.server:
        os.environ["ARBITER_URL"] = args.server

    commands = {
        "ps": cmd_ps,
        "jobs": cmd_jobs,
        "submit": cmd_submit,
        "status": cmd_status,
        "cancel": cmd_cancel,
        "health": cmd_health,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
