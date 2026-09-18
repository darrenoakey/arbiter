#!/usr/bin/env python3
"""Laptop-side receiver for arbiter GPU-idle kill reports.

Spark cannot reach agentd3 (loopback-only on 127.0.0.1:8620). This process
listens on the LAN, allowlists spark, and opens an agentd3 investigation.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEFAULT_LISTEN = "0.0.0.0:8655"
DEFAULT_AGENTD3 = "http://127.0.0.1:8620"
ALLOWED_HOSTS = {"10.0.0.254", "127.0.0.1", "::1", "10.0.0.44"}


class HookConfig:
    def __init__(self, listen: str, agentd3: str, allowed: set[str]):
        self.listen = listen
        self.agentd3 = agentd3.rstrip("/")
        self.allowed = allowed


def split_host_port(listen: str) -> tuple[str, int]:
    host, _, port_s = listen.rpartition(":")
    if not host or not port_s:
        raise ValueError(f"listen must be host:port, got {listen!r}")
    return host, int(port_s)


def client_host(addr: str) -> str:
    if addr.startswith("::ffff:"):
        return addr[7:]
    return addr


def agentd3_request(base: str, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read()
    if not body:
        return {}
    return json.loads(body.decode())


def open_investigation(cfg: HookConfig, report: dict[str, Any]) -> dict[str, Any]:
    conv = report.get("conversation") or {}
    prompt = report.get("prompt") or ""
    if not prompt:
        raise ValueError("report missing prompt")
    created = agentd3_request(
        cfg.agentd3,
        "POST",
        "/v1/conversations",
        {
            "model": conv.get("model") or "agentic-high",
            "cwd": conv.get("cwd") or "/Users/darrenoakey/src/arbiter",
            "policy": conv.get("policy") or "yolo",
            "title": conv.get("title") or "GPU idle kill",
            "source": conv.get("source") or "arbiter-gpu-idle-watchdog",
            "idempotency_key": conv.get("idempotency_key"),
            "origin": conv.get("origin") or {"kind": "service", "actor": "arbiter-gpu-idle-watchdog"},
        },
    )
    conversation_id = created.get("conversation_id")
    if not conversation_id:
        raise ValueError(f"agentd3 create missing conversation_id: {created}")
    agentd3_request(
        cfg.agentd3,
        "POST",
        f"/v1/conversations/{conversation_id}/messages",
        {"text": prompt},
    )
    return {"conversation_id": conversation_id, "status": "started"}


def make_handler(cfg: HookConfig):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _deny(self, code: int, msg: str) -> None:
            body = json.dumps({"error": msg}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _ok(self, payload: dict[str, Any], code: int = 202) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/investigate":
                self._deny(404, "not found")
                return
            host = client_host(self.client_address[0])
            if host not in cfg.allowed:
                self._deny(403, f"forbidden source {host}")
                return
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                report = json.loads(raw.decode())
                result = open_investigation(cfg, report)
            except Exception as exc:  # noqa: BLE001 — surface any forward failure
                self._deny(502, str(exc))
                return
            self._ok(result)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/healthz":
                self._deny(404, "not found")
                return
            self._ok({"ok": True}, code=200)

    return Handler


class HookServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def parse_args(argv: list[str]) -> HookConfig:
    listen = DEFAULT_LISTEN
    agentd3 = DEFAULT_AGENTD3
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--listen" and args:
            listen = args.pop(0)
        elif arg == "--agentd3" and args:
            agentd3 = args.pop(0)
        elif arg in ("-h", "--help"):
            print("usage: gpu_idle_hook.py [--listen HOST:PORT] [--agentd3 URL]")
            raise SystemExit(0)
        else:
            raise SystemExit(f"unknown arg: {arg}")
    return HookConfig(listen, agentd3, set(ALLOWED_HOSTS))


def serve(cfg: HookConfig) -> None:
    host, port = split_host_port(cfg.listen)
    httpd = HookServer((host, port), make_handler(cfg))
    bound = httpd.server_address
    print(f"gpu-idle-hook listening on {bound[0]}:{bound[1]} -> {cfg.agentd3}", flush=True)
    httpd.serve_forever()


def main() -> None:
    serve(parse_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
