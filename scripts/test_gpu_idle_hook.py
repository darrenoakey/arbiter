#!/usr/bin/env python3
"""Stdlib tests for scripts/gpu_idle_hook.py."""

from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_idle_hook  # noqa: E402


class FakeAgentd3(BaseHTTPRequestHandler):
    conversations: list[dict] = []
    messages: list[dict] = []

    def log_message(self, fmt: str, *args) -> None:
        return

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode())

    def _write(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        payload = self._read_json()
        if self.path == "/v1/conversations":
            self.conversations.append(payload)
            self._write(200, {"conversation_id": "conv-idle-1", "status": "running"})
            return
        if self.path == "/v1/conversations/conv-idle-1/messages":
            self.messages.append(payload)
            self._write(200, {"ok": True})
            return
        self._write(404, {"error": self.path})


class GpuIdleHookTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeAgentd3.conversations = []
        FakeAgentd3.messages = []
        self.agentd = ThreadingHTTPServer(("127.0.0.1", 0), FakeAgentd3)
        self.agentd_thread = threading.Thread(target=self.agentd.serve_forever, daemon=True)
        self.agentd_thread.start()
        agentd_url = f"http://127.0.0.1:{self.agentd.server_address[1]}"
        cfg = gpu_idle_hook.HookConfig("127.0.0.1:0", agentd_url, {"127.0.0.1"})
        self.hook = gpu_idle_hook.HookServer(("127.0.0.1", 0), gpu_idle_hook.make_handler(cfg))
        self.hook_thread = threading.Thread(target=self.hook.serve_forever, daemon=True)
        self.hook_thread.start()
        self.hook_url = f"http://127.0.0.1:{self.hook.server_address[1]}"

    def tearDown(self) -> None:
        self.hook.shutdown()
        self.hook.server_close()
        self.agentd.shutdown()
        self.agentd.server_close()

    def test_investigate_opens_agentd3_conversation(self) -> None:
        report = {
            "prompt": "investigate job abc123; logs at /mnt/arbiter-store/output/logs/",
            "conversation": {
                "model": "agentic-high",
                "cwd": "/Users/darrenoakey/src/arbiter",
                "policy": "yolo",
                "title": "GPU idle kill: abc123 ltx2",
                "source": "arbiter-gpu-idle-watchdog",
                "idempotency_key": "gpu-idle-abc123-1",
                "origin": {"kind": "service", "actor": "arbiter-gpu-idle-watchdog", "ref": "abc123"},
            },
        }
        req = urllib.request.Request(
            self.hook_url + "/investigate",
            data=json.dumps(report).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode())
        self.assertEqual(body["conversation_id"], "conv-idle-1")
        self.assertEqual(FakeAgentd3.conversations[0]["model"], "agentic-high")
        self.assertEqual(FakeAgentd3.conversations[0]["origin"]["kind"], "service")
        self.assertIn("abc123", FakeAgentd3.messages[0]["text"])

    def test_healthz(self) -> None:
        with urllib.request.urlopen(self.hook_url + "/healthz", timeout=5) as resp:
            body = json.loads(resp.read().decode())
        self.assertTrue(body["ok"])

    def test_unknown_path_is_404(self) -> None:
        req = urllib.request.Request(self.hook_url + "/nope", data=b"{}", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(caught.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
