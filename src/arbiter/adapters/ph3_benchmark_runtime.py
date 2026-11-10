"""Framework-free measurement and cancellation for the P-H3 worker.

No torch/Comfy import, listener or model is created by this module. The
caller supplies real runtime callbacks only inside the capped Arbiter worker.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import time
from typing import Callable

from arbiter.adapters.base import CancelledException


@dataclass
class BenchmarkObserver:
    """Implements Comfy's event sink while recording synchronized node phases."""
    synchronize: Callable[[], None] = lambda: None
    clock: Callable[[], float] = time.perf_counter
    client_id: str | None = None
    last_node_id: str | None = None
    last_prompt_id: str | None = None
    graph: dict = field(default_factory=dict)
    node_seconds: dict[str, float] = field(default_factory=dict)
    gpu_load_seconds: dict[str, float] = field(default_factory=dict)
    cached_nodes: set[str] = field(default_factory=set)
    _active_node: str | None = None
    _started: float = 0.0

    def begin(self, graph: dict, prompt_id: str) -> None:
        self.graph = graph
        self.client_id = prompt_id
        self.last_prompt_id = prompt_id
        self.node_seconds.clear()
        self.gpu_load_seconds.clear()
        self.cached_nodes.clear()
        self._active_node = None

    def _close(self) -> None:
        if self._active_node is not None:
            self.synchronize()
            elapsed = self.clock() - self._started
            self.node_seconds[self._active_node] = self.node_seconds.get(self._active_node, 0.0) + elapsed
            self._active_node = None

    def send_sync(self, event: str, data: dict, sid=None) -> None:
        if event == "execution_cached":
            self.cached_nodes.update(data.get("nodes", []))
        elif event == "executing":
            self._close()
            node = data.get("node")
            if node is not None:
                self.synchronize()
                self._active_node = node
                self._started = self.clock()
                self.last_node_id = node
        elif event in ("execution_success", "execution_error", "execution_interrupted"):
            self._close()
        elif event == "executed" and data.get("node") == self._active_node:
            # SaveImage emits a UI output immediately; no later bookkeeping is
            # accidentally assigned to its PNG-encoding duration.
            self._close()

    def phase_totals(self) -> dict:
        groups = {
            "UNETLoader": "cpu_weight_load_seconds", "CLIPLoader": "cpu_weight_load_seconds",
            "VAELoader": "cpu_weight_load_seconds", "PH3ImageConditioning": "text_encode_seconds",
            "SamplerCustomAdvanced": "denoise_seconds", "VAEDecode": "vae_decode_seconds",
            "SaveImage": "png_encode_write_seconds",
        }
        totals = {name: 0.0 for name in groups.values()}
        for node, duration in self.node_seconds.items():
            kind = self.graph.get(node, {}).get("class_type")
            if kind in groups:
                # Model transfers/manager work inside encoding/sampling/decode
                # are measured separately rather than labelled inference.
                transfer = self.gpu_load_seconds.get(node, 0.0)
                totals[groups[kind]] += max(0.0, duration - transfer)
        totals["gpu_model_load_calls_seconds"] = sum(self.gpu_load_seconds.values())
        totals["generation_excluding_model_load_and_png_seconds"] = sum(
            totals[k] for k in ("text_encode_seconds", "denoise_seconds", "vae_decode_seconds")
        )
        return {k: round(v, 6) for k, v in totals.items()}

    @contextmanager
    def measure_gpu_load_calls(self, management):
        """Instrument only this worker's manager; restore it on all outcomes."""
        original = management.load_models_gpu
        depth = 0
        def measured(*args, **kwargs):
            nonlocal depth
            outer = depth == 0
            node = self._active_node
            if outer:
                self.synchronize()
                started = self.clock()
            depth += 1
            try:
                return original(*args, **kwargs)
            finally:
                depth -= 1
                if outer:
                    self.synchronize()
                    self.gpu_load_seconds[node or "outside_node"] = self.gpu_load_seconds.get(node or "outside_node", 0.0) + self.clock() - started
        management.load_models_gpu = measured
        try:
            yield
        finally:
            management.load_models_gpu = original


def run_cancellable(operation: Callable, cancel_flag: threading.Event, interrupt: Callable[[], None]):
    """Re-signal while cancelled so Comfy's entry reset cannot lose a cancel.

    The watchdog is joined before the worker lock is released: it cannot
    leave an interrupt flag racing onto a subsequent request.
    """
    if cancel_flag.is_set():
        raise CancelledException("P-H3 cancelled before execution")
    done = threading.Event()
    def watch():
        while not done.wait(0.025):
            if cancel_flag.is_set():
                interrupt()
    watcher = threading.Thread(target=watch, name="ph3-cancel", daemon=True)
    watcher.start()
    try:
        result = operation()
        if cancel_flag.is_set():
            raise CancelledException("P-H3 cancelled during execution")
        return result
    finally:
        done.set()
        watcher.join()
