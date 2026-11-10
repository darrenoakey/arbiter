"""Real framework-free event accounting and cancellation; no GPU/services."""
import threading
from types import SimpleNamespace

import pytest

from arbiter.adapters.base import CancelledException
from arbiter.adapters.ph3_benchmark_runtime import BenchmarkObserver, run_cancellable
from arbiter.adapters.p_h3_image_comfy import PH3Request, build_graph, graph_node_id


def test_repeated_seed_recomputes_required_nodes_but_keeps_loader_ids():
    from pathlib import Path
    request = PH3Request("same literal prompt", 42001, 1024, 1024)
    a, b = build_graph(request, Path(".")), build_graph(request, Path("."))
    for name in ("PH3ImageConditioning", "SamplerCustomAdvanced", "SaveImage"):
        assert graph_node_id(a, name) != graph_node_id(b, name)
    for name in ("UNETLoader", "CLIPLoader", "VAELoader"):
        assert graph_node_id(a, name) == graph_node_id(b, name)
    assert a["6"]["inputs"]["noise_seed"] == b["6"]["inputs"]["noise_seed"] == 42001


def test_synchronized_phase_records_do_not_mix_png_or_load_into_generation():
    ticks = iter([0.0, 1.0, 1.0, 5.0, 5.0, 5.5])
    observer = BenchmarkObserver(clock=lambda: next(ticks))
    graph = {"e": {"class_type": "PH3ImageConditioning"},
             "s": {"class_type": "SamplerCustomAdvanced"}, "p": {"class_type": "SaveImage"}}
    observer.begin(graph, "own-job")
    observer.send_sync("executing", {"node": "e"})
    observer.send_sync("executing", {"node": "s"})
    observer.send_sync("executing", {"node": "p"})
    observer.send_sync("executed", {"node": "p"})
    observer.gpu_load_seconds["s"] = 0.75
    totals = observer.phase_totals()
    assert totals["text_encode_seconds"] == 1.0
    assert totals["denoise_seconds"] == 3.25
    assert totals["png_encode_write_seconds"] == 0.5
    assert totals["gpu_model_load_calls_seconds"] == 0.75
    assert totals["generation_excluding_model_load_and_png_seconds"] == 4.25


def test_load_measurement_restores_callback_even_when_operation_raises():
    ticks = iter([3.0, 4.5])
    observer = BenchmarkObserver(clock=lambda: next(ticks))
    def operation():
        raise ValueError("own operation")
    manager = SimpleNamespace(load_models_gpu=operation)
    with pytest.raises(ValueError, match="own operation"):
        with observer.measure_gpu_load_calls(manager):
            manager.load_models_gpu()
    assert manager.load_models_gpu is operation
    assert observer.gpu_load_seconds == {"outside_node": 1.5}


def test_cancellation_is_signalled_while_operation_runs_and_watcher_is_joined():
    cancel = threading.Event()
    started = threading.Event()
    interrupted = threading.Event()
    def request_cancel():
        assert started.wait(1)
        cancel.set()
    sender = threading.Thread(target=request_cancel)
    sender.start()
    def operation():
        started.set()
        assert interrupted.wait(2), "cancel was not delivered during the operation"
    with pytest.raises(CancelledException):
        run_cancellable(operation, cancel, interrupted.set)
    sender.join()
    assert not any(t.name == "ph3-cancel" and t.is_alive() for t in threading.enumerate())


def test_precancel_does_not_start_operation():
    cancel = threading.Event()
    cancel.set()
    called = []
    with pytest.raises(CancelledException):
        run_cancellable(lambda: called.append(True), cancel, lambda: None)
    assert called == []
