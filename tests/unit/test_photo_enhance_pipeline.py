"""Unit tests for the photo-enhance pipeline pieces that run without a GPU.

The heavy-path correctness (SeedVR2 CUDA pass, arbiter sub-jobs, climb
against qwen3-vl) is verified by the bring-up probe job on spark — an
actual inference through the real stack — per the repo's adapter
doctrine. These tests pin the pure logic: masks (incl. the no-person
fallback that used to crash the climb), tile planning/blending, the
chat protocol translation, and JSON extraction.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from arbiter.adapters.photo_enhance_impl import detail as detail_module
from arbiter.adapters.photo_enhance_impl.detail import (
    OVERLAP,
    SCALE,
    TILE,
    Tile,
    assemble_tiles,
    axis_offsets,
    feather,
    plan_tiles,
)
from arbiter.adapters.photo_enhance_impl.masks import person_masks
from arbiter.adapters.photo_enhance_impl.vision_chat import VisionChat, parse_json_object


def test_photo_enhance_module_imports_without_scipy():
    """Regression: module-level imports broke every other adapter worker.

    Every worker boots through arbiter.adapters.__init__, which imports
    this adapter; a module-level `import scipy` (via the vendored
    operators) made ALL models fail to spawn with ModuleNotFoundError in
    venvs that lack photo-enhance's deps. Heavy imports must stay lazy.
    """
    import importlib
    import sys

    class _ScipyBlocker:
        def find_spec(self, name, path=None, target=None):
            if name == "scipy" or name.startswith("scipy."):
                raise ModuleNotFoundError(f"{name!r} blocked by regression test")
            return None

    blocker = _ScipyBlocker()
    sys.meta_path.insert(0, blocker)
    sys.modules.pop("arbiter.adapters.photo_enhance", None)
    try:
        module = importlib.import_module("arbiter.adapters.photo_enhance")
    finally:
        sys.meta_path.remove(blocker)
    assert module.PhotoEnhanceAdapter.model_id == "photo-enhance"
    from arbiter.adapters.registry import list_registered

    assert "photo-enhance" in list_registered()


def test_module_without_spec_breaks_find_spec_and_spec_fixes_it():
    """Regression: shim modules must carry a __spec__.

    types.ModuleType sets __spec__ to None, and importlib.util.find_spec
    raises ValueError for sys.modules entries without one — diffusers
    probes flash_attn exactly this way, which killed the worker load.
    """
    import importlib.util
    import sys
    import types

    mod = types.ModuleType("shimprobe_test")
    sys.modules["shimprobe_test"] = mod
    try:
        with pytest.raises(ValueError):
            importlib.util.find_spec("shimprobe_test")
        mod.__spec__ = importlib.util.spec_from_loader("shimprobe_test", loader=None)
        assert importlib.util.find_spec("shimprobe_test") is not None
    finally:
        del sys.modules["shimprobe_test"]


def test_person_masks_partition_a_real_cutout():
    cutout = Image.new("RGBA", (32, 24), (10, 20, 30, 0))  # transparent background
    for x in range(8, 24):  # person block in the middle
        for y in range(4, 20):
            cutout.putpixel((x, y), (200, 150, 100, 255))
    masks = person_masks(cutout)
    total = masks["person"] + masks["background"]
    assert np.allclose(total, 1.0, atol=1e-5)
    assert 0.02 < masks["person"].mean() < 0.8


def test_person_masks_without_a_person_cover_the_frame():
    empty = Image.new("RGBA", (8, 6), (10, 20, 30, 0))
    masks = person_masks(empty)
    # an empty person mask crashed the climb's segment crops; both
    # segments must fall back to the full frame so edits apply globally
    assert np.all(masks["person"] == 1.0)
    assert np.all(masks["background"] == 1.0)


def test_plan_tiles_covers_with_overlap_shifted_edges():
    tiles = plan_tiles(2 * TILE + 300, TILE)
    assert tiles[0].left == 0 and tiles[0].top == 0
    assert all(t.width == TILE and t.height == TILE for t in tiles)
    # last tile sits flush with the far edge
    assert max(t.left + t.width for t in tiles) == 2 * TILE + 300
    # and everything is covered
    covered = np.zeros((TILE, 2 * TILE + 300), dtype=bool)
    for t in tiles:
        covered[t.top : t.top + t.height, t.left : t.left + t.width] = True
    assert covered.all()


def test_axis_offsets_small_axis_is_single_window():
    assert axis_offsets(500, TILE, OVERLAP) == [0]


def test_assemble_tiles_blends_overlaps_seamlessly():
    # a linear gradient frame upscaled 2x: reconstruction must reproduce it
    # within rounding tolerance, including across the overlap band
    width, height = TILE + 400, TILE
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:, :, 0] = np.linspace(0, 255, width)[None, :]
    base[:, :, 1] = np.linspace(255, 0, height)[:, None]
    tiles = plan_tiles(width, height)
    upscaled = []
    for t in tiles:
        crop = Image.fromarray(base[t.top : t.top + t.height, t.left : t.left + t.width].astype(np.uint8), "RGB")
        crop = crop.resize((t.width * SCALE, t.height * SCALE), Image.Resampling.BILINEAR)
        upscaled.append(crop)
    result = np.asarray(
        assemble_tiles(tiles, upscaled, width * SCALE, height * SCALE), dtype=np.float32
    )
    expected = np.asarray(
        Image.fromarray(base.astype(np.uint8), "RGB").resize((width * SCALE, height * SCALE), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    assert np.abs(result - expected).max() <= 2.0


def test_recover_detail_aligns_odd_tiles_and_covers_the_frame():
    # awkward size: not a multiple of 16 on either axis, larger than one tile
    from arbiter.adapters.photo_enhance_impl.detail import DIVISIBILITY, recover_detail

    source = Image.new("RGB", (1600, 700), (30, 60, 90))
    seen = []

    def executor(crop: Image.Image) -> Image.Image:
        seen.append(crop.size)
        # the runner contract: exact 2x of the (already /16) crop
        return crop.resize((crop.width * SCALE, crop.height * SCALE), Image.Resampling.NEAREST)

    result = recover_detail(source, executor)
    assert result.size == source.size
    assert seen, "executor was never called"
    assert all(w % DIVISIBILITY == 0 and h % DIVISIBILITY == 0 for w, h in seen)


def test_assemble_tiles_fills_sub_16px_exposed_border():
    """Regression: /16-snapped tiles on a 1197 px-tall image expose a 13 px
    band the shifted tiles no longer cover; assemble must edge-fill it
    instead of raising, since the caller resamples back to native size."""
    tiles = [
        Tile(0, 0, 13, 1536, 1184),
        Tile(1, 56, 13, 1536, 1184),
    ]
    upscaled = [Image.new("RGB", (t.width * SCALE, t.height * SCALE), (90, 120, 30)) for t in tiles]
    width, height = 1592 * SCALE, 1197 * SCALE
    result = assemble_tiles(tiles, upscaled, width, height)
    assert result.size == (width, height)
    pixels = np.asarray(result)
    assert np.allclose(pixels[100, 100], (90, 120, 30), atol=2)
    assert np.allclose(pixels[5, 5], (90, 120, 30), atol=2)


def test_feather_is_never_zero_and_ramps_at_both_ends():
    weights = feather(100, 10)
    assert weights.min() > 0
    assert weights[0] < weights[9] < 1.0
    assert weights[-1] < weights[-10] < 1.0
    assert weights[50] == 1.0


def test_openai_turn_maps_images_to_data_url_parts():
    turn = {"role": "user", "content": "describe", "images": ["QUJD"]}
    mapped = VisionChat._openai_turn(turn)
    assert mapped["role"] == "user"
    parts = mapped["content"]
    assert parts[0] == {"type": "text", "text": "describe"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    text_only = VisionChat._openai_turn({"role": "assistant", "content": "hi"})
    assert text_only["content"] == [{"type": "text", "text": "hi"}]


def test_parse_json_object_handles_fences_and_think_tags():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('<think>reasoning</think>\n{"a": 1} trailing') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("no object here")


def test_parse_json_object_round_trips_operators_style_proposals():
    proposal = {"segment": "person", "tool": "fill_light", "amount": 0.4, "why": "face is dark"}
    wrapped = f'Here is my proposal:\n```json\n{json.dumps(proposal)}\n```\nDone.'
    assert parse_json_object(wrapped) == proposal


# ##################################################################
# test vision chat retries through a 5xx backend restart
# regression: qwen3-vl answers 500 while the vllm worker restarts
# (~5 min fleet-wide, several times a day); a climb turn must ride it
# out. Serves 500, 500, then 200 from a real loopback HTTP server.
def test_vision_chat_retries_5xx_then_succeeds():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from arbiter.adapters.photo_enhance_impl.settings import Settings
    from arbiter.adapters.photo_enhance_impl.vision_chat import VisionChat

    calls = {"count": 0}

    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.1 + Content-Length on every reply: a length-less close-delimited
        # body races the client's read and shows up as a flaky connection reset.
        protocol_version = "HTTP/1.1"

        def reply(self, code, body):
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls["count"] += 1
            if calls["count"] < 3:
                self.reply(500, b"backend restarting")
                return
            self.reply(200, json.dumps({"choices": [{"message": {"content": '{"a": 1}'}}]}).encode())

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        chat = VisionChat(Settings(arbiter_url=f"http://127.0.0.1:{server.server_port}"))
        chat.retry_backoff_seconds = 0.0
        chat.user("propose an edit")
        assert chat.ask() == '{"a": 1}'
        assert calls["count"] == 3
    finally:
        server.shutdown()
        server.server_close()


# ##################################################################
# test climb transcript is bounded to the served context window
# regression: job 0ba205f57358 (2026-09-18) grew the climb thread to
# 13885 input tokens at turn ~20; every later turn died with vllm's
# "maximum context length is 16384 tokens" error, and the 5xx path sat
# on 60s backoffs before failing a photo that had already cost 25
# minutes of GPU. A long thread must be trimmed before it is sent.
def test_fit_to_context_keeps_opening_and_newest_turns_within_budget():
    from arbiter.adapters.photo_enhance_impl.vision_chat import (
        DROPPED_TURNS_NOTE,
        VisionChat,
        estimate_tokens,
    )

    messages = [{"role": "user", "content": "OPENING RULES " + "r" * 4000}]
    for turn in range(40):
        messages.append({"role": "assistant", "content": f"proposal {turn} " + "p" * 900})
        messages.append({"role": "user", "content": f"verdict {turn} " + "v" * 900})
    messages[-1]["images"] = ["ZmFrZQ==", "ZmFrZQ==", "ZmFrZQ==", "ZmFrZQ=="]

    budget = VisionChat.prompt_budget()
    assert estimate_tokens(messages) > budget, "fixture must overflow the window"

    fitted = VisionChat.fit_to_context(messages, budget)
    assert estimate_tokens(fitted) <= budget
    assert fitted[0] is messages[0], "opening rules turn must survive"
    assert fitted[1]["content"] == DROPPED_TURNS_NOTE
    assert fitted[-1] is messages[-1], "newest turn must survive"
    assert fitted[-1]["images"] == messages[-1]["images"], "images on the newest turn are kept"
    assert len(fitted) < len(messages)


def test_fit_to_context_leaves_short_threads_untouched():
    from arbiter.adapters.photo_enhance_impl.vision_chat import VisionChat

    messages = [{"role": "user", "content": "short opening"}, {"role": "assistant", "content": "ok"}]
    assert VisionChat.fit_to_context(messages, VisionChat.prompt_budget()) == messages


def test_context_overflow_tokens_reads_vllm_validation_error():
    from arbiter.adapters.photo_enhance_impl.vision_chat import context_overflow_tokens

    detail = (
        "vllm.exceptions.VLLMValidationError: This model's maximum context length is 16384 tokens. "
        "However, you requested 2500 output tokens and your prompt contains at least 13885 input "
        "tokens, for a total of at least 16385 tokens. (parameter=input_tokens, value=13885)"
    )
    assert context_overflow_tokens(detail) == 1
    assert context_overflow_tokens("backend restarting") is None


# The backend's context rejection is permanent for that prompt size, so the
# climb must re-trim and retry IMMEDIATELY — never spend a 60s backoff on it.
def test_vision_chat_trims_and_retries_immediately_on_context_overflow():
    import json
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from arbiter.adapters.photo_enhance_impl.settings import Settings
    from arbiter.adapters.photo_enhance_impl.vision_chat import VisionChat

    seen: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def reply(self, code, body):
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(len(payload["messages"]))
            if len(seen) == 1:
                self.reply(
                    500,
                    b"This model's maximum context length is 16384 tokens. However, you requested "
                    b"2500 output tokens and your prompt contains at least 13885 input tokens, for "
                    b"a total of at least 16385 tokens.",
                )
                return
            self.reply(200, json.dumps({"choices": [{"message": {"content": '{"ok": 1}'}}]}).encode())

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        chat = VisionChat(Settings(arbiter_url=f"http://127.0.0.1:{server.server_port}"))
        chat.retry_backoff_seconds = 5.0  # a backoff here would be the bug
        chat.user("OPENING " + "o" * 3000)
        for turn in range(30):
            chat.assistant(f"proposal {turn} " + "p" * 900)
            chat.user(f"verdict {turn} " + "v" * 900)
        started = time.monotonic()
        assert chat.ask() == '{"ok": 1}'
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()

    assert len(seen) == 2, f"expected one trimmed retry, got {seen}"
    assert seen[1] < seen[0], "retry must send fewer turns"
    assert elapsed < 2.0, f"context retry waited {elapsed:.1f}s; it must not back off"
