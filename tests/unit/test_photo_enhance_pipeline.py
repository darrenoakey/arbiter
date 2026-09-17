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

from arbiter.adapters.photo_enhance_impl.detail import (
    OVERLAP,
    SCALE,
    TILE,
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
