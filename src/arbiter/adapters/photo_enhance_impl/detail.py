"""Detail recovery tiling, ported from photo-enhance's detail.py.

The tile planning, feathered blend, and Lanczos resample are identical to
the Mac implementation; only the executor differs: instead of shelling out
to an mflux process, each tile goes through the in-process SeedVR2 CUDA
runner the adapter provides. One SeedVR2 run at a time is guaranteed by
the arbiter (one photo-enhance worker, max_instances=1), so the
machine-wide flock is not needed here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

TILE = 1536
OVERLAP = 128
SCALE = 2
DIVISIBILITY = 16


# ##################################################################
# tile
@dataclass(frozen=True)
class Tile:
    index: int
    left: int
    top: int
    width: int
    height: int

    @property
    def name(self) -> str:
        return f"tile_{self.index:03d}"


# ##################################################################
# plan tiles
def plan_tiles(width: int, height: int, tile: int = TILE, overlap: int = OVERLAP) -> list[Tile]:
    tiles = []
    for top in axis_offsets(height, tile, overlap):
        for left in axis_offsets(width, tile, overlap):
            tiles.append(Tile(len(tiles), left, top, min(tile, width), min(tile, height)))
    return tiles


def axis_offsets(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    offsets = list(range(0, length - tile, stride))
    offsets.append(length - tile)
    return offsets


# ##################################################################
# align tile
# SeedVR2 needs /16 input; snap each planned window down to a multiple of
# 16 and, when the window sits flush with the image's far edge, shift it
# back so coverage is never lost. Returns a new Tile.
def align_tile(source_width: int, source_height: int, tile: Tile) -> Tile:
    width = tile.width - (tile.width % DIVISIBILITY)
    height = tile.height - (tile.height % DIVISIBILITY)
    left = tile.left
    top = tile.top
    if tile.left + tile.width >= source_width:
        left = tile.left + tile.width - width
    if tile.top + tile.height >= source_height:
        top = tile.top + tile.height - height
    return Tile(tile.index, left, top, width, height)


# ##################################################################
# recover detail
# generative texture pre-pass: SeedVR2 upscales each 1536 px tile 2x on
# spark's CUDA GPU; the upscaled tiles are feather-blended into a 2x frame
# and Lanczos-resampled back to native size, so the grade that follows
# works on recovered hair/foliage/fabric detail rather than phone mush.
def recover_detail(image: Image.Image, upscale_tile: Callable[[Image.Image], Image.Image]) -> Image.Image:
    source = image.convert("RGB")
    tiles = [align_tile(source.width, source.height, t) for t in plan_tiles(source.width, source.height)]
    upscaled_tiles = []
    for tile in tiles:
        crop = source.crop((tile.left, tile.top, tile.left + tile.width, tile.top + tile.height))
        log.info("seedvr2 2x on %s (%dx%d) ...", tile.name, tile.width, tile.height)
        upscaled = upscale_tile(crop)
        expected = (tile.width * SCALE, tile.height * SCALE)
        if upscaled.size != expected:
            raise RuntimeError(f"seedvr2 returned {upscaled.size} for {tile.name}, expected {expected}")
        upscaled_tiles.append(upscaled)
    result = assemble_tiles(tiles, upscaled_tiles, source.width * SCALE, source.height * SCALE)
    log.info("seedvr2 produced %dx%d; resampling back to %dx%d", *result.size, *source.size)
    return result.resize(source.size, Image.Resampling.LANCZOS)


# ##################################################################
# assemble tiles
def assemble_tiles(tiles: list[Tile], upscaled: list[Image.Image], width: int, height: int) -> Image.Image:
    total = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)
    band = OVERLAP * SCALE
    for tile, image in zip(tiles, upscaled, strict=True):
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        expected = (tile.height * SCALE, tile.width * SCALE)
        if pixels.shape[:2] != expected:
            raise RuntimeError(f"seedvr2 tile {tile.name} is {pixels.shape[1]}x{pixels.shape[0]}, expected {expected[1]}x{expected[0]}")
        ramp = feather(tile.height * SCALE, band)[:, None] * feather(tile.width * SCALE, band)[None, :]
        top, left = tile.top * SCALE, tile.left * SCALE
        total[top : top + expected[0], left : left + expected[1]] += pixels * ramp[:, :, None]
        weight[top : top + expected[0], left : left + expected[1]] += ramp[:, :, None]
    covered = (weight > 0)[..., 0]
    if not covered.any():
        raise RuntimeError("seedvr2 tiles do not cover the frame")
    # normalize only the covered pixels first: the border fill below must
    # replicate finished pixels, not un-normalized weighted sums.
    blended = np.zeros_like(total)
    blended[covered] = total[covered] / weight[covered]
    if not covered.all():
        # SeedVR2 snaps tiles to /16 px; when a single tile spans a whole
        # axis its flush-edge shift can expose a sub-16 px border on the
        # opposite side. Replicate the nearest covered pixels there — the
        # frame is Lanczos-resampled back to native size by the caller, so
        # the band is invisible.
        covered_rows = np.where(covered.any(axis=1))[0]
        covered_cols = np.where(covered.any(axis=0))[0]
        first_row, last_row = covered_rows[0], covered_rows[-1]
        first_col, last_col = covered_cols[0], covered_cols[-1]
        blended[:first_row] = blended[first_row]
        blended[last_row + 1 :] = blended[last_row]
        blended[:, :first_col] = blended[:, first_col : first_col + 1]
        blended[:, last_col + 1 :] = blended[:, last_col : last_col + 1]
    return Image.fromarray(np.clip(blended + 0.5, 0, 255).astype(np.uint8))


# ##################################################################
# feather
def feather(length: int, band: int) -> np.ndarray:
    band = max(1, min(band, length // 2))
    ramp = (np.arange(1, band + 1, dtype=np.float32) / (band + 1)).astype(np.float32)
    weights = np.ones(length, dtype=np.float32)
    weights[:band] = ramp
    weights[length - band :] = np.minimum(weights[length - band :], ramp[::-1])
    return weights
