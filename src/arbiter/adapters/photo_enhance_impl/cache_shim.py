from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from pathlib import Path

from PIL import Image


# ##################################################################
# content cache
# record/replay store for expensive real calls (SeedVR2, reference renders,
# vision replies). The key is a SHA-256 of every input that determines the
# output, so an identical request replays instantly and anything new is
# computed for real. Entries recorded for the test sample are committed so
# the per-file test gate stays fast on a fresh checkout.
class ContentCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    # ##################################################################
    # key
    # hash a namespace plus arbitrary JSON-able parts and raw byte blobs.
    @staticmethod
    def key(namespace: str, parts: object, blobs: list[bytes] = ()) -> str:
        digest = hashlib.sha256(namespace.encode())
        digest.update(json.dumps(parts, sort_keys=True, default=str).encode())
        for blob in blobs:
            digest.update(hashlib.sha256(blob).digest())
        return digest.hexdigest()

    # ##################################################################
    # path
    def path(self, namespace: str, key: str, suffix: str) -> Path:
        return self.root / namespace / f"{key}.{suffix}"

    # ##################################################################
    # image
    # replay a PNG if recorded, else produce it and record it. An image with
    # an alpha channel (a cutout) keeps it; everything else is RGB.
    def image(self, namespace: str, key: str, produce: Callable[[], Image.Image]) -> Image.Image:
        location = self.path(namespace, key, "png")
        if location.is_file():
            with Image.open(location) as stored:
                return stored.convert("RGBA" if stored.mode == "RGBA" else "RGB")
        produced = produce()
        result = produced.convert("RGBA" if produced.mode == "RGBA" else "RGB")
        location.parent.mkdir(parents=True, exist_ok=True)
        result.save(location, format="PNG")
        return result

    # ##################################################################
    # text
    # replay a text reply if recorded, else produce and record it.
    def text(self, namespace: str, key: str, produce: Callable[[], str]) -> str:
        location = self.path(namespace, key, "txt")
        if location.is_file():
            return location.read_text()
        result = produce()
        location.parent.mkdir(parents=True, exist_ok=True)
        location.write_text(result)
        return result


# ##################################################################
# image bytes
# canonical PNG bytes of an image for hashing.
def image_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()
