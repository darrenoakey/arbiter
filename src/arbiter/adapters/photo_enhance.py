"""Photo-enhance adapter — the whole pipeline on spark.

One job does everything photo-enhance did on the Mac, with all heavy
compute on spark:

1. BiRefNet person segmentation      (arbiter background-remove, spark)
2. SeedVR2 2x detail recovery        (in-process CUDA, vendored SeedVR2)
3. FLUX.2-klein pro-DSLR reference   (arbiter reference-image-edit, spark)
4. Guided per-segment photometric climb:
   - aesthetic scoring per attempt    (arbiter aesthetic-scorer, spark)
   - climb conversation               (arbiter chat, qwen3-vl on spark)
   - edits themselves                 (local numpy/PIL, CPU)

The Mac-side consumer submits ONE job and waits; the result PNG is the
enhanced print file.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from arbiter.adapters.base import InferenceError, ModelAdapter
from arbiter.adapters.photo_enhance_impl.client import ArbiterClient
from arbiter.adapters.photo_enhance_impl.climb import GuidedClimb
from arbiter.adapters.photo_enhance_impl.detail import recover_detail
from arbiter.adapters.photo_enhance_impl.masks import person_masks
from arbiter.adapters.photo_enhance_impl.reference import render_reference
from arbiter.adapters.photo_enhance_impl.settings import Settings
from arbiter.adapters.photo_enhance_impl.vision_chat import VisionChat
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

SUB_WHY = "photo-enhance pipeline job"


@register
class PhotoEnhanceAdapter(ModelAdapter):
    model_id = "photo-enhance"

    def __init__(self) -> None:
        self._settings = Settings()
        self._upscaler = None

    # ##################################################################
    # load
    def load(self, device: str = "cuda") -> None:
        from arbiter.seedvr.stills import SeedVR2Upscaler

        upscaler = SeedVR2Upscaler(self._settings.seedvr2_home, self._settings.seedvr2_ckpt)
        upscaler.load()
        self._upscaler = upscaler
        log.info("photo-enhance worker ready (seedvr2 3B + arbiter services)")

    # ##################################################################
    # unload
    def unload(self) -> None:
        if self._upscaler is not None:
            self._upscaler.close()
            self._upscaler = None
        self._cleanup_gpu()

    # ##################################################################
    # infer
    def infer(self, params: dict, output_dir: Path, cancel_flag) -> dict:
        if self._upscaler is None:
            raise InferenceError("photo-enhance not loaded")
        self._check_cancel(cancel_flag)

        image = self._resolve_image(params)

        # bring-up probe: SeedVR2 detail pass only, no arbiter services
        if params.get("stage") == "probe":
            if params.get("detail", True):
                image = recover_detail(image, self._upscaler.upscale)
            self._check_cancel(cancel_flag)
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            image.save(out / "result.png")
            return {"format": "png", "file": "result.png", "width": image.width, "height": image.height}

        client = ArbiterClient(self._settings, params.get("why") or SUB_WHY)

        # 1. person masks via BiRefNet
        log.info("segmenting %dx%d with BiRefNet ...", *image.size)
        cutout = client.remove_background(image)
        masks = person_masks(cutout)
        self._check_cancel(cancel_flag)

        # 2. SeedVR2 detail pre-pass
        if params.get("detail", True):
            image = recover_detail(image, self._upscaler.upscale)
            self._check_cancel(cancel_flag)

        # 3. pro-DSLR reference renders via FLUX.2-klein
        reference = render_reference(image, client, int(params.get("candidates", 6)))
        self._check_cancel(cancel_flag)

        # 4. guided climb: photometric edits toward the reference,
        #    graded by the aesthetic scorer, proposed by qwen3-vl
        chat = VisionChat(self._settings)
        climb = GuidedClimb(client, chat, masks, reference.image)
        result = climb.run(image, int(params.get("turns", 45)))
        self._check_cancel(cancel_flag)

        # 5. install outputs
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.image.save(out / "result.png")
        (out / "climb.json").write_text(
            json.dumps(
                {
                    "scores": result.scores,
                    "distance": result.distance,
                    "accepted": result.accepted,
                    "attempts": result.attempts,
                    "transcript": result.transcript,
                },
                indent=1,
            )
        )
        log.info(
            "photo-enhance done: overall %.3f, %d accepted edits, distance %.1f",
            result.scores["overall_aesthetic"],
            len(result.accepted),
            result.distance,
        )
        return {
            "format": "png",
            "file": "result.png",
            "width": result.image.width,
            "height": result.image.height,
            "scores": result.scores,
            "distance": result.distance,
            "accepted_edits": len(result.accepted),
        }

    # ##################################################################
    # estimate time
    def estimate_time(self, params: dict) -> float:
        # seedvr2 ~1 min per 1536px tile + renders ~10 s each + ~35 s per
        # climb turn; a full-size photo lands around 45-60 minutes.
        tiles = 4
        ms = tiles * 60_000 + int(params.get("candidates", 6)) * 10_000 + int(params.get("turns", 45)) * 35_000
        return float(ms + 5 * 60_000)

    # ##################################################################
    # loaded check used by tests
    @property
    def loaded(self) -> bool:
        return self._upscaler is not None
