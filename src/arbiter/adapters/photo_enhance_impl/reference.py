from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .client import ArbiterClient as Arbiter
from .cache_shim import ContentCache, image_bytes

log = logging.getLogger(__name__)

# the brief that started all of this: same photo, best kit, best photographer
REFERENCE_PROMPT = (
    "The exact same photograph, but taken with a professional full-frame DSLR and the best quality "
    "85mm f/1.4 lens by an amazing portrait photographer, with a full lighting setup: soft key light "
    "and fill light on the subject's face so it is evenly and beautifully lit with natural healthy skin "
    "tones, no harsh backlight halo, gentle rim light on the hair, creamy shallow depth of field on the "
    "background, the background properly exposed with rich natural colour, crisp detail on eyes, hair "
    "and foreground objects, professional colour grading, magazine editorial quality. Keep the same "
    "people, same pose, same expression, same clothes, same objects, same framing and composition."
)
# seed / steps spread that covered the useful range on FLUX.2-klein; guidance
# is fixed at 1.0 because the step-distilled model ignores it
CANDIDATE_VARIANTS = (
    (1, 4, 1.0),
    (2, 4, 1.0),
    (3, 8, 1.0),
    (4, 8, 1.0),
    (5, 12, 1.0),
    (6, 12, 1.0),
)


# ##################################################################
# candidate
# one rendered reference plus its aesthetic scores.
@dataclass(frozen=True)
class Candidate:
    seed: int
    steps: int
    guidance: float
    image: Image.Image
    scores: dict[str, float]


# ##################################################################
# render reference
# ask the scoped reference-image-edit adapter for several "pro DSLR"
# versions of the photo, score each, and return the best. The winner is a
# TARGET for the grade only: it redraws faces, so it never ships.
def render_reference(image: Image.Image, arbiter: Arbiter, count: int) -> Candidate:
    cache = ContentCache(Path(arbiter.settings.cache_dir))
    source = image_bytes(image)
    variants = CANDIDATE_VARIANTS[:count]
    keys = [ContentCache.key("reference", {"prompt": REFERENCE_PROMPT, "variant": v}, [source]) for v in variants]
    # submit every uncached render up front so the model loads once per run
    jobs = {
        key: arbiter.submit_reference_edit(image, REFERENCE_PROMPT, *variant)
        for key, variant in zip(keys, variants, strict=True)
        if not cache.path("reference", key, "png").is_file()
    }
    candidates = []
    for key, variant in zip(keys, variants, strict=True):
        try:
            rendered = cache.image("reference", key, functools.partial(collect_render, arbiter, jobs.get(key)))
            candidates.append(score_candidate(image, arbiter, rendered, *variant))
        except (RuntimeError, OSError) as err:
            # one killed or evicted render must not waste the others; it is left out
            log.error("reference candidate seed=%d steps=%d failed: %s", variant[0], variant[1], err)
    if not candidates:
        raise RuntimeError(f"all {len(variants)} reference renders failed; see log")
    best = max(candidates, key=lambda c: c["scores"]["overall_aesthetic"])
    log.info(
        "reference: best seed=%d steps=%d overall=%.3f",
        best["seed"],
        best["steps"],
        best["scores"]["overall_aesthetic"],
    )
    return Candidate(**best)


# ##################################################################
# collect render
# wait for a queued render; only ever called on a cache miss, so a job id
# is always present.
def collect_render(arbiter: Arbiter, job_id: str | None) -> Image.Image:
    if job_id is None:
        raise RuntimeError("render was not submitted but is missing from the cache")
    return arbiter.reference_edit_result(job_id)


# ##################################################################
# score candidate
# score one render; resized to the source size so
# per-segment statistics line up pixel for pixel with the frame being graded.
def score_candidate(
    image: Image.Image, arbiter: Arbiter, rendered: Image.Image, seed: int, steps: int, guidance: float
) -> dict:
    rendered = rendered.resize(image.size, Image.Resampling.LANCZOS)
    scores = arbiter.aesthetic_score(rendered)
    log.info(
        "reference candidate seed=%d steps=%d g=%.1f overall=%.3f", seed, steps, guidance, scores["overall_aesthetic"]
    )
    return {"seed": seed, "steps": steps, "guidance": guidance, "image": rendered, "scores": scores}
