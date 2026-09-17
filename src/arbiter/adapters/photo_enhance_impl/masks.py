"""Person/background masks from a BiRefNet cutout.

Identical to photo-enhance's person_masks, including the no-person
fallback (an empty person mask would crash the climb's segment crops;
with no person in frame both segments become the full frame so edits
apply globally).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

MASK_FEATHER = 1.0


def person_masks(cutout: Image.Image) -> dict[str, np.ndarray]:
    alpha = cutout.split()[-1].filter(ImageFilter.GaussianBlur(MASK_FEATHER))
    person = np.asarray(alpha).astype(np.float32) / 255.0
    if person.max() < 0.5:
        return {"person": np.ones_like(person), "background": np.ones_like(person)}
    return {"person": person, "background": 1.0 - person}
