from __future__ import annotations

import numpy as np
from PIL import Image

SRGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
    dtype=np.float32,
)
XYZ_TO_SRGB = np.array(
    [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]],
    dtype=np.float32,
)
D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
LAB_DELTA = 6 / 29
STAT_KEYS = ("L", "a", "b", "L_std", "chroma")
STAT_WEIGHTS = {"L": 1.0, "a": 1.5, "b": 1.5, "L_std": 0.7, "chroma": 1.0}


# ##################################################################
# rgb to lab
# sRGB floats in [0,1] to CIE L*a*b* channels, vectorised over an image.
def rgb_to_lab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ SRGB_TO_XYZ.T / D65
    f = np.where(xyz > LAB_DELTA**3, np.cbrt(xyz), xyz / (3 * LAB_DELTA * LAB_DELTA) + 4 / 29)
    return 116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])


# ##################################################################
# lab to rgb
# inverse of rgb_to_lab, clipped back into displayable sRGB.
def lab_to_rgb(lightness: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fy = (lightness + 16) / 116
    f = np.stack([fy + a / 500, fy, fy - b / 200], axis=-1)
    xyz = np.where(f > LAB_DELTA, f**3, (f - 4 / 29) * (3 * LAB_DELTA * LAB_DELTA)) * D65
    linear = np.clip(xyz @ XYZ_TO_SRGB.T, 0, None)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return np.clip(srgb, 0, 1)


# ##################################################################
# to array
# PIL RGB image to float32 array in [0,1].
def to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB")).astype(np.float32) / 255.0


# ##################################################################
# to image
# float array in [0,1] back to an 8-bit PIL image.
def to_image(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8))


# ##################################################################
# segment stats
# mean L/a/b, L spread and mean chroma over the pixels a mask selects;
# the compact description of "how a region is lit and coloured".
def segment_stats(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    lightness, a, b = rgb_to_lab(rgb)
    selected = mask > 0.5
    return {
        "L": float(lightness[selected].mean()),
        "a": float(a[selected].mean()),
        "b": float(b[selected].mean()),
        "L_std": float(lightness[selected].std()),
        "chroma": float(np.hypot(a[selected], b[selected]).mean()),
    }


# ##################################################################
# stats distance
# weighted euclidean distance between two segment_stats results; colour
# axes weigh more than lightness because a cast is the most visible miss.
def stats_distance(stats: dict[str, float], reference: dict[str, float]) -> float:
    total = sum(STAT_WEIGHTS[key] * (stats[key] - reference[key]) ** 2 for key in STAT_KEYS)
    return float(np.sqrt(total))
