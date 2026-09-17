from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter

from .color import lab_to_rgb, rgb_to_lab, to_array, to_image

# amount range per tool; every tool takes exactly one strength called amount
TOOL_RANGES: dict[str, tuple[float, float]] = {
    "fill_light": (0.0, 1.0),
    "lift_shadows": (0.0, 1.0),
    "recover_highlights": (0.0, 1.0),
    "exposure": (-1.0, 1.0),
    "contrast": (-1.0, 1.0),
    "white_balance": (-1.0, 1.0),
    "skin_tone_refine": (0.0, 1.0),
    "vibrance": (0.0, 1.0),
    "saturation": (-1.0, 1.0),
    "selective_color": (0.0, 1.0),
    "clarity": (0.0, 1.0),
    "micro_sharpen": (0.0, 1.0),
    "realistic_bokeh": (0.0, 1.0),
    "detail_reconstruct": (0.0, 1.0),
}
PERSON_ONLY = {"fill_light", "skin_tone_refine"}
BACKGROUND_ONLY = {"selective_color", "realistic_bokeh"}
TOOL_HELP = {
    "fill_light": "soft bounce fill on shadowed face/body, suppresses shadow colour noise",
    "lift_shadows": "lift deep shadows (L<65) without touching highlights",
    "recover_highlights": "compress blown highlights (L>70) to restore texture",
    "exposure": "amount = EV: negative darkens, positive brightens",
    "contrast": "S-curve around the segment mean; negative flattens",
    "white_balance": "amount = temperature: negative cooler, positive warmer; optional tint -1..1, negative = less red/magenta",
    "skin_tone_refine": "pull blotchy red skin toward healthy golden-peach",
    "vibrance": "boost muted colours, leaves saturated ones alone",
    "saturation": "scale all chroma; negative desaturates",
    "selective_color": "enrich one scenery colour; target = ocean | sky | sand",
    "clarity": "local midtone contrast for pop",
    "micro_sharpen": "fine edge sharpening on hair, eyes, food",
    "realistic_bokeh": "progressive depth blur, stronger toward the horizon",
    "detail_reconstruct": "4x resample with multi-scale sharpening, then back down",
}


# ##################################################################
# tools for segment
# the tool names a given segment may use.
def tools_for_segment(segment: str) -> list[str]:
    excluded = BACKGROUND_ONLY if segment == "person" else PERSON_ONLY
    return [name for name in TOOL_RANGES if name not in excluded]


# ##################################################################
# blend
# mix a processed frame back over the original through a weight map.
def blend(rgb: np.ndarray, processed: np.ndarray, weight: np.ndarray) -> np.ndarray:
    w = weight[..., None]
    return np.clip(rgb * (1 - w) + processed * w, 0, 1)


# ##################################################################
# lift shadows
def lift_shadows(rgb: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    lightness, a, b = rgb_to_lab(rgb)
    shadow = np.clip((65.0 - lightness) / 55.0, 0, 1) ** 1.4
    return lab_to_rgb(np.clip(lightness + shadow * mask * amount * 22.0, 0, 100), a, b)


# ##################################################################
# recover highlights
def recover_highlights(rgb: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    lightness, a, b = rgb_to_lab(rgb)
    highlight = np.clip((lightness - 70.0) / 30.0, 0, 1) ** 1.3
    return lab_to_rgb(np.clip(lightness - highlight * mask * amount * 18.0, 0, 100), a, b)


# ##################################################################
# exposure
def exposure(rgb: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    lightness, a, b = rgb_to_lab(rgb)
    return lab_to_rgb(np.clip(lightness + mask * amount * 16.0, 0, 100), a, b)


# ##################################################################
# contrast
def contrast(rgb: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    lightness, a, b = rgb_to_lab(rgb)
    selected = mask > 0.05
    mean = float(lightness[selected].mean()) if selected.any() else 50.0
    return lab_to_rgb(np.clip(lightness + mask * (lightness - mean) * amount * 0.7, 0, 100), a, b)


# ##################################################################
# fill light
def fill_light(rgb: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    lightness, a, b = rgb_to_lab(rgb)
    fill = np.clip((60.0 - lightness) / 50.0, 0, 1) * amount * 22.0 * mask
    lifted = np.clip(lightness + fill, 0, 100)
    suppress = np.clip((40.0 - lifted) / 40.0, 0, 1) * amount * 0.25 * mask
    return lab_to_rgb(lifted, a * (1 - suppress), b * (1 - suppress))


# ##################################################################
# white balance
def white_balance(rgb: np.ndarray, mask: np.ndarray, amount: float, tint: float = 0.0) -> np.ndarray:
    lightness, a, b = rgb_to_lab(rgb)
    mid = np.clip((85.0 - lightness) / 25.0, 0, 1) * mask
    return lab_to_rgb(lightness, a + mid * (amount * 2.5 + tint * 8.0), b + mid * amount * 10.0)


# ##################################################################
# skin tone refine
def skin_tone_refine(rgb: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    lightness, a, b = rgb_to_lab(rgb)
    skin = (
        np.clip((lightness - 20) / 15, 0, 1)
        * np.clip((85 - lightness) / 15, 0, 1)
        * np.clip((a - 8) / 8, 0, 1)
        * np.clip((b - 5) / 8, 0, 1)
        * np.clip((45 - a) / 10, 0, 1)
        * mask
    )
    return lab_to_rgb(lightness, a + skin * amount * (16.0 - a) * 0.55, b + skin * amount * (18.0 - b) * 0.45)


# ##################################################################
# vibrance
def vibrance(rgb: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    lightness, a, b = rgb_to_lab(rgb)
    muted = np.clip(1.0 - np.hypot(a, b) / 45.0, 0, 1)
    gain = 1.0 + mask * muted * amount * 1.6
    return lab_to_rgb(lightness, a * gain, b * gain)


# ##################################################################
# saturation
def saturation(rgb: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    lightness, a, b = rgb_to_lab(rgb)
    gain = 1.0 + mask * amount * 0.8
    return lab_to_rgb(lightness, a * gain, b * gain)


# ##################################################################
# selective color
def selective_color(rgb: np.ndarray, mask: np.ndarray, amount: float, target: str = "ocean") -> np.ndarray:
    lightness, a, b = rgb_to_lab(rgb)
    if target == "sky":
        region = np.clip((lightness - 70) / 20, 0, 1) * np.clip(-b / 8, 0, 1) * mask
        return lab_to_rgb(lightness, a - region * amount * 2.0, b - region * amount * 8.0)
    band = np.clip((lightness - (45 if target == "sand" else 35)) / 20, 0, 1) * np.clip((85 - lightness) / 15, 0, 1)
    if target == "sand":
        region = band * np.clip(b / 10, 0, 1) * mask
        return lab_to_rgb(lightness, a + region * amount * 2.0, b + region * amount * 6.0)
    region = band * np.clip(-a / 8, 0, 1) * mask
    return lab_to_rgb(lightness, a - region * amount * 4.0, b - region * amount * 6.0)


# ##################################################################
# unsharp
# shared PIL unsharp-mask helper for the texture tools.
def unsharp(rgb: np.ndarray, radius: float, percent: int, threshold: int) -> np.ndarray:
    return to_array(to_image(rgb).filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)))


# ##################################################################
# clarity
def clarity(rgb: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    return blend(rgb, unsharp(rgb, 28, int(50 * amount), 3), mask * amount)


# ##################################################################
# micro sharpen
def micro_sharpen(rgb: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    return blend(rgb, unsharp(rgb, 1.1, int(60 + 100 * amount), 2), mask * amount)


# ##################################################################
# realistic bokeh
def realistic_bokeh(rgb: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    if amount < 0.05:
        return rgb
    height = rgb.shape[0]
    depth = np.broadcast_to(np.linspace(1.0, 0.25, height).astype(np.float32)[:, None], mask.shape)
    weight = gaussian_filter(depth, sigma=3.0) * mask * amount
    image = to_image(rgb)
    near = to_array(image.filter(ImageFilter.GaussianBlur(radius=3.0 * amount)))
    far = to_array(image.filter(ImageFilter.GaussianBlur(radius=7.0 * amount)))
    return blend(rgb, near * 0.45 + far * 0.55, weight)


# ##################################################################
# detail reconstruct
# 4x Lanczos resample with two-scale sharpening then back down; a cheap
# non-generative detail pass the model may layer on a segment.
def detail_reconstruct(rgb: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    image = to_image(rgb)
    width, height = image.size
    up = image.resize((width * 4, height * 4), Image.Resampling.LANCZOS)
    up = up.filter(ImageFilter.UnsharpMask(radius=3.6, percent=80, threshold=2))
    up = up.filter(ImageFilter.UnsharpMask(radius=0.9, percent=50, threshold=1))
    down = up.resize((width, height), Image.Resampling.LANCZOS)
    return blend(
        rgb, to_array(down.filter(ImageFilter.UnsharpMask(radius=0.9, percent=65, threshold=2))), mask * amount
    )


OPERATORS = {
    "fill_light": fill_light,
    "lift_shadows": lift_shadows,
    "recover_highlights": recover_highlights,
    "exposure": exposure,
    "contrast": contrast,
    "white_balance": white_balance,
    "skin_tone_refine": skin_tone_refine,
    "vibrance": vibrance,
    "saturation": saturation,
    "selective_color": selective_color,
    "clarity": clarity,
    "micro_sharpen": micro_sharpen,
    "realistic_bokeh": realistic_bokeh,
    "detail_reconstruct": detail_reconstruct,
}


# ##################################################################
# normalise call
# coerce a model-proposed call into a canonical one: known tool, allowed
# segment, amount clipped to range, extras validated. Raises ValueError
# with a message the model can act on.
def normalise_call(call: dict, segments: tuple[str, ...] = ("person", "background")) -> dict:
    segment = call.get("segment")
    tool = call.get("tool")
    if segment not in segments:
        raise ValueError(f"segment must be one of {segments}, got {segment!r}")
    if tool not in tools_for_segment(segment):
        raise ValueError(f"tool {tool!r} is not available on segment {segment!r}")
    low, high = TOOL_RANGES[tool]
    raw = call.get("amount", call.get("ev", call.get("temp")))
    if raw is None:
        raise ValueError("every tool takes exactly one strength key, amount")
    clean = {"segment": segment, "tool": tool, "amount": float(np.clip(float(raw), low, high))}
    if tool == "white_balance":
        clean["tint"] = float(np.clip(float(call.get("tint", 0.0)), -1, 1))
    if tool == "selective_color":
        target = call.get("target", "ocean")
        clean["target"] = target if target in ("ocean", "sky", "sand") else "ocean"
    return clean


# ##################################################################
# apply call
# run one normalised call on a frame through its segment mask.
def apply_call(rgb: np.ndarray, masks: dict[str, np.ndarray], call: dict) -> np.ndarray:
    extras = {key: value for key, value in call.items() if key in ("tint", "target")}
    return OPERATORS[call["tool"]](rgb, masks[call["segment"]], call["amount"], **extras)
