"""Model adapters — auto-import to trigger registration.

Each adapter is imported in isolation. If one adapter has a broken import
(missing file, syntax error, bad dependency), it does NOT take down the
whole package — it's logged and skipped. Workers for other models will
still start fine.
"""
import importlib
import logging

_log = logging.getLogger(__name__)

_ADAPTERS = [
    "birefnet",
    "flux",
    "moondream",
    "whisper_large",
    "tts_custom",
    "tts_clone",
    "tts_design",
    "latentsync",
    "sadtalker",
    "sonic",
    "ltx2",
    "ltx2_encode",
    "ltx2_denoise1",
    "ltx2_denoise2",
    "aesthetic_scorer",
    "z_image",
    "lora_train",
    "composite",
]

for _name in _ADAPTERS:
    try:
        importlib.import_module(f".{_name}", __name__)
    except Exception as e:  # noqa: BLE001
        _log.warning("adapter %s failed to import: %s", _name, e)
