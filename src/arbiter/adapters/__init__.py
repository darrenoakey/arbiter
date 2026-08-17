"""Model adapters — import every supported adapter or fail initialization."""

import importlib

_ADAPTER_MODULES = (
    "birefnet",
    "moondream",
    "whisper_large",
    "tts_custom",
    "tts_clone",
    "tts_design",
    "kokoro_tts",
    "latentsync",
    "sadtalker",
    "echomimic",
    "wan_s2v",
    "sonic",
    "ltx2",
    "minimax_h3",
    "minimax_h3_local",
    "ltx2_encode",
    "ltx2_denoise1",
    "ltx2_denoise2",
    "ltx2_dev_denoise1",
    "ltx2_dev_denoise2",
    "face_restore",
    "face_restore_codeformer",
    "aesthetic_scorer",
    "lora_train",
    "composite",
    "insightface",
    "embed_text",
    "demucs_sep",
    "rvc",
    "voice_fit",
)

for _module_name in _ADAPTER_MODULES:
    importlib.import_module(f".{_module_name}", __name__)
