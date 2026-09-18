"""Fail-closed still-image owner policy tests."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from textwrap import dedent

import pytest

from arbiter.image_policy import (
    REFERENCE_IMAGE_EDIT_MODEL,
    STILL_IMAGE_DISABLED_MESSAGE,
    StillImageGenerationDisabled,
    is_disabled_still_image_model,
)


def test_reference_image_edit_is_the_only_still_image_exception():
    """The reference editor is allowed; the checkpoint it uses is still denied under any other id."""
    assert not is_disabled_still_image_model(REFERENCE_IMAGE_EDIT_MODEL)
    assert not is_disabled_still_image_model("Reference_Image_Edit")
    assert is_disabled_still_image_model("black-forest-labs/FLUX.2-klein-9B")
    assert is_disabled_still_image_model("flux2")
    assert is_disabled_still_image_model("reference-image-edit-lora")


def test_reference_image_edit_requires_an_input_image(tmp_path):
    """No text-to-image path exists: a prompt without an image fails before any model call."""
    from arbiter.adapters.base import InferenceError
    from arbiter.adapters.reference_image_edit import ReferenceImageEditAdapter, fit_within

    adapter = ReferenceImageEditAdapter()
    adapter._pipe = object()  # type: ignore[assignment]  # never invoked on this path
    with pytest.raises(InferenceError, match="requires an input image"):
        adapter.infer({"prompt": "pro dslr"}, tmp_path, threading.Event())
    with pytest.raises(InferenceError, match="prompt is required"):
        adapter.infer({"image": "eA=="}, tmp_path, threading.Event())
    assert fit_within(1592, 1197) == (1536, 1152)
    assert fit_within(4000, 3000) == (1536, 1152)
    assert fit_within(1024, 768) == (1024, 768)


@pytest.mark.parametrize(
    "model_id",
    [
        "flux",
        "FLUX_2",
        "black-forest-labs/FLUX.1-schnell",
        "kontext-pro",
        "Tongyi-MAI/Z_Image_Turbo",
        "sdxl-lightning",
        "stable_diffusion-3",
        "pixart-sigma",
        "portrait-lora",
        "Qwen/Image-Generator",
    ],
)
def test_disabled_aliases(model_id):
    assert is_disabled_still_image_model(model_id)


@pytest.mark.parametrize(
    "model_id",
    ["birefnet", "ltx2", "ltx2-dev-denoise2-lora", "lora-train", "moondream"],
)
def test_non_still_models_remain_allowed(model_id):
    assert not is_disabled_still_image_model(model_id)


@pytest.mark.parametrize(
    "model_id",
    [
        "flora",
        "floral",
        "florence-2",
        "llm:flora",
        "voice/flora-v2",
        "chloral-voice",
        "color-adjuster",
    ],
)
def test_lora_boundary_substrings_remain_allowed(model_id):
    assert not is_disabled_still_image_model(model_id)


@pytest.mark.parametrize(
    "model_id",
    [
        "portrait-lora",
        "portrait_lora",
        "LORA/portrait",
        "artist.LoRA.v2",
        "flux-lora",
    ],
)
def test_lora_tokens_remain_disabled(model_id):
    assert is_disabled_still_image_model(model_id)


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("arbiter.adapters.flux", "FluxSchnellAdapter"),
        ("arbiter.adapters.flux2", "Flux2KleinAdapter"),
        ("arbiter.adapters.z_image", "ZImageTurboAdapter"),
    ],
)
def test_retained_adapters_refuse_load_and_infer_before_framework_import(
    module_name, class_name, tmp_path
):
    module = __import__(module_name, fromlist=[class_name])
    adapter_class = getattr(module, class_name)

    class RenamedAdapter(adapter_class):
        model_id = "birefnet"

    adapter = RenamedAdapter()
    adapter.model_id = "birefnet"
    with pytest.raises(StillImageGenerationDisabled, match="actively disabled"):
        adapter.load()
    with pytest.raises(StillImageGenerationDisabled, match="actively disabled"):
        adapter.infer({}, tmp_path, threading.Event())


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("arbiter.adapters.flux", "FluxSchnellAdapter"),
        ("arbiter.adapters.z_image", "ZImageTurboAdapter"),
    ],
)
def test_retained_image_edit_helpers_refuse_direct_invocation(module_name, class_name):
    module = __import__(module_name, fromlist=[class_name])
    adapter = getattr(module, class_name)()
    adapter.model_id = "birefnet"
    with pytest.raises(StillImageGenerationDisabled, match="actively disabled"):
        adapter._get_img2img_pipe()


def test_adapter_package_import_is_clean_strict_and_complete():
    expected = [
        "aesthetic-scorer",
        "birefnet",
        "composite",
        "demucs",
        "echomimic",
        "embed-text",
        "face-restore",
        "face-restore-codeformer",
        "insightface",
        "latentsync",
        "lora-train",
        "ltx2",
        "ltx2-denoise1",
        "ltx2-denoise2",
        "ltx2-dev-denoise1",
        "ltx2-dev-denoise2",
        "ltx2-encode",
        "ltx25-denoise1",
        "ltx25-encode",
        "minimax-fast-h3",
        "minimax-h3",
        "minimax-h3-local",
        "moondream",
        "music-generate",
        "photo-enhance",
        "reference-image-edit",
        "rvc-convert",
        "rvc-train",
        "sadtalker",
        "sonic",
        "tts-clone",
        "tts-custom",
        "tts-design",
        "tts-kokoro",
        "vocal-stem",
        "voice-fit",
        "wan-s2v",
        "whisper-large",
        "yue2",
    ]
    script = dedent(
        """
        import json
        import arbiter.adapters
        from arbiter.adapters.registry import list_registered
        print(json.dumps(list_registered()))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-W", "error", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    assert json.loads(proc.stdout) == expected


def test_direct_worker_invocation_refuses_before_adapter_startup():
    proc = subprocess.run(
        [sys.executable, "-m", "arbiter.worker_main", "flux2"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert STILL_IMAGE_DISABLED_MESSAGE in proc.stderr
