import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from arbiter.adapters.id_lora import IDLoraAdapter
from arbiter.adapters.registry import get_adapter_class
from arbiter.schemas import IDLoraParams, JOB_TYPE_PARAMS, JOB_TYPE_TO_MODEL


def valid_params():
    return {
        "prompt": "[VISUAL]: A person speaking. [SPEECH]: Hello. [SOUNDS]: Speech.",
        "encoded_file": "/inbox/encoded.pt",
        "image_file": "/inbox/scene.png",
        "ref_audio_file": "/inbox/speaker.wav",
    }


# Valid params: exercise the actual public contract without GPU substitution.


def test_id_lora_contract():
    values = IDLoraParams.model_validate(valid_params())
    assert (values.width, values.height, values.num_frames, values.steps) == (
        512,
        512,
        121,
        30,
    )
    assert JOB_TYPE_PARAMS["video-id-lora"] is IDLoraParams
    assert JOB_TYPE_TO_MODEL["video-id-lora"] == "id-lora"
    assert get_adapter_class("id-lora") is IDLoraAdapter
    for frames in (161, 201):
        assert (
            IDLoraParams.model_validate(
                valid_params() | {"num_frames": frames}
            ).num_frames
            == frames
        )


# Contract: public routing and authors' frame clock agree.


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": "unstructured"},
        {"num_frames": 200},
        {"width": 513},
        {"steps": 0},
        {"reference_audio": "wrong key"},
        {"seed": -1},
    ],
)
def test_id_lora_rejects_invalid_params(change):
    with pytest.raises(ValidationError):
        IDLoraParams.model_validate(valid_params() | change)


# Invalid params: reject silently ignored controls and unsupported tensor shapes.


def test_id_lora_worker_budget():
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "config/spark/id-lora.model.json").read_text())
    assert config["max_concurrent"] == config["max_instances"] == 1
    assert config["memory_gb"] == 82
    assert config["worker_cmd"] == [
        "/home/darren/src/arbiter/venvs/id-lora/bin/python",
        "-m",
        "arbiter.worker_main",
        "id-lora",
    ]


# Budget: the guarded worker is mandatory and cannot fan out concurrent pipelines.


def test_id_lora_exact_policy():
    from arbiter.image_policy import is_disabled_still_image_model

    assert not is_disabled_still_image_model("id-lora")
    for name in ("id-lora-custom", "flux-lora", "id-lora-image"):
        assert is_disabled_still_image_model(name)


# Policy: allow the exact video adapter without opening arbitrary LoRA workers.
