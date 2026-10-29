import io
import threading
import zipfile

import pytest
from PIL import Image
from pydantic import ValidationError

from arbiter.adapters.aesthetic_scorer import AestheticScorerAdapter
from arbiter.adapters.base import InferenceError
from arbiter.schemas import AestheticScoreParams


def test_legacy_schema_and_batch_contract():
    assert AestheticScoreParams(image="base64").image == "base64"
    assert AestheticScoreParams(image_file="/inbox/photo.jpg").images_file is None
    assert AestheticScoreParams(images_file="/inbox/originals.zip").images_file
    for params in (
        {"images_file": ""},
        {"images_file": "batch.zip", "image": "base64"},
        {"images_file": "batch.zip", "image_file": "photo.jpg"},
    ):
        with pytest.raises(ValidationError):
            AestheticScoreParams(**params)


def test_zip_members_are_decoded_without_extraction(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (17, 23), (20, 40, 60)).save(buffer, format="PNG")
    path = tmp_path / "originals.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../outside.png", buffer.getvalue())
        archive.writestr("invalid.heic", b"not an image")
    with zipfile.ZipFile(path) as archive:
        image = AestheticScorerAdapter._archive_image(archive, archive.infolist()[0])
        try:
            assert image.size == (17, 23)
            assert image.getpixel((0, 0)) == (20, 40, 60)
        finally:
            image.close()
        with pytest.raises(OSError):
            AestheticScorerAdapter._archive_image(archive, archive.infolist()[1])
    assert not (tmp_path.parent / "outside.png").exists()


def test_zip_bad_members_are_ordered_item_errors(tmp_path):
    path = tmp_path / "originals.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("folder/", b"")
        archive.writestr("z.jpg", b"bad jpeg")
        archive.writestr("a.png", b"bad png")
    result = AestheticScorerAdapter().infer(
        {"images_file": str(path)}, tmp_path / "result", threading.Event()
    )
    assert [item["name"] for item in result["items"]] == ["z.jpg", "a.png"]
    assert all("error" in item and "scores" not in item for item in result["items"])
    assert "scores" not in result
    assert (tmp_path / "result/result.json").is_file()


@pytest.mark.parametrize("content", [b"not a zip", b"PK\x03\x04truncated"])
def test_bad_archive_is_client_error(tmp_path, content):
    path = tmp_path / "bad.zip"
    path.write_bytes(content)
    with pytest.raises(InferenceError, match="bad input images ZIP"):
        AestheticScorerAdapter().infer(
            {"images_file": str(path)}, tmp_path / "result", threading.Event()
        )


def test_empty_zip_is_client_error(tmp_path):
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("directory/", b"")
    with pytest.raises(InferenceError, match="no file members"):
        AestheticScorerAdapter().infer(
            {"images_file": str(path)}, tmp_path / "result", threading.Event()
        )


def test_cancelled_batch_does_not_return_item_errors(tmp_path):
    from arbiter.adapters.base import CancelledException

    flag = threading.Event()
    flag.set()
    with pytest.raises(CancelledException):
        AestheticScorerAdapter().infer(
            {"images_file": str(tmp_path / "never-read.zip")}, tmp_path, flag
        )
