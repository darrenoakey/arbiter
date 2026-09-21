# Local pipeline rewrite uses real files and preserves upstream bytes across reruns.
import importlib.util
import json
from pathlib import Path


def test_local_pipeline_rewrite_is_idempotent(tmp_path):
    source = Path(__file__).parents[2] / "scripts/download_trellis2.py"
    spec = importlib.util.spec_from_file_location("trellis_download", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "weights"
    model = root / "TRELLIS.2-4B"
    model.mkdir(parents=True)
    pipeline = model / "pipeline.json"
    original = json.dumps({"args": {
        "models": {"sparse_structure_decoder": "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"},
        "image_cond_model": {"args": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"}},
        "rembg_model": {"args": {"model_name": "briaai/RMBG-2.0"}},
    }})
    pipeline.write_text(original)
    module.configure_pipeline(root)
    first = pipeline.read_bytes()
    module.configure_pipeline(root)
    assert pipeline.read_bytes() == first
    assert (model / "pipeline.json.upstream").read_text() == original
    args = json.loads(first)["args"]
    assert args["image_cond_model"]["args"]["model_name"] == str(root / "dinov3-vitl16-pretrain-lvd1689m")
    assert args["rembg_model"]["args"]["model_name"] == str(root / "RMBG-2.0")
    assert args["models"]["sparse_structure_decoder"] == str(
        root / "TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
    )
