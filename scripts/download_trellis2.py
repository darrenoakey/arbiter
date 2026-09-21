# Official checkpoint installer; the authenticated download token arrives only on stdin.
import json
import sys
from pathlib import Path


# Pin each download to its resolved repository revision and retain provenance.
def download_models(root: Path, token: str) -> None:
    from huggingface_hub import HfApi, snapshot_download

    repositories = [
        ("microsoft/TRELLIS.2-4B", "TRELLIS.2-4B", None),
        ("microsoft/TRELLIS-image-large", "TRELLIS-image-large", ["ckpts/ss_dec_conv3d_16l8_fp16.*"]),
        ("facebook/dinov3-vitl16-pretrain-lvd1689m", "dinov3-vitl16-pretrain-lvd1689m", None),
        ("briaai/RMBG-2.0", "RMBG-2.0", None),
    ]
    api = HfApi(token=token)
    provenance = {}
    for repository, directory, patterns in repositories:
        revision = api.model_info(repository).sha
        print(f"Downloading official {repository} revision {revision}", flush=True)
        snapshot_download(
            repo_id=repository, revision=revision, token=token,
            local_dir=root / directory, allow_patterns=patterns,
            ignore_patterns=["*.onnx", "*.pth", "*.bin"], max_workers=4,
        )
        provenance[repository] = revision
    (root / "revisions.json").write_text(json.dumps(provenance, indent=2) + "\n")
    configure_pipeline(root)


# Keep all inference dependencies local; rewrite only machine-specific paths.
def configure_pipeline(root: Path) -> None:
    pipeline = root / "TRELLIS.2-4B" / "pipeline.json"
    original = pipeline.with_name("pipeline.json.upstream")
    if not original.exists():
        original.write_bytes(pipeline.read_bytes())
    config = json.loads(original.read_text())
    args = config["args"]
    args["models"]["sparse_structure_decoder"] = str(
        root / "TRELLIS-image-large" / "ckpts" / "ss_dec_conv3d_16l8_fp16"
    )
    args["image_cond_model"]["args"]["model_name"] = str(root / "dinov3-vitl16-pretrain-lvd1689m")
    args["rembg_model"]["args"]["model_name"] = str(root / "RMBG-2.0")
    pipeline.write_text(json.dumps(config, indent=2) + "\n")
    print("Official checkpoints ready with local pipeline paths", flush=True)


if __name__ == "__main__":
    credential = sys.stdin.read().strip()
    if not credential:
        raise SystemExit("Missing HF token on stdin; retrieve it from configured daz-secrets provider")
    try:
        download_models(Path("/home/darren/local/models/trellis2"), credential)
    except Exception as error:  # noqa: BLE001 - redact authenticated client errors at the CLI boundary
        print(f"Checkpoint installation failed: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1) from None
