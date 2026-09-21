# TRELLIS.2 Deployment & Dependency Reference on Spark (ARM64 GB10)

## Architecture & Hardware Overview
- **Host**: Spark (NVIDIA DGX Spark GB10, Grace Blackwell ARM64, CUDA 13.0, 128GB unified memory).
- **Driver**: 580.178.04, Toolkit 13.0.88.
- **Python Runtime**: Python 3.12.3 in dedicated real-binary-copy venv:
  `/home/darren/src/arbiter/venvs/trellis2/bin/python`
- **Upstream Source**: Pinned at `/home/darren/src/arbiter/local/trellis2` (commit `75fbf0183001ed9876c8dbb35de6b68552ee08bd`).
- **Model Checkpoints**: Root NVMe storage `/home/darren/local/models/trellis2/` (root disk has ~450GB free vs ~160GB on /mnt/t9).

## Pinned Dependencies & Provenance

| Package | Version / Commit | Notes |
|---------|------------------|-------|
| `torch` | `2.12.0+cu130` | Pinned PyTorch build compatible with flash-attn and C++17 CUDA extensions. |
| `torchvision` | `0.27.0+cu130` | Matching vision build; torchaudio removed (unused for 3D). |
| `flash-attn` | `2.8.4` | Pre-built SBSA wheel from `https://pypi.jetson-ai-lab.io/sbsa/cu130`. Compatible with PyTorch 2.12. |
| `transformers` | `4.57.6` | Pinned for DINOv3ViTModel compatibility (`model.layer` attribute access expected by upstream DinoV3FeatureExtractor). |
| `o-voxel` | `0.0.1` (`75fbf01`) | Submodule `third_party/eigen` at `21e4582`. Built with `TORCH_CUDA_ARCH_LIST="12.0"`. |
| `FlexGEMM` | `6dd94a8` | GitHub `JeffreyXiang/FlexGEMM`. Sparse convolution / grid-sample Triton + CUDA kernels. |
| `CuMesh` | `12289e1` | GitHub `JeffreyXiang/CuMesh` (recursive with `cubvh` and `eigen`). CUDA mesh simplification/remeshing. |
| `nvdiffrast` | `253ac4f` (`v0.4.0`) | GitHub `NVlabs/nvdiffrast`. Rasterization CUDA extensions (`_nvdiffrast_c`). |
| `nvdiffrec` | `b296927` (`renderutils`) | GitHub `JeffreyXiang/nvdiffrec`. Package `nvdiffrec_render` with `EnvironmentLight`. |
| `utils3d` | `9a4eb15` (`0.0.2`) | GitHub `EasternJournalist/utils3d`. Requires system `libx11-dev` for `glcontext`. |

## Pipeline Configuration & Gated Dependencies

### Local `pipeline.json` Resolution
Upstream `pipeline.json` references external HuggingFace repository IDs:
1. `sparse_structure_decoder`: `"microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"` (Public, MIT).
2. `image_cond_model`: `"facebook/dinov3-vitl16-pretrain-lvd1689m"` (Meta Gated License).
3. `rembg_model`: `"briaai/RMBG-2.0"` (BRIA Gated Commercial/Non-commercial License).

To ensure deterministic local offline serving without HuggingFace network round-trips:
- The local model root stores the downloads under `/home/darren/local/models/trellis2/`.
- `/home/darren/local/models/trellis2/TRELLIS.2-4B/pipeline.json` points directly to absolute local paths:
  - `sparse_structure_decoder` -> `/home/darren/local/models/trellis2/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16`
  - `image_cond_model.args.model_name` -> `/home/darren/local/models/trellis2/dinov3-vitl16-pretrain-lvd1689m`
  - `rembg_model.args.model_name` -> `/home/darren/local/models/trellis2/RMBG-2.0`
- The unmodified original upstream configuration is preserved at `pipeline.json.upstream`.

### Licensing & Gated Model Access Blocker
TRELLIS.2 code and weights are under the **MIT License**. However, the full end-to-end pipeline relies on components that are **NOT MIT**:
- **DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`)**: Governed by the **Meta DINOv3 License**. Gated repo on Hugging Face.
- **RMBG-2.0 (`briaai/RMBG-2.0`)**: Governed by the **BRIA 2.0 License** (free for non-commercial research, commercial agreement required for commercial use). Gated repo on Hugging Face.

### Authorized downloads

Both official repositories accepted the owner's existing Hugging Face credential on 2026-09-21. The legacy `~/.huggingface/huggingface.token` was migrated once into the configured `daz-secrets` provider as `huggingface/token`, with exact readback verification. Runtime inference uses only the downloaded local weights; it does not read legacy token files or require a token.

Copy `scripts/setup-trellis2.sh` and `scripts/download_trellis2.py` together onto Spark for recreation. The installer builds the isolated dependencies and reads a Hugging Face token from stdin for checkpoint downloads. Retrieve the token through `daz-secrets` on the operator machine and transmit it through SSH stdin, never argv, environment variables, logs or a remote credential file. For an already installed venv, invoke only `download_trellis2.py` through that venv. Downloads use the official repositories, resolve and record their commit revisions in `revisions.json`, and write the local pipeline only after all downloads succeed. No third-party gated-weight mirrors are used.

*Background removal*: RGBA inputs with a non-opaque alpha mask skip background-removal inference during upstream preprocessing. The current upstream pipeline still constructs its configured remover at load time, so its local checkpoint remains required.

## Arbiter API

Job type: `image-to-3d`; model: `trellis2`. Submit `image` as base64 or `image_file` staged via `arbiter_client.stage_file()`. Native input is an image, not text. Optional parameters: `resolution` (512/1024/1536, default 1024), `steps` (1–50, default 12), `seed` (default 42), `texture_size` (1024/2048/4096, default 2048), `decimation` (100000–1000000 faces), `include_stl` (default false). Worker-side validation enforces bounds even when the Go server routes raw parameters.

The result is `result.glb` with embedded PNG PBR textures. Optional `geometry.stl` has no colors. Results also report geometry counts, load/inference time and CUDA peak allocation for calibration. The adapter uses dense SDPA, sparse flash attention and FlexGEMM; all model execution must go through an Arbiter job. Initial 48 GB/120-second declarations are conservative estimates, not measured Spark performance, until live calibration is recorded.

