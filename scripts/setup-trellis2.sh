#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
sudo apt-get install -y libx11-dev libgl1-mesa-dev libegl1-mesa-dev

echo "=== TRELLIS.2 Spark GB10 Setup Started ($(date)) ==="

export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
export CC=gcc
export CXX=g++

VENV="/home/darren/src/arbiter/venvs/trellis2"
SRC="/home/darren/src/arbiter/local/trellis2"
EXT_DIR="/tmp/trellis_extensions"
MODELS_ROOT="/home/darren/local/models/trellis2"

mkdir -p "$EXT_DIR"
mkdir -p "$MODELS_ROOT"

echo "=== 1. Checking / Creating venv with real binary copies ==="
if [ ! -d "$VENV" ]; then
    python3 -m venv --copies "$VENV"
fi

printf '%s\n' /home/darren/src/arbiter/src > "$VENV/lib/python3.12/site-packages/arbiter.pth"

echo "=== 2. Installing PyTorch 2.12 + cu130 and flash-attn ==="
"$VENV/bin/pip" install --upgrade pip setuptools wheel ninja
"$VENV/bin/pip" install torch==2.12.0+cu130 torchvision==0.27.0+cu130 --index-url https://download.pytorch.org/whl/cu130
"$VENV/bin/pip" install flash-attn==2.8.4 --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130

echo "=== 3. Installing Dependencies (transformers pinned to 4.57.6 for DinoV3) ==="
"$VENV/bin/pip" install transformers==4.57.6 huggingface_hub \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless trimesh \
    tensorboard pandas lpips zstandard kornia timm pillow plyfile moderngl glcontext

echo "Installing utils3d (pinned)..."
"$VENV/bin/pip" install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 --no-build-isolation

echo "=== 4. Pinned Upstream TRELLIS.2 Source (SHA 75fbf01) ==="
if [ ! -d "$SRC" ]; then
    git clone https://github.com/microsoft/TRELLIS.2.git "$SRC"
fi
cd "$SRC"
git checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
git submodule update --init --recursive
echo "$SRC" > "$VENV/lib/python3.12/site-packages/trellis2.pth"

echo "=== 5. Building o-voxel ==="
cd "$SRC/o-voxel"
"$VENV/bin/pip" install -e . --no-build-isolation

echo "=== 6. Building FlexGEMM (pinned SHA 6dd94a8) ==="
cd "$EXT_DIR"
if [ ! -d "FlexGEMM" ]; then
    git clone https://github.com/JeffreyXiang/FlexGEMM.git
fi
cd FlexGEMM
git checkout 6dd94a859c26ee8246888502eada3dd8ad85532e
"$VENV/bin/pip" install . --no-build-isolation

echo "=== 7. Building CuMesh (pinned SHA 12289e1) ==="
cd "$EXT_DIR"
if [ ! -d "CuMesh" ]; then
    git clone --recursive https://github.com/JeffreyXiang/CuMesh.git
fi
cd CuMesh
git checkout 12289e1062f0603f2f0d0771b02e1395d247f26f
git submodule update --init --recursive
"$VENV/bin/pip" install . --no-build-isolation

echo "=== 8. Building nvdiffrast (pinned SHA 253ac4f / v0.4.0) ==="
cd "$EXT_DIR"
if [ ! -d "nvdiffrast" ]; then
    git clone https://github.com/NVlabs/nvdiffrast.git
fi
cd nvdiffrast
git checkout 253ac4fcea7de5f396371124af597e6cc957bfae
"$VENV/bin/pip" install . --no-build-isolation

echo "=== 9. Building nvdiffrec (pinned SHA b296927 / renderutils branch) ==="
cd "$EXT_DIR"
if [ ! -d "nvdiffrec" ]; then
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git
fi
cd nvdiffrec
git checkout b296927cc7fd01c2ac1087c8065c4d7248f72da4
"$VENV/bin/pip" install . --no-build-isolation

echo "=== 10. Downloading Official Checkpoints and Setting Up Local Pipeline Config ==="
"$VENV/bin/python" "$SCRIPT_DIR/download_trellis2.py"

echo "=== 11. CPU Import Verification (Class inspection only, no model allocation) ==="
"$VENV/bin/python" - << 'PYEOF'
import inspect
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

import flash_attn
print(f"flash_attn: {flash_attn.__version__}")

import o_voxel
print("o_voxel: OK")

import flex_gemm
print("flex_gemm: OK")

import cumesh
print("cumesh: OK")

import nvdiffrast.torch as dr
print("nvdiffrast: OK")

import nvdiffrec_render
print("nvdiffrec_render: OK")

import trellis2
print("trellis2: OK")

from transformers import DINOv3ViTModel
assert hasattr(DINOv3ViTModel, 'from_pretrained'), "Must have from_pretrained class method"
print("DINOv3ViTModel class verified!")
print("ALL VERIFICATIONS PASSED!")
PYEOF

echo "=== TRELLIS.2 Spark GB10 Setup COMPLETED SUCCESSFULLY ($(date)) ==="
