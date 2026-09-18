"""Single-image SeedVR2 runner over the vendored official code.

Loads the 3B one-step restoration DiT + video VAE once per worker
lifetime and upscales stills 2x (the photo-enhance detail pre-pass).
The vendored inference script this is modelled on
(projects/inference_seedvr2_3b.py) targets video directories; this
wrapper drives the same runner components for one image at a time with
batch 1 and sequence parallelism 1.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

VENDOR = Path(__file__).resolve().parent / "vendor"
DIVISIBILITY = 16


# ##################################################################
# sdpa varlen
# flash_attn_varlen_func(q, k, v, cu_seqlens_q/k, max_seqlen_q/k, ...)
# with (total_tokens, heads, head_dim) tensors. Batch is almost always a
# single concatenated sequence here (one image, text concat'd in), so run
# SDPA per cu_seqlens segment and concatenate the outputs.
def _sdpa_varlen(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None,
                 dropout_p=0.0, softmax_scale=None, causal=False, deterministic=False, **unused):
    import torch
    import torch.nn.functional as F

    if cu_seqlens_q is None:
        # no sequence boundaries: one segment
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0),
            dropout_p=dropout_p, is_causal=causal, scale=softmax_scale,
        )
        return out.squeeze(0).transpose(0, 1)
    bounds_q = [int(x) for x in cu_seqlens_q.tolist()]
    bounds_k = [int(x) for x in (cu_seqlens_k if cu_seqlens_k is not None else cu_seqlens_q).tolist()]
    outputs = []
    for start, end in zip(bounds_q[:-1], bounds_q[1:]):
        if end <= start:
            continue
        k_start, k_end = 0, bounds_k[-1]
        if causal:
            k_end = end
        segment_q = q[start:end].transpose(0, 1).unsqueeze(0)
        segment_k = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        segment_v = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            segment_q, segment_k, segment_v,
            dropout_p=dropout_p, is_causal=causal, scale=softmax_scale,
        )
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0) if outputs else q


# ##################################################################
# install dependency shims
# Inject fake flash_attn and apex modules BEFORE any vendored import so
# the unconditional imports resolve to torch fallbacks.
def _install_shims() -> None:
    import importlib.util
    import torch

    def _spec(name: str):
        # importlib.util.find_spec(name) consults sys.modules first and
        # raises ValueError if the cached module has no __spec__; diffusers
        # probes flash_attn/apex exactly that way at import time.
        return importlib.util.spec_from_loader(name, loader=None)

    if "flash_attn" not in sys.modules:
        flash = types.ModuleType("flash_attn")
        flash.flash_attn_varlen_func = _sdpa_varlen
        flash.__spec__ = _spec("flash_attn")
        sys.modules["flash_attn"] = flash
    if "apex" not in sys.modules:
        apex = types.ModuleType("apex")
        normalization = types.ModuleType("apex.normalization")
        normalization.FusedLayerNorm = torch.nn.LayerNorm
        normalization.FusedRMSNorm = torch.nn.RMSNorm
        normalization.__spec__ = _spec("apex.normalization")
        apex.normalization = normalization
        apex.__spec__ = _spec("apex")
        sys.modules["apex"] = apex
        sys.modules["apex.normalization"] = normalization


# ##################################################################
# seedvr2 upscaler
class SeedVR2Upscaler:
    """Loaded-once, many-images 2x detail recovery for stills."""

    def __init__(self, home: str, checkpoint: str) -> None:
        self.home = Path(home)
        self.checkpoint = checkpoint
        self.runner = None
        self._text_embeds: dict | None = None

    # ##################################################################
    # load
    # shims first, then the vendored runner: DiT (3B fp16 checkpoint) +
    # video VAE, batch 1 / sp 1. Weights live under home/ckpts on spark.
    def load(self) -> None:
        _install_shims()
        vendor = str(VENDOR)
        if vendor not in sys.path:
            sys.path.append(vendor)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29777")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        import torch  # noqa: F401 -- verifies the CUDA stack early
        from common.config import load_config
        from common.seed import set_seed
        from omegaconf import OmegaConf

        from projects.video_diffusion_sr.infer import VideoDiffusionInfer

        os.chdir(self.home)
        config = load_config(str(VENDOR / "configs_3b" / "main.yaml"))
        self._config = config
        self._set_seed = set_seed
        runner = VideoDiffusionInfer(config)
        OmegaConf.set_readonly(runner.config, False)
        from common.distributed import init_torch

        init_torch(cudnn_benchmark=False)
        runner.configure_dit_model(device="cuda", checkpoint=self.checkpoint)
        runner.configure_vae_model()
        if hasattr(runner.vae, "set_memory_limit"):
            runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
        runner.config.diffusion.cfg.scale = 1.0
        runner.config.diffusion.cfg.rescale = 0.0
        runner.config.diffusion.timesteps.sampling.steps = 1
        runner.configure_diffusion()
        self.runner = runner
        pos_path = Path(os.environ.get("SEEDVR_POS_EMB", str(VENDOR / "pos_emb.pt")))
        neg_path = Path(os.environ.get("SEEDVR_NEG_EMB", str(VENDOR / "neg_emb.pt")))
        self._text_embeds = {
            "texts_pos": [torch.load(pos_path, map_location="cpu")],
            "texts_neg": [torch.load(neg_path, map_location="cpu")],
        }
        log.info("seedvr2 3B loaded from %s (pos emb: %s)", self.checkpoint, pos_path)

    # ##################################################################
    # upscale
    # one still, 2x. The frame is top-left trimmed to a multiple of 16
    # (mirroring the script's DivisibleCrop) so the 2x output lines up
    # exactly with the caller's tile grid.
    def upscale(self, image: Image.Image, seed: int = 42, scale: int = 2) -> Image.Image:
        if self.runner is None:
            raise RuntimeError("seedvr2 not loaded")
        import torch
        from torchvision.transforms import Compose, Lambda, Normalize
        from torchvision.io import read_image

        from data.image.transforms.divisible_crop import DivisibleCrop
        from data.image.transforms.na_resize import NaResize
        from data.video.transforms.rearrange import Rearrange
        from common.distributed import get_device

        runner = self.runner
        width, height = image.size
        trim_w = width - (width % DIVISIBILITY)
        trim_h = height - (height % DIVISIBILITY)
        if trim_w <= 0 or trim_h <= 0:
            raise ValueError(f"image {width}x{height} too small for /{DIVISIBILITY}")
        image = image.crop((0, 0, trim_w, trim_h))

        # target pixel area = (2w)*(2h); NaResize scales to that area
        resolution = int(round(((trim_w * scale) * (trim_h * scale)) ** 0.5))
        transform = Compose(
            [
                NaResize(resolution=resolution, mode="area", downsample_only=False),
                Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
                DivisibleCrop((DIVISIBILITY, DIVISIBILITY)),
                Normalize(0.5, 0.5),
                Rearrange("t c h w -> c t h w"),
            ]
        )

        self._set_seed(seed, same_across_ranks=True)
        frame = torch.from_numpy(np.asarray(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        condition = transform(frame.to(get_device()))

        # encode condition latents (VAE phase: DiT off GPU). Autocast like
        # the DiT/decode phases: without it the encoder accumulates fp32
        # feature maps (a 3072 px frame OOMed a 40 GB worker at the
        # inflation concat).
        runner.dit.to("cpu")
        runner.vae.to(get_device())
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
            cond_latents = runner.vae_encode([condition])
        runner.vae.to("cpu")
        runner.dit.to(get_device())

        embeds = {
            "texts_pos": [emb.to(get_device()) for emb in self._text_embeds["texts_pos"]],
            "texts_neg": [emb.to(get_device()) for emb in self._text_embeds["texts_neg"]],
        }

        noise = torch.randn_like(cond_latents[0])
        aug_noise = torch.randn_like(cond_latents[0])
        noises, aug_noises, cond_latents_moved = noise, aug_noise, [c.to(get_device()) for c in cond_latents]

        cond_noise_scale = 0.0
        t = torch.tensor([1000.0], device=get_device()) * cond_noise_scale
        shape = torch.tensor(noises.size()[1:], device=get_device())[None]
        t = runner.timestep_transform(t, shape)
        latent_blur = runner.schedule.forward(cond_latents_moved[0], aug_noises, t)
        conditions = [
            runner.get_condition(noises, task="sr", latent_blur=latent_blur),
        ]

        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
            video_tensors = runner.inference(
                noises=[noises],
                conditions=conditions,
                dit_offload=True,
                **embeds,
            )
        sample = video_tensors[0]
        if sample.ndim == 3:  # c h w (single frame without time dim)
            sample = sample.permute(1, 2, 0)
        else:  # c t h w -> h w c for the first frame
            sample = sample[:, 0].permute(1, 2, 0)
        sample = sample.float().cpu().clamp(-1, 1).mul_(0.5).add_(0.5).mul_(255).round().to(torch.uint8).numpy()

        # wavelet colour fix against the (resized) input, as in the script;
        # optional — a mismatch must never kill the tile.
        try:
            from projects.video_diffusion_sr.color_fix import wavelet_reconstruction

            restored_t = torch.from_numpy(sample).permute(2, 0, 1).unsqueeze(0).float()
            input_t = torch.from_numpy(np.asarray(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).float()
            if restored_t.shape[-2:] == input_t.shape[-2:]:
                fixed = wavelet_reconstruction(restored_t, input_t)
                sample = fixed.squeeze(0).clamp(0, 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        except Exception as err:
            log.warning("seedvr2 colour fix skipped: %s", err)

        runner.dit.to("cpu")
        torch.cuda.empty_cache()
        runner.dit.to(get_device())
        out_w, out_h = sample.shape[1], sample.shape[0]
        if out_w < trim_w * scale or out_h < trim_h * scale:
            raise RuntimeError(f"seedvr2 returned {out_w}x{out_h}, expected at least {trim_w * scale}x{trim_h * scale}")
        result = Image.fromarray(sample, "RGB")
        # exact 2x contract for the tile blender: crop the top-left 2x box
        return result.crop((0, 0, trim_w * scale, trim_h * scale))

    # ##################################################################
    # close
    def close(self) -> None:
        if self.runner is None:
            return
        import torch

        self.runner.dit.to("cpu")
        if getattr(self.runner, "vae", None) is not None:
            self.runner.vae.to("cpu")
        self.runner = None
        self._text_embeds = None
        torch.cuda.empty_cache()
        log.info("seedvr2 unloaded")
