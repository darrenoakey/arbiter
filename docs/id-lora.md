# ID-LoRA LTX-2.3 joint video and voice generation

`video-id-lora` maps to `id-lora`, an audited, single-concurrency Python worker.
It generates a new scene and new speech together, conditioned on a first-frame
image and approximately five seconds of the desired speaker's real voice.
This is NOT Sonic, external TTS, a lip-sync pass, or LTX-2.5. One speaker per shot;
multiple independently assigned voices in the same shot are not supported.
Identity resemblance is generative, not a biometric guarantee.

## Contract

First submit `video-id-lora-encode` with `params: {"prompt": "<structured prompt>", "force": true}`. It returns `encoded.pt` and the API supplies its result_path. Encode ALL prompts first to reuse resident Gemma. Download each bundle and stage it for its render; the exact prompt is verified against the bundle.

Stage all three files with `arbiter_client.stage_file()`; the server removes staged
files after completion, so stage again for another job. Submit:

```json
{
  "type": "video-id-lora",
  "source": {"who": "id-lora-client", "why": "salt-hank-parody"},
  "params": {
    "encoded_file": "/mnt/arbiter-store/inbox/<staged-encoded.pt>",
    "image_file": "/mnt/arbiter-store/inbox/<staged-image>",
    "ref_audio_file": "/mnt/arbiter-store/inbox/<staged-audio>",
    "prompt": "[VISUAL]: The man gestures at his sandwich and speaks. [SPEECH]: Salt Hank has solved lunch. [SOUNDS]: Warm amused conversational speech.",
    "width": 512, "height": 512, "num_frames": 121,
    "steps": 30, "seed": 42, "force": true
  }
}
```

Dimensions are explicit multiples of 32, 256–1024; no implicit aspect cropping
calculation changes the requested output size. Frames must be `8*n+1`, maximum
241, at native 25 fps. 161 and 201 frames correspond to 6.44 and 8.04 seconds.
The upstream defaults are 512², 121 frames, 30 Euler steps, seed 42, video CFG3,
audio CFG7, identity guidance3, STG1 and AV bimodal guidance3. The adapter uses
those defaults, BF16, and the authors' one-stage pipeline. Output is `result.mp4`
with native generated audio; the returned metadata includes the upstream pin.
Separate encode52GB and denoise82GB declarations and latency30min are conservative provisional budgets,
not measured performance claims. Runtime deadline4h; no concurrent pipelines.

## Pinned runtime and reuse

Source: https://github.com/ID-LoRA/ID-LoRA at
`13dc9c286a2d3c1a21a7781f9ac0fe7727ec56cd`, checkout with the hash-pinned Spark optimization patch
`/home/darren/src/id-lora-runtime`. Import
`ID-LoRA-2.3/scripts/inference_one_stage.py` in-process, never spawn a CUDA CLI.
Apply `runtime/id_lora/spark.patch` to that exact source before starting workers. Every Python source file is verified against `adapters/id_lora_runtime.json`, including patched files, plus the existing Spark FastPipeline file. The worker redirects upstream prints to stderr to protect the JSON protocol.

Reuse `/mnt/t9/models/ltx2/ltx-2.3-22b-dev.safetensors` (46,149,344,974 bytes),
SHA256 `7ab7225325bc403448ea84b6db2269811a880e5118cd2ee2b6282a93d585016f`.
Reuse Gemma3 12B weights and tokenizer in that model's text_encoder/tokenizer
folders. `/mnt/t9/models/id-lora/gemma` contains a symlink view selecting ONLY
`model*.safetensors`, not the duplicate diffusion_pytorch_model shards, since
upstream recursively loads all safetensors from the chosen model folder.

TalkVid: `AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K` revision
`6eabedadefbcfa6c95092d50006629d9413102fd`, file `lora_weights.safetensors`,
1,157,884,304 bytes, installed `/mnt/t9/models/id-lora/talkvid.safetensors`,
SHA256 `e5af73441743b4852f228b03e444888dff3da80d2666033af2367ab7bda6d8b9`.
LoRA hash, source commit and patched-source manifest are verified on load; verify the large
base checksum during installation. No automatic weight downloads at inference.

## Spark optimizations retained

The encode adapter directly reuses the installed LTX2 `FastPipeline` encoder
lifecycle, resident Gemma+embedding processor, and CPU `encoded.pt` writer;
negative context is cached across prompts. Denoise loads that CPU bundle by the
existing `load_denoise1_input` helper, then transfers only contexts to CUDA.
The scheduler can evict encode before loading denoise (never co-resident Gemma
and dev transformer). Submit encodes together, then renders together.

`runtime/id_lora/spark.patch` ports the exact production loader techniques:
CPU safetensors mmap → heap clone → CUDA, in-place FP32→BF16 conversion (with
copy-on-write for a real registry), indexed shard selection, and the shared
Arbiter HeapTrimGuard. Denoise retains transformer and reference encoders across
shots. Video decode uses the production TilingConfig.default tiled iterator,
with a cached video decoder/audio decoder/vocoder. It does NOT copy the 46GB
transformer CPU→GPU around every decode. Cancellation is checked every step.
The deployed LTX2 dev source uses BF16 and native attention dispatch; no separate
quantization, torch.compile or SageAttention optimization was active to retain.

The required compatibility delta is the authors' joint AV denoising loop and
reference audio tokens. Existing `denoise_video_only` freezes supplied audio
and cannot synthesize new words in a reference voice, so it is deliberately
not reused. Distilled stage2/upscaling is not part of this one-stage release.
The installed FastPipeline source is independently hash-pinned from Spark
ltx2-spark commit692d020 plus its deployed optimizations, not silently replaced.


## Environment recreation

Create a REAL directory venv (not a symlink):
`python3 -m venv --copies /home/darren/src/arbiter/venvs/id-lora`.
Its `lib/python3.12/site-packages/id_lora_runtime.pth` contains these paths in
this order:

```
/home/darren/src/id-lora-runtime/ID-LoRA-2.3/packages/ltx-core/src
/home/darren/src/id-lora-runtime/ID-LoRA-2.3/packages/ltx-pipelines/src
/home/darren/src/id-lora-runtime/ID-LoRA-2.3/packages/ltx-trainer/src
/home/darren/src/arbiter/.venv/lib/python3.12/site-packages
/home/darren/src/arbiter/src
```

The three LTX packages are isolated from existing LTX2/2.5 packages. Shared
binary dependencies are reused read-only to avoid duplicating10GB on a nearly
full root disk: torch/torchaudio2.10.0+cu130, torchvision0.25.0+cu130,
transformers4.57.6, accelerate1.12.0, av17.0.0, einops0.8.2, safetensors0.7.0.
No pip changes to the shared venv. Source installation and weight download run
detached, before deployment. Never import torch on Spark to test installation;
submit an audited job after greenline deployment instead. The generic worker
sets its CUDA cap before `load()`, and the scheduler budgets the entire phased
pipeline throughout inference. Keep all existing guardians and drain rules.

## Verification

CPU tests cover schemas, frame clocks, registry, exact policy allowlisting,
unknown-parameter rejection and single-worker budget. Real GPU validation must
use a genuine reference image/audio through `POST /v1/jobs`; record its job ID,
inspect errors and logs, and ffprobe the completed audio/video before claiming
success. Schema tests are not evidence of a successful render.
