# Local 3D model selection (2026-09-21)

## Recommendation

Use **Microsoft TRELLIS.2-4B** as the default for image-conditioned, colorful textured assets. This is an engineering recommendation for the requested cartoon brain coral, not a claim that every candidate was benchmarked on Spark. It combines geometry generation with base color, roughness, metallic and opacity attributes, exports a PBR GLB, and its upstream project uses MIT licensing. The bundled pipeline also uses DINOv3 and BRIA RMBG-2.0; their separate access and license terms still apply, so the whole dependency stack must not be described as MIT-only.

The official project reports a 24 GB GPU minimum and tests on A100/H100. Its quoted 3/17/60-second generation timings for 512/1024/1536 resolutions are **H100 results, not Spark estimates**. DGX Spark has enough total memory but its ARM64 CPU and GB10/CUDA 13 require separate dependency and inference verification. Custom CuMesh, FlexGEMM, O-Voxel and rasterization extensions are the main deployment risk.

## Alternatives

| Candidate | Relevant strengths | Reason not selected as default |
|---|---|---|
| TRELLIS.2-4B | Integrated detailed geometry and PBR materials; MIT project; 512–1536 output pipeline | Native image-to-3D rather than direct text; ARM64/CUDA extension work |
| Hunyuan3D 2.1 | Strong open-weight shape plus PBR texture pipeline; upstream states 10 GB shape, 21 GB texture, 29 GB combined | More restrictive community license, including territorial exclusions; the published comparison predates TRELLIS.2 |
| TripoSG | Open image-to-shape system with explicit support for cartoons/sketches; detailed geometry | Its principal release addresses shape, not the complete colorful PBR asset requested here |
| Meshy | Convenient hosted text/image-to-3D product | No locally deployable Meshy model weights were established in this research; a hosted API does not meet the local inference requirement |

Do not equate a newer hosted Hunyuan or Tripo service version with downloadable weights. Evaluate the concrete released checkpoint and license, not just the product name.

## Scope

The Arbiter adapter performs **image-to-3D locally on Spark**. A text-only request first needs a reference image; that is a separate operation, not native TRELLIS.2 text conditioning. Agent-generated reference images use the existing approved Mac mini IGS route. Once a reference image is supplied, mesh and material generation are local; no hosted Meshy/Hunyuan inference API is involved.

The brain-coral acceptance run uses an isolated colorful cartoon reference with visible whole-object silhouette, then verifies the exported GLB's geometry and embedded textures and renders the actual mesh for inspection. A single image cannot establish the true unseen back surface or production-ready/printable topology. STL, if exported, carries geometry only and loses color.

## Primary sources

- [TRELLIS.2 code, requirements, examples and H100 timing table](https://github.com/microsoft/TRELLIS.2)
- [TRELLIS.2 paper](https://arxiv.org/abs/2512.14692)
- [TRELLIS.2-4B checkpoint](https://huggingface.co/microsoft/TRELLIS.2-4B)
- [TRELLIS.2 MIT license](https://github.com/microsoft/TRELLIS.2/blob/main/LICENSE)
- [Hunyuan3D 2.1 code, PBR examples and memory requirements](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
- [Hunyuan3D 2.1 community license](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE)
- [TripoSG code and released capabilities](https://github.com/VAST-AI-Research/TripoSG)
- [Meshy product](https://www.meshy.ai/)

Installation, measured calibration and API usage belong in the adapter setup documentation; do not substitute upstream benchmark claims for a recorded Arbiter job.
