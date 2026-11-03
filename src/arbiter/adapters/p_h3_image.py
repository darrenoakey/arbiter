"""PrunaAI/P-H3-Image bf16 — owner-requested comparison adapter (offline preparation).

No upstream pipeline or weight is executed by this module's unit tests.  Before
deployment, the immutable diffusers runtime commit must be reviewed/approved,
its PH3ImageModularPipeline contract must be checked, and the complete pinned
snapshot must be downloaded.  The model is MiniMax H3 Community License
software; its territory restrictions apply.  This comparison does not change
IGS/generate_image/SDK routing.

The card specifies 8 steps, no CFG, prompt upsampling on by default, 8 MP for
text-to-image.  Defaults here deliberately disable the upsampler and use 1024²
so the house botanist/Sydney/LUMEN jobs are like-for-like.  Native 2912² jobs
must be labelled as a separate arm, never compared as equal-resolution speed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from arbiter.adapters.base import CancelledException, InferenceError, LoadError, ModelAdapter
from arbiter.adapters.flux3_klein import write_png_durably
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

PH3_HF_REPO = "PrunaAI/P-H3-Image"
PH3_REVISION = "7f98acef503cf2bcfba821a78cb1cf1a9928b2df"
DEFAULT_WEIGHTS_DIR = Path("/mnt/t9/models/p-h3-image/P-H3-Image")
DEFAULT_STEPS = 8
DEFAULT_SIDE = 1024
NATIVE_SIDE = 2912
MIN_SIDE = 256
MAX_SIDE = 4096
SIDE_MULTIPLE = 32
MAX_TOTAL_PIXELS = 16_777_216  # model card says it can be pushed to 16 MP
MAX_SEED = 2**63 - 1
MAX_PROMPT_CHARS = 16000
SAMPLE_SIGMAS = (1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0)
EXPECTED_DIM = 5376
EXPECTED_LAYERS = 50
EXPECTED_PIPELINE_CLASS = "PH3ImageModularPipeline"
DIFFUSERS_DECLARED_VERSION = "0.42.0.dev0"

# Official HF expanded-tree response at PH3_REVISION.  Pinning these directly
# permits offline verification without executing or fetching anything on load.
_SNAPSHOT = json.loads(r'''[{"type":"directory","oid":"64900ecb4c4209ec6026763c0e6d8ba30fe2d374","size":0,"path":"assets","lastCommit":{"id":"f1b561915c76203a95c89234673fb8150efe701f","title":"Upload 4 files","date":"2026-11-03T15:41:26.000Z"}},{"type":"directory","oid":"2f1a6512212fa0900fcfbd09f25695bc46d1df2b","size":0,"path":"audio_scheduler","lastCommit":{"id":"f4f5282a0fec218d0e7aa7953f6833a6794c53ed","title":"Set shift_provided_sigmas in the scheduler configs","date":"2026-10-30T11:32:05.000Z"}},{"type":"directory","oid":"c56ae48e35969888a2a39393a18a1134a65295f3","size":0,"path":"processor","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"}},{"type":"directory","oid":"2b9f66c7d3c9b0075c5701cbd3baf74eb73d2d79","size":0,"path":"prompt_upsampler","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"}},{"type":"directory","oid":"86e91fd0e447076bf2b16aeb539ea0ee23fb0f56","size":0,"path":"prompt_upsampler_processor","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"}},{"type":"directory","oid":"2f1a6512212fa0900fcfbd09f25695bc46d1df2b","size":0,"path":"scheduler","lastCommit":{"id":"f4f5282a0fec218d0e7aa7953f6833a6794c53ed","title":"Set shift_provided_sigmas in the scheduler configs","date":"2026-10-30T11:32:05.000Z"}},{"type":"directory","oid":"15b3e41d9e4280affb5b55380ebce0c8f257252a","size":0,"path":"text_connector","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"}},{"type":"directory","oid":"53239042cd2efebdd374138b5b7971803d0748ac","size":0,"path":"text_encoder","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"}},{"type":"directory","oid":"4737a36f616939cab6d43e2bc6d2cc577d21571f","size":0,"path":"tokenizer","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"}},{"type":"directory","oid":"3c0a75747d657b2ec5bd81aa2912adbb1d2d3db2","size":0,"path":"transformer","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"}},{"type":"directory","oid":"a7e9ed78c30c630cf28dde9e0fa9ba719ac9e110","size":0,"path":"vae","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"}},{"type":"file","oid":"f9970755ac53f31c70a950a70b0f1c92725e9ac8","size":2323,"path":".gitattributes","lastCommit":{"id":"f1b561915c76203a95c89234673fb8150efe701f","title":"Upload 4 files","date":"2026-11-03T15:41:26.000Z"}},{"type":"file","oid":"c389b45855337ccec8ddeb389f4fe902abcd0b19","size":17604,"path":"LICENSE","lastCommit":{"id":"1458c0a8991a8f1e34f80b3a1c722c1e2f665c60","title":"Add license and model card metadata","date":"2026-10-28T15:22:41.000Z"}},{"type":"file","oid":"f938136e3adacfd92be087f6e113b5d6d97f678f","size":11544,"path":"LICENSE-APACHE","lastCommit":{"id":"85eaa44b49dbaec668198f43923cdaced0418a1d","title":"Add the Apache 2.0 license and a notice","date":"2026-10-28T17:19:44.000Z"}},{"type":"file","oid":"c2d50e1553f7fab7fe68b7295ffa03454adfe460","size":696,"path":"NOTICE","lastCommit":{"id":"85eaa44b49dbaec668198f43923cdaced0418a1d","title":"Add the Apache 2.0 license and a notice","date":"2026-10-28T17:19:44.000Z"}},{"type":"file","oid":"06401d6257bacedf46419ccdf5d909d4625b4681","size":19251,"path":"README.md","lastCommit":{"id":"7f98acef503cf2bcfba821a78cb1cf1a9928b2df","title":"Update README.md","date":"2026-11-03T16:08:56.000Z"}},{"type":"file","oid":"888d92b475b388e92012f2609de7ee44c52eadf6","size":117028,"lfs":{"oid":"3b7cf9ec4e1a4faa753445981f67426d815937fb596bc6cb36b145dff6c88954","size":117028,"pointerSize":131},"xetHash":"334c7d35028f3f984398f33772e885f38443051fd358fc174813a00f31b90ad7","path":"assets/P-H3-Image-latency.png","lastCommit":{"id":"f1b561915c76203a95c89234673fb8150efe701f","title":"Upload 4 files","date":"2026-11-03T15:41:26.000Z"}},{"type":"file","oid":"9f0292926afddbfa3ed6dd31c7540bc2b185e87b","size":137997,"lfs":{"oid":"3820371e94b4babaf5731617750febce2a3eb5a7a4e7caa9ab5be3561b43d7ed","size":137997,"pointerSize":131},"xetHash":"681995e4dfcf8e3fb543a4c7c083a042fb75d4f00958410964fa341664992d9f","path":"assets/P-H3-Image-memory.png","lastCommit":{"id":"f1b561915c76203a95c89234673fb8150efe701f","title":"Upload 4 files","date":"2026-11-03T15:41:26.000Z"}},{"type":"file","oid":"85b394614ed0b41620f6004c1602e814f593bb56","size":120023,"lfs":{"oid":"22c62534e0f276df0cc0902c0130c403ae9267ecb582bc99adf0eae22c4c0347","size":120023,"pointerSize":131},"xetHash":"a57d7a2fcbcbbc91196011c67c34a0205f404aa78f118cf8a40d9c1f3bd60f7b","path":"assets/P-H3-Image-weights.png","lastCommit":{"id":"f1b561915c76203a95c89234673fb8150efe701f","title":"Upload 4 files","date":"2026-11-03T15:41:26.000Z"}},{"type":"file","oid":"e51e8c6858824953ff4700d0adf5c8110be91b60","size":624068,"lfs":{"oid":"24e01545a0cd9716417d41cb5e134847a21b2d475bc29223dc590dc1dd8100d8","size":624068,"pointerSize":131},"xetHash":"2b66c5ee23d91566fd4eb35b61c2b28139a2c3e4591efe3a71b7b5515aeb34e9","path":"assets/edits.jpg","lastCommit":{"id":"90dba41ae56f746ccd0c668d4d73305669fe3097","title":"Add model card","date":"2026-10-29T10:36:10.000Z"}},{"type":"file","oid":"5a3b4464c68307dcb414f3c137df0db9dce799a4","size":761139,"lfs":{"oid":"ee0c9775b222d2be273fe3303fec57ad61c44309303f72005c768ca29e0cd64a","size":761139,"pointerSize":131},"xetHash":"6c307afa031476a8c3d8561fcdbc2f4bb21fccfa56b4b3fbfced3b05783e6e85","path":"assets/multi_image.jpg","lastCommit":{"id":"90dba41ae56f746ccd0c668d4d73305669fe3097","title":"Add model card","date":"2026-10-29T10:36:10.000Z"}},{"type":"file","oid":"fc47cba712156c7491a24af002d807b143f5daf6","size":1639607,"lfs":{"oid":"09439720c1684b11be7abd49a302f620cbd25e2d503661175a6635c62fc0ed51","size":1639607,"pointerSize":132},"xetHash":"6c5ea2316d54ca4e828c36225445db52ef75f17b1321e591df3c82e931430748","path":"assets/p-h3-image.jpg","lastCommit":{"id":"f1b561915c76203a95c89234673fb8150efe701f","title":"Upload 4 files","date":"2026-11-03T15:41:26.000Z"}},{"type":"file","oid":"71179a243ed8d57b350705e6f2c575add64e5d38","size":567255,"lfs":{"oid":"12f9fa3963b0354105685fcfc46ea38af9fbace472d58df8cbbb990d09270eaf","size":567255,"pointerSize":131},"xetHash":"1c800e64906cf69160d99c2780478b161415cdf13fb63543c83baad40a218275","path":"assets/prompt.jpg","lastCommit":{"id":"90dba41ae56f746ccd0c668d4d73305669fe3097","title":"Add model card","date":"2026-10-29T10:36:10.000Z"}},{"type":"file","oid":"1bd30ca0a33f7740511820f2ca0df1bdfc66524c","size":601241,"lfs":{"oid":"a887f14fe44ff20276397b322b944958245f0e83ae27567f4276f3e0f9074e71","size":601241,"pointerSize":131},"xetHash":"ac94536b9549a48672a529c4937bbf0bbaa843db5a3e32cf0ee1c46b2e9593f0","path":"assets/resolution_ladder.jpg","lastCommit":{"id":"90dba41ae56f746ccd0c668d4d73305669fe3097","title":"Add model card","date":"2026-10-29T10:36:10.000Z"}},{"type":"file","oid":"7b7ee0a10c7d8190e843c7dbec0d25656291bd52","size":25760,"path":"assets/resolutions.png","lastCommit":{"id":"90dba41ae56f746ccd0c668d4d73305669fe3097","title":"Add model card","date":"2026-10-29T10:36:10.000Z"}},{"type":"file","oid":"0c476447fff5a930379a2f52c2e95f70ae8e81e8","size":366751,"lfs":{"oid":"22821811d9936a800b5b8f178bc0ca2865a208a0bc40a2ce86f471083a130d92","size":366751,"pointerSize":131},"xetHash":"119ebcfd6f8b1bcf4a2c75d0c0449d8dc937a16549ed3207c24fa1fa8268f23f","path":"assets/styles.jpg","lastCommit":{"id":"90dba41ae56f746ccd0c668d4d73305669fe3097","title":"Add model card","date":"2026-10-29T10:36:10.000Z"}},{"type":"file","oid":"1480d9378e244f39670360ff206baae3f5c42ac2","size":132,"path":"audio_scheduler/scheduler_config.json","lastCommit":{"id":"f4f5282a0fec218d0e7aa7953f6833a6794c53ed","title":"Set shift_provided_sigmas in the scheduler configs","date":"2026-10-30T11:32:05.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=1480d9378e244f39670360ff206baae3f5c42ac2&utm_source=huggingface"},"avScan":{"status":"safe","message":"","version":"1.5.4/28169"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"f6ed0aa183479e467a19fea8238caf8fa3936bcf","size":3525,"path":"modular_model_index.json","lastCommit":{"id":"ee4a685de1fbd4337a50bcf9b3c407e4f8463d49","title":"Install diffusers from GitHub and use the new repo name","date":"2026-11-03T14:44:16.000Z"}},{"type":"file","oid":"124386803f142761528f710e77ae483f5f8c4fc4","size":5292,"path":"processor/chat_template.jinja","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=124386803f142761528f710e77ae483f5f8c4fc4&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"43c4343ec493f200e452f87a7c6649ecdfcce1ac","size":1220,"path":"processor/processor_config.json","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"safe","message":"This file has no security findings.","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=43c4343ec493f200e452f87a7c6649ecdfcce1ac&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"c7afbed2efcdf019f88ab0572ec29d3bf595dfe2","size":11422650,"lfs":{"oid":"be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506","size":11422650,"pointerSize":133},"xetHash":"dde3c5a4d468da6fe61f11ffa8f04af2d101f8f4a0fe118c37bd31bb8c3017b6","path":"processor/tokenizer.json","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"safe","message":"This file has no security findings.","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=c7afbed2efcdf019f88ab0572ec29d3bf595dfe2&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"safe","message":"0/76 engines detect it as malicious.","reportLink":"https://www.virustotal.com/gui/file/be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506?utm_source=huggingface"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"42bc491eccb6f2ad617f7fe78ea277b8d37c44fc","size":734,"path":"processor/tokenizer_config.json","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"safe","message":"This file has no security findings.","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=42bc491eccb6f2ad617f7fe78ea277b8d37c44fc&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"23f1a71d36887b2cba20ab073400aad7d1146dd0","size":2883,"path":"prompt_upsampler/config.json","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=23f1a71d36887b2cba20ab073400aad7d1146dd0&utm_source=huggingface"},"avScan":{"status":"safe","message":"","version":"1.5.4/28167"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"be823ebf1b2a1b9cde81f8e9efb7f4d0e479e3d0","size":116,"path":"prompt_upsampler/generation_config.json","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"safe","message":"This file has no security findings.","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=be823ebf1b2a1b9cde81f8e9efb7f4d0e479e3d0&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"400467f07125cc6b3c645cbf93cf809f28890b34","size":4942706120,"lfs":{"oid":"5af4cf75e63a8d7cb96d03d25add224b243380f04a8c83ef4b47c40a50e36962","size":4942706120,"pointerSize":135},"xetHash":"deb333577e13fc712b6c68229475b51519066e165dd5da7a5cf14e8b108d9032","path":"prompt_upsampler/model-00001-of-00004.safetensors","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=400467f07125cc6b3c645cbf93cf809f28890b34&utm_source=huggingface"},"avScan":{"status":"unscanned","message":"","version":"1.5.4/28167"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/5af4cf75e63a8d7cb96d03d25add224b243380f04a8c83ef4b47c40a50e36962?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"790dd2a95f558147d7866cd81d5db8b96c5d2204","size":4987757928,"lfs":{"oid":"ffb3821101877085712f6d51510bf0796d7b5bd50d3804f299472ce94b449302","size":4987757928,"pointerSize":135},"xetHash":"d314e8459f39b7352e6791dabd4b020057f6f8aa7794c7a1bbdbb14ae26cca11","path":"prompt_upsampler/model-00002-of-00004.safetensors","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=790dd2a95f558147d7866cd81d5db8b96c5d2204&utm_source=huggingface"},"avScan":{"status":"unscanned","message":"","version":"1.5.4/28167"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/ffb3821101877085712f6d51510bf0796d7b5bd50d3804f299472ce94b449302?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"8dcff4c87e84f7f7b4328ed7ba89348ebb8eb7ab","size":4954810240,"lfs":{"oid":"078775ad9f57fac960fd5bf53754b9e4b07dd403e60b15de15998e876926a312","size":4954810240,"pointerSize":135},"xetHash":"b98d05d6b29237904917e969ba040a5ba72f7771bc6c552d9bfdf284cf1388a7","path":"prompt_upsampler/model-00003-of-00004.safetensors","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=8dcff4c87e84f7f7b4328ed7ba89348ebb8eb7ab&utm_source=huggingface"},"avScan":{"status":"unscanned","message":"","version":"1.5.4/28167"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/078775ad9f57fac960fd5bf53754b9e4b07dd403e60b15de15998e876926a312?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"98da82f3ea58615b34c1f168682d653cdf3bca3a","size":3934446832,"lfs":{"oid":"24196f3a3680e0d72270fa6c7d27379690d5887fa4af5bcd1f5beecdd843fbcf","size":3934446832,"pointerSize":135},"xetHash":"b28b421aece1c03f4ac898da3b29139a8f8016d3832d9a261575e929371ac752","path":"prompt_upsampler/model-00004-of-00004.safetensors","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=98da82f3ea58615b34c1f168682d653cdf3bca3a&utm_source=huggingface"},"avScan":{"status":"unscanned","message":"","version":"1.5.4/28167"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/24196f3a3680e0d72270fa6c7d27379690d5887fa4af5bcd1f5beecdd843fbcf?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"778b7bbb7ebec3f1e100b1059b1c6d54bd00e3c1","size":69253,"path":"prompt_upsampler/model.safetensors.index.json","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"safe","message":"This file has no security findings.","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=778b7bbb7ebec3f1e100b1059b1c6d54bd00e3c1&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"a585dec894e63da457d9440ec6aa7caa16d20860","size":7756,"path":"prompt_upsampler_processor/chat_template.jinja","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=a585dec894e63da457d9440ec6aa7caa16d20860&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"43c4343ec493f200e452f87a7c6649ecdfcce1ac","size":1220,"path":"prompt_upsampler_processor/processor_config.json","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"safe","message":"This file has no security findings.","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=43c4343ec493f200e452f87a7c6649ecdfcce1ac&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"a200a8dd650a52c17957673398767ac34b96dbeb","size":19989339,"lfs":{"oid":"a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f","size":19989339,"pointerSize":133},"xetHash":"e558cdad9a88d177076f9525ed4ef2d55dfd5dafafd6baac7c6105590ab49b7b","path":"prompt_upsampler_processor/tokenizer.json","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"safe","message":"This file has no security findings.","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=a200a8dd650a52c17957673398767ac34b96dbeb&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"safe","message":"0/76 engines detect it as malicious.","reportLink":"https://www.virustotal.com/gui/file/a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f?utm_source=huggingface"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"be6a686f511579ab55a589555505790c31f3aa7a","size":1507,"path":"prompt_upsampler_processor/tokenizer_config.json","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"safe","message":"This file has no security findings.","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=be6a686f511579ab55a589555505790c31f3aa7a&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"1480d9378e244f39670360ff206baae3f5c42ac2","size":132,"path":"scheduler/scheduler_config.json","lastCommit":{"id":"f4f5282a0fec218d0e7aa7953f6833a6794c53ed","title":"Set shift_provided_sigmas in the scheduler configs","date":"2026-10-30T11:32:05.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=1480d9378e244f39670360ff206baae3f5c42ac2&utm_source=huggingface"},"avScan":{"status":"safe","message":"","version":"1.5.4/28169"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"c24e2ef2f5d9fa79330b3ef2c2b459acb4eeb17c","size":208,"path":"text_connector/config.json","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=c24e2ef2f5d9fa79330b3ef2c2b459acb4eeb17c&utm_source=huggingface"},"avScan":{"status":"safe","message":"","version":"1.5.4/28166"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"88355d58b1fee547b98b9299b68af20b802207bc","size":262308352,"lfs":{"oid":"4c4ad8cbf37db8a2da22cdd9905e566e259cf3477ea242f6aee1cf3fb695f070","size":262308352,"pointerSize":134},"xetHash":"3d5345075316fdb3c1a3fd06302e52cdcc0310b8d19590509709df0701bbbc24","path":"text_connector/diffusion_pytorch_model.safetensors","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=88355d58b1fee547b98b9299b68af20b802207bc&utm_source=huggingface"},"avScan":{"status":"safe","message":"","version":"1.5.4/28166"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/4c4ad8cbf37db8a2da22cdd9905e566e259cf3477ea242f6aee1cf3fb695f070?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"5d62f1fc7e3f9c57fc575c2ca6c9ffae995b41f7","size":1696,"path":"text_encoder/config.json","lastCommit":{"id":"288ce44c9249ae5bd0d84bc05d5cf146761b3341","title":"Add text encoder, VAE and decoder","date":"2026-10-27T09:35:13.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=5d62f1fc7e3f9c57fc575c2ca6c9ffae995b41f7&utm_source=huggingface"},"avScan":{"status":"safe","message":"","version":"1.5.4/28164"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"2b37ede0e7fb7939f7a4e61375966dbcd54e476f","size":242,"path":"text_encoder/generation_config.json","lastCommit":{"id":"288ce44c9249ae5bd0d84bc05d5cf146761b3341","title":"Add text encoder, VAE and decoder","date":"2026-10-27T09:35:13.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=2b37ede0e7fb7939f7a4e61375966dbcd54e476f&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"0d39124a2a00d13afa03c0799746561966826a30","size":4990822120,"lfs":{"oid":"be8454bc80eb7b91da25273ab71e0f4a627cf1d5588e91da2e052297da8879cd","size":4990822120,"pointerSize":135},"xetHash":"4900f3c5de8f34bc536ca2d0deff74d0d464d748d7d2536a5913560c121de0af","path":"text_encoder/model-00001-of-00002.safetensors","lastCommit":{"id":"288ce44c9249ae5bd0d84bc05d5cf146761b3341","title":"Add text encoder, VAE and decoder","date":"2026-10-27T09:35:13.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=0d39124a2a00d13afa03c0799746561966826a30&utm_source=huggingface"},"avScan":{"status":"unscanned","message":"","version":"1.5.4/28165"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/be8454bc80eb7b91da25273ab71e0f4a627cf1d5588e91da2e052297da8879cd?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"ac0af7476c86f5b9b226e26f1e44427d2d677b47","size":3884897208,"lfs":{"oid":"d3a4781fcec3eeaf8ba5adef647df3d21eed238dfc6be5e4aa96cf941cbe61b1","size":3884897208,"pointerSize":135},"xetHash":"7bac8dc427b2a8835e317ac8229e1b8c143d11b15166e16b39d656016915626f","path":"text_encoder/model-00002-of-00002.safetensors","lastCommit":{"id":"288ce44c9249ae5bd0d84bc05d5cf146761b3341","title":"Add text encoder, VAE and decoder","date":"2026-10-27T09:35:13.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=ac0af7476c86f5b9b226e26f1e44427d2d677b47&utm_source=huggingface"},"avScan":{"status":"unscanned","message":"","version":"1.5.4/28165"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/d3a4781fcec3eeaf8ba5adef647df3d21eed238dfc6be5e4aa96cf941cbe61b1?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"910582ba1f6124d908dfa1f716d64d26576cfd6f","size":547,"path":"transformer/config.json","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=910582ba1f6124d908dfa1f716d64d26576cfd6f&utm_source=huggingface"},"avScan":{"status":"safe","message":"","version":"1.5.4/28167"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"188443fa128815b68f83401db825b251e84c0308","size":9717836216,"lfs":{"oid":"b689c640d136d19bf342bf0f8244c54a2943df4accb447873734a006329d4720","size":9717836216,"pointerSize":135},"xetHash":"2033c63a83a393b4033619ced3b40f18b43599d0250803c82f8e4ab07ad7f617","path":"transformer/diffusion_pytorch_model-00001-of-00007.safetensors","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=188443fa128815b68f83401db825b251e84c0308&utm_source=huggingface"},"avScan":{"status":"unscanned","message":"","version":"1.5.4/28167"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/b689c640d136d19bf342bf0f8244c54a2943df4accb447873734a006329d4720?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"fdd387814f0449e4a301ee5c53d8f6dd26e48f2d","size":9500437776,"lfs":{"oid":"96f3806ddce6f9791f2d1c443c0e0326514dbbdc1dcbdd5fbc4d2995a9e3aab9","size":9500437776,"pointerSize":135},"xetHash":"2d05dc637db804732ac4e45edb61740bffd878b0b37e59b2e843559e817e1cb4","path":"transformer/diffusion_pytorch_model-00002-of-00007.safetensors","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=fdd387814f0449e4a301ee5c53d8f6dd26e48f2d&utm_source=huggingface"},"avScan":{"status":"unscanned","message":"","version":"1.5.4/28167"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/96f3806ddce6f9791f2d1c443c0e0326514dbbdc1dcbdd5fbc4d2995a9e3aab9?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"b5befa54e3eeabae371543f6714e84981dfd0e4b","size":9866738392,"lfs":{"oid":"b930512e9f9b2d30190806938dff3bf3ef32b673a7c9e3f4f54b45dc56631eb1","size":9866738392,"pointerSize":135},"xetHash":"18a9c0444a4772649054961e356da45887b63c66797d37ca36ce4ab6aa9c0b81","path":"transformer/diffusion_pytorch_model-00003-of-00007.safetensors","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=b5befa54e3eeabae371543f6714e84981dfd0e4b&utm_source=huggingface"},"avScan":{"status":"unscanned","message":"","version":"1.5.4/28167"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/b930512e9f9b2d30190806938dff3bf3ef32b673a7c9e3f4f54b45dc56631eb1?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"2f088ae3946a203dafd225cc18d9dccc2dc69119","size":9500437816,"lfs":{"oid":"c0f62c509e1fe316c63df8cd762aa221cd794609c404b5f03c3dbbda2161154c","size":9500437816,"pointerSize":135},"xetHash":"02faed7ac08e0b738e8968c87726c904cd1a722dda222f329cc32bb79871f2a5","path":"transformer/diffusion_pytorch_model-00004-of-00007.safetensors","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=2f088ae3946a203dafd225cc18d9dccc2dc69119&utm_source=huggingface"},"avScan":{"status":"unscanned","message":"","version":"1.5.4/28167"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/c0f62c509e1fe316c63df8cd762aa221cd794609c404b5f03c3dbbda2161154c?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"bac1d1e72f8f7418a3d635fb2582fe1c57bdedc5","size":9866738392,"lfs":{"oid":"b1fa4c087263e72a187c3377a606bc2b4da8be2e03d5112c59d72cd4ff056166","size":9866738392,"pointerSize":135},"xetHash":"b6255a4f4d6c00f0f341943b6ab0f4ed1ba3bcaaeeb1ea7686f957e020cbf937","path":"transformer/diffusion_pytorch_model-00005-of-00007.safetensors","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=bac1d1e72f8f7418a3d635fb2582fe1c57bdedc5&utm_source=huggingface"},"avScan":{"status":"unscanned","message":"","version":"1.5.4/28167"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/b1fa4c087263e72a187c3377a606bc2b4da8be2e03d5112c59d72cd4ff056166?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"6ac6b100de6ddc6250a68cb3eda709638137f363","size":9500437816,"lfs":{"oid":"d173d97711b49a81cca22d38ddb3f605dab2d96b9f217077d205b82702322df0","size":9500437816,"pointerSize":135},"xetHash":"ef71ec724397ec8a3f952a5220477fa41a5c251129a7a5d26f6f3e567236b62d","path":"transformer/diffusion_pytorch_model-00006-of-00007.safetensors","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=6ac6b100de6ddc6250a68cb3eda709638137f363&utm_source=huggingface"},"avScan":{"status":"unscanned","message":"","version":"1.5.4/28167"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/d173d97711b49a81cca22d38ddb3f605dab2d96b9f217077d205b82702322df0?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"300cbba7900e8e4d5b0defce16c25928cea8ea8f","size":8327877680,"lfs":{"oid":"f06823dfd540c58b2f9c343eded05424d9d6488bcd6847c73b5fd37caf9f659b","size":8327877680,"pointerSize":135},"xetHash":"a57d0ef3a8167d9896ef02db23203c2523c7e43cfa989767db67ba7e75c0cfc6","path":"transformer/diffusion_pytorch_model-00007-of-00007.safetensors","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=300cbba7900e8e4d5b0defce16c25928cea8ea8f&utm_source=huggingface"},"avScan":{"status":"unscanned","message":"","version":"1.5.4/28167"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/f06823dfd540c58b2f9c343eded05424d9d6488bcd6847c73b5fd37caf9f659b?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"7f14556562e4f37c2d0b9105ddaa722971bfd7c5","size":64489,"path":"transformer/diffusion_pytorch_model.safetensors.index.json","lastCommit":{"id":"fed0ed5b895e894c8cf5636e9d099a3f6596cf36","title":"Use a dense AdaLN and add the prompt upsampler","date":"2026-10-28T16:57:34.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"safe","message":"This file has no security findings.","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=7f14556562e4f37c2d0b9105ddaa722971bfd7c5&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"124386803f142761528f710e77ae483f5f8c4fc4","size":5292,"path":"tokenizer/chat_template.jinja","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=124386803f142761528f710e77ae483f5f8c4fc4&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"c7afbed2efcdf019f88ab0572ec29d3bf595dfe2","size":11422650,"lfs":{"oid":"be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506","size":11422650,"pointerSize":133},"xetHash":"dde3c5a4d468da6fe61f11ffa8f04af2d101f8f4a0fe118c37bd31bb8c3017b6","path":"tokenizer/tokenizer.json","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"safe","message":"This file has no security findings.","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=c7afbed2efcdf019f88ab0572ec29d3bf595dfe2&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"safe","message":"0/76 engines detect it as malicious.","reportLink":"https://www.virustotal.com/gui/file/be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506?utm_source=huggingface"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"42bc491eccb6f2ad617f7fe78ea277b8d37c44fc","size":734,"path":"tokenizer/tokenizer_config.json","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"safe","message":"This file has no security findings.","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=42bc491eccb6f2ad617f7fe78ea277b8d37c44fc&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"a8eee822eb7528c2641011d5c0514a73e051b714","size":64778,"path":"text_encoder/model.safetensors.index.json","lastCommit":{"id":"288ce44c9249ae5bd0d84bc05d5cf146761b3341","title":"Add text encoder, VAE and decoder","date":"2026-10-27T09:35:13.000Z"},"securityFileStatus":{"status":"safe","protectAiScan":{"status":"safe","message":"This file has no security findings.","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=a8eee822eb7528c2641011d5c0514a73e051b714&utm_source=huggingface"},"avScan":{"status":"safe","message":"No security issues detected","reportLink":"https://fdtn.ai/ai-supply-chain/hugging-face?utm_source=huggingface"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"unscanned","message":"Not a machine-learning model"}}},{"type":"file","oid":"264176ff6979d50961c954649c4c3d7c727b0463","size":1971,"path":"vae/config.json","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=264176ff6979d50961c954649c4c3d7c727b0463&utm_source=huggingface"},"avScan":{"status":"safe","message":"","version":"1.5.4/28164"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned"},"jFrogScan":{"status":"queued","message":""}}},{"type":"file","oid":"d296a2b1323b8c81cb14a28f5fa49a8a6f125f7d","size":5559055392,"lfs":{"oid":"264964208bea153ca6366570b66d793c75b0fc64716fb6515a41b54e3dce6bf1","size":5559055392,"pointerSize":135},"xetHash":"931277ab9dea5437092942c40be18569bfdb75f0937c5e290daf2da4d7083267","path":"vae/diffusion_pytorch_model.safetensors","lastCommit":{"id":"f678d26f599b464fc384faea1ce3163a6d27175d","title":"Convert to the diffusers modular layout","date":"2026-10-27T11:39:38.000Z"},"securityFileStatus":{"status":"queued","protectAiScan":{"status":"unscanned","reportLink":"https://insights-db.paloaltonetworks.com/models/PrunaAI/P-H3-Image/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/files?blob-id=d296a2b1323b8c81cb14a28f5fa49a8a6f125f7d&utm_source=huggingface"},"avScan":{"status":"unscanned","message":"","version":"1.5.4/28166"},"pickleImportScan":{"status":"unscanned","pickleImports":[],"version":"0.0.0"},"virusTotalScan":{"status":"unscanned","message":"File not scanned in VirusTotal","reportLink":"https://www.virustotal.com/gui/file/264964208bea153ca6366570b66d793c75b0fc64716fb6515a41b54e3dce6bf1?utm_source=huggingface"},"jFrogScan":{"status":"queued","message":""}}}]''')

_REQUIRED_DIRS = (
    "audio_scheduler", "processor", "prompt_upsampler",
    "prompt_upsampler_processor", "scheduler", "text_connector",
    "text_encoder", "tokenizer", "transformer", "vae",
)
_UNSUPPORTED_PARAMS = (
    "image", "image_file", "images", "image_files", "references",
    "negative_prompt", "guidance_scale", "cfg_scale", "true_cfg_scale",
    "sigmas", "num_images_per_prompt", "offload", "model_kind",
    "output_resolution", "trust_remote_code",
)


@dataclass(frozen=True)
class PH3Request:
    prompt: str
    seed: int
    width: int
    height: int
    prompt_upsampling: bool


def snapshot_files() -> dict[str, dict]:
    """Public immutable metadata; no Hugging Face cache or network required."""
    return {
        row["path"]: {k: row[k] for k in ("oid", "size", "lfs") if k in row}
        for row in _SNAPSHOT if row.get("type") == "file"
    }


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()
    with path.open("rb") as f:
        digest.update(f"blob {size}\0".encode())
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot_file(entry: Path, relpath: str) -> str:
    """Check content against official git/LFS ids, not mutable cache metadata."""
    meta = snapshot_files().get(relpath)
    if meta is None:
        raise LoadError(f"p-h3-image has no pinned official metadata for {relpath}")
    target = entry / relpath
    if target.is_symlink() or not target.is_file():
        raise LoadError(f"p-h3-image snapshot is missing a real file: {target}")
    if target.stat().st_size != meta["size"]:
        raise LoadError(f"p-h3-image size mismatch for {relpath}")
    lfs = meta.get("lfs")
    expected = lfs["oid"] if lfs else meta["oid"]
    actual = _sha256(target) if lfs else _git_blob_sha1(target)
    if actual != expected:
        raise LoadError(f"p-h3-image content hash mismatch for {relpath}")
    return actual


def verify_ph3_snapshot(entry: Path) -> dict:
    if entry.is_symlink() or not entry.is_dir():
        raise LoadError(f"p-h3-image weights dir is not a real directory: {entry}")
    files = snapshot_files()
    for name in ("modular_model_index.json", "LICENSE", "LICENSE-APACHE", "NOTICE", *_REQUIRED_DIRS):
        if not (entry / name).exists():
            raise LoadError(f"p-h3-image snapshot is missing required path: {entry / name}")
    required = [
        name for name in files
        if name in ("modular_model_index.json", "LICENSE", "LICENSE-APACHE", "NOTICE")
        or name.split("/", 1)[0] in _REQUIRED_DIRS
    ]
    verified = {name: verify_snapshot_file(entry, name) for name in sorted(required)}
    index = json.loads((entry / "modular_model_index.json").read_text())
    if index.get("_class_name") != EXPECTED_PIPELINE_CLASS:
        raise LoadError(f"p-h3-image unexpected pipeline class: {index.get('_class_name')}")
    if index.get("_diffusers_version") != DIFFUSERS_DECLARED_VERSION:
        raise LoadError(f"p-h3-image runtime contract changed: {index.get('_diffusers_version')}")
    if tuple(index.get("sample_sigmas", ())) != SAMPLE_SIGMAS:
        raise LoadError("p-h3-image sampling grid differs from the pinned 8-step checkpoint")
    cfg = json.loads((entry / "transformer/config.json").read_text())
    if cfg.get("hidden_size") != EXPECTED_DIM or cfg.get("num_layers") != EXPECTED_LAYERS:
        raise LoadError(f"p-h3-image unexpected transformer geometry: {cfg}")
    return {
        "hf_repo": PH3_HF_REPO, "revision": PH3_REVISION,
        "weights_dir": str(entry), "verified_files": len(verified),
        "verification": "official immutable git-blob/LFS content ids",
    }


def parse_request(params: dict) -> PH3Request:
    for key in _UNSUPPORTED_PARAMS:
        if params.get(key) not in (None, False):
            raise InferenceError(f"p-h3-image comparison does not support {key!r}")
    prompt = params.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InferenceError("prompt is required")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise InferenceError(f"prompt exceeds {MAX_PROMPT_CHARS} characters")

    def integer(key: str, default: int) -> int:
        value = params.get(key, default)
        if value is None:
            value = default
        if isinstance(value, bool) or not isinstance(value, int):
            raise InferenceError(f"{key} must be an integer")
        return value

    width = integer("width", DEFAULT_SIDE)
    height = integer("height", DEFAULT_SIDE)
    for side in (width, height):
        if not MIN_SIDE <= side <= MAX_SIDE or side % SIDE_MULTIPLE:
            raise InferenceError(f"width and height must be {MIN_SIDE}-{MAX_SIDE} and multiples of 32")
    if width * height > MAX_TOTAL_PIXELS:
        raise InferenceError(f"width*height must be <= {MAX_TOTAL_PIXELS}")
    seed = integer("seed", 42)
    if not 0 <= seed <= MAX_SEED:
        raise InferenceError(f"seed must be between 0 and {MAX_SEED}")
    steps = integer("steps", DEFAULT_STEPS)
    if steps != DEFAULT_STEPS:
        raise InferenceError("p-h3-image is pinned to its official 8-step sampling grid")
    upsampling = params.get("prompt_upsampling", False)
    if upsampling is None:
        upsampling = False
    if not isinstance(upsampling, bool):
        raise InferenceError("prompt_upsampling must be a boolean")
    if upsampling:
        raise InferenceError("prompt upsampling requires a separately reviewed adapter arm; not implemented")
    return PH3Request(prompt, seed, width, height, upsampling)


@register
class PH3ImageAdapter(ModelAdapter):
    model_id = "p-h3-image-bf16"

    def __init__(self) -> None:
        self._pipe = None
        self._load_info: dict | None = None
        self._renders = 0
        self._lock = threading.Lock()

    def load(self, device: str = "cuda") -> None:
        if device != "cuda":
            raise LoadError(f"{self.model_id} requires cuda")
        declared = os.environ.get("ARBITER_MEMORY_GB")
        if not declared:
            raise LoadError(f"{self.model_id} refuses to load outside an Arbiter CUDA-capped worker")
        started = time.perf_counter()
        weights = verify_ph3_snapshot(DEFAULT_WEIGHTS_DIR)
        verified = time.perf_counter()

        # Runtime approval gate: the pinned snapshot declares diffusers
        # 0.42.0.dev0.  The upstream commit and these API contracts must be
        # reviewed before deployment; merely importing ModularPipeline is not
        # proof of PH3 support.  Removing prompt upsampling follows the card's
        # blocks.sub_blocks.pop("prompt_upsampler") recipe. Upsampling is not
        # yet enabled by this adapter: a distinct approved implementation arm
        # is required before accepting prompt_upsampling=true.
        blocks = ModularPipeline.from_pretrained(
            str(DEFAULT_WEIGHTS_DIR), revision=PH3_REVISION, local_files_only=True
        ).blocks
        blocks.sub_blocks.pop("prompt_upsampler")
        pipe = blocks.init_pipeline(
            PH3_HF_REPO, revision=PH3_REVISION, local_files_only=True
        )
        pipe.load_components(dtype=torch.bfloat16)
        pipe.to(device)
        loaded = time.perf_counter()
        transformer = getattr(pipe, "transformer", None)
        if transformer is None and hasattr(pipe, "components"):
            transformer = pipe.components.get("transformer")
        if transformer is None:
            raise LoadError("p-h3-image runtime exposes no transformer for cancellation/memory checks")
        runtime = {"torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                   "device_name": torch.cuda.get_device_name()}
        self._pipe = pipe
        self._transformer = transformer
        self._renders = 0
        self._load_info = {
            "weights": weights, "runtime": runtime, "declared_memory_gb": float(declared),
            "verify_seconds": round(verified - started, 3),
            "pipeline_load_seconds": round(loaded - verified, 3),
            "load_seconds": round(loaded - started, 3),
            "resident_allocated_bytes": torch.cuda.memory_allocated(),
            "resident_reserved_bytes": torch.cuda.memory_reserved(),
        }

    def unload(self) -> None:
        self._pipe = None
        self._load_info = None
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        if (params.get("_job_type") or self.model_id) != "p-h3-image":
            raise InferenceError("p-h3-image-bf16 serves only the p-h3-image job type")
        request = parse_request(params)
        self._check_cancel(cancel_flag)
        with self._lock:
            if self._pipe is None or self._load_info is None:
                raise InferenceError(f"{self.model_id} is not loaded")
            return self._render(request, output_dir, cancel_flag)

    def _render(self, request: PH3Request, output_dir: Path, cancel_flag: threading.Event) -> dict:
        import torch

        forwards = 0
        def cancel_before_forward(_module, _args, _kwargs):
            nonlocal forwards
            if cancel_flag.is_set():
                raise CancelledException(f"Job cancelled for model {self.model_id}")
            forwards += 1

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        handle = self._transformer.register_forward_pre_hook(cancel_before_forward, with_kwargs=True)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                out = self._pipe(
                    prompt=request.prompt, width=request.width, height=request.height,
                    sigmas=list(SAMPLE_SIGMAS),
                    generator=torch.Generator("cpu").manual_seed(request.seed),
                    output=["images", "prompt"],
                )
            torch.cuda.synchronize()
        finally:
            handle.remove()
        generated = time.perf_counter()
        self._check_cancel(cancel_flag)
        image = out["images"][0] if isinstance(out, dict) else out[0][0]
        write_png_durably(output_dir / "result.png", image)
        written = time.perf_counter()
        self._renders += 1
        info = self._load_info
        gib = float(1024**3)
        return {
            "format": "png", "file": "result.png", "mode": "t2i",
            "model": PH3_HF_REPO, "model_id": self.model_id,
            "prompt": request.prompt, "seed": request.seed,
            "width": image.width, "height": image.height, "steps": DEFAULT_STEPS,
            "sample_sigmas": list(SAMPLE_SIGMAS),
            "precision": {"transformer": "bfloat16", "text_encoder": "bfloat16", "vae": "bfloat16"},
            "comparison": {
                "prompt_upsampling_requested": request.prompt_upsampling,
                "resolution_class": "native-8mp" if (image.width, image.height) == (NATIVE_SIDE, NATIVE_SIDE) else "custom",
                "like_for_like_house_1mp": (image.width, image.height) == (1024, 1024) and not request.prompt_upsampling,
            },
            "provenance": {**info["weights"], **info["runtime"]},
            "timings": {
                "render_index": self._renders, "generate_seconds": round(generated - started, 3),
                "png_write_seconds": round(written - generated, 3), "total_seconds": round(written - started, 3),
                "transformer_forwards": forwards, "load_seconds": info["load_seconds"],
                "verify_seconds": info["verify_seconds"], "pipeline_load_seconds": info["pipeline_load_seconds"],
            },
            "memory": {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / gib, 3),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / gib, 3),
                "declared_gb": info["declared_memory_gb"],
            },
        }

    def estimate_time(self, params: dict) -> float:
        request = parse_request(params)
        return 180.0 * (request.width * request.height / (1024 * 1024))
