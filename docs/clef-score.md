# Cloudflare Clef / Clef-flash scoring through Arbiter

Jobs `clef-score` (model `clef-scorer`, `Cloudflare/clef` @ `2f3de3dd85f379784083b0814d997ab627200f0c`, Qwen3.8-27B backbone, ~55 GB BF16) and `clef-flash-score` (model `clef-flash-scorer`, `Cloudflare/clef-flash` @ `17f0b0ad64efb65d273590632833508766b2aae6`, Qwen3.5-9B, ~19 GB) run Cloudflare's open-weight, Jev/SystemOne-compatible decision models. The adapter pins every weight shard, the joint head and the executed `joint_schema_model.py` release code by SHA-256. It loads BF16 on CUDA and scores one record per forward pass with the release's own `encode_record` and `collate_records`.

## Input

Stage a JSONL file with `arbiter_client.stage_file()`. Each line must be exactly `{"id", "state", "questions"}`, and questions may carry only `type`, `instructions` and `criteria`. Records that carry labels or any other key are rejected, so held-out labels can never reach the worker. Submit params `{"dataset_file": <staged path>, "dataset_sha256": <sha256 of the file>}` and optionally `max_length` (default 16384 tokens; only the state is trimmed). The result is a ZIP containing `rows.jsonl` (per-question option probabilities, input tokens, CUDA-synchronized latency) and `report.json` (pins, runtime, latency summary).

## Worker environment

Both models share `venvs/clef-scorer`, a hardlinked copy of `venvs/nimble-scorer` (torch 2.10 cu130, transformers 5.17, which includes `Qwen3_5ForConditionalGeneration`, plus pillow and torchvision):

```bash
cd /home/darren/src/arbiter && cp -a --link venvs/nimble-scorer venvs/clef-scorer
```

Weights live in Hugging Face caches under `/home/darren/clef-models/clef-2f3de3dd` and `/home/darren/clef-models/clef-flash-17f0b0ad` on the root NVMe. Do not use `/home/darren/models`: it is a symlink to the nearly full `/mnt/t9`, and a Clef download there filled that disk. The first load hashes every shard and writes a `VERIFIED-<revision>.json` marker; later loads trust the marker only while every shard size still matches.

## Long operations

Downloading weights (~74 GB), the first hash-verification load and full-dataset scoring all exceed one agent turn. Run downloads detached (`nohup ... &` with a done marker), submit jobs asynchronously, record the job IDs, and sleep. Never hold a turn polling. Never load the model from a Spark shell; it runs only inside the Arbiter worker.
