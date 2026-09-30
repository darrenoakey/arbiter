# Bespoke Nimble scoring through Arbiter

This adapter pins `bespokelabs/Bespoke-Nimble-9B` at `bd792f44ec8e265be861bfcdf4e05967ffe0e858`, Qwen3.5-9B at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, and Nimble scorer/prompt code at `62076b4f2d365b5879dafcf7f6dd072a1fe76df7`. It uses BF16, raw temperature 1.0, and the official candidate-only CUDA scorer. The worker verifies the merge manifest and every shard hash before loading.

Only the frozen `decision-v4/development` and `transfer-v4/development` files are accepted. Their SHA-256 and record counts are pinned in `nimble_score.py`; modified input and locked partitions are rejected before inference. Labels are added to saved outputs only after scoring. Never send a locked-test file to this job.

## Worker environment

The model is loaded only inside the registered Arbiter worker. Never run a Python/Torch model command directly on Spark. Before registering/releasing this worker, create its isolated environment on Spark from the existing Arbiter CUDA environment so it inherits the working CUDA-enabled Torch wheel:

```bash
cd /home/darren/src/arbiter
cp -a --link venvs/routing-decide venvs/nimble-scorer
uv pip install --python venvs/nimble-scorer/bin/python -r src/arbiter/nimble_score_requirements.txt
```

This is dependency setup only; do not load the model from the shell. The `config/spark/nimble-scorer.model.json` entry selects this venv and stores the merged checkpoint under `/mnt/t9/models/nimble-scorer-bd792/merged`. On first Arbiter load, the worker fetches public pinned snapshots, validates the prompt contract, and merges the adapter into BF16. Later loads reuse `READY.json` only after verifying all weights.

## Evaluation operation

Stage the checksummed development JSONL with `arbiter_client.stage_file()` and submit one asynchronous `nimble-score` job per suite, with `suite` set to `decision-v4` or `transfer-v4` and `dataset_file` set to the staged inbox path. Record the job IDs and poll status using the Arbiter API; do not hold an agent turn on model load or evaluation. Download the completed ZIP with the Arbiter client and compare its `rows.jsonl` against the saved direct Jev outputs using `(id, question)` and exact candidate order.

The first model download, CPU merge, and full suite scoring can exceed one turn. Always detach the caller, checkpoint the job ID/status and partial files, sleep, then resume by checking the job status. Do not resubmit a completed or in-flight suite except with an explicit fresh-job force flag.
