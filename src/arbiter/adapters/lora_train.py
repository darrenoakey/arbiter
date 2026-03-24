"""LoRA fine-tuning adapter — trains a LoRA adapter on chat data using Unsloth/TRL."""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from arbiter.adapters.base import ModelAdapter, InferenceError, CancelledException
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

# Training data and output live here
TRAINING_ROOT = Path("/home/darren/training")
TRAINING_ROOT.mkdir(parents=True, exist_ok=True)


@register
class LoraTrainAdapter(ModelAdapter):
    model_id = "lora-train"

    def __init__(self):
        self._loaded = False

    def load(self, device: str = "cuda") -> None:
        """Pre-import training libraries so they're ready when a job arrives."""
        log.info("Pre-loading training libraries...")
        # Import eagerly so first job doesn't pay the import cost
        import torch
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
        self._device = device
        self._loaded = True
        log.info("Training libraries ready. CUDA: %s, device: %s",
                 torch.cuda.is_available(),
                 torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

    def unload(self) -> None:
        """Release any cached state."""
        log.info("Unloading training adapter.")
        self._loaded = False
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag: threading.Event) -> dict:
        """Run LoRA training. Despite the name 'infer', this trains a model.

        Required params:
            data_dir (str): Path to directory containing train.jsonl (and optionally valid.jsonl)
            model_name (str): HuggingFace model ID (e.g. "unsloth/Meta-Llama-3.1-8B-bnb-4bit")

        Optional params:
            run_name (str): Name for this training run (default: auto-generated)
            lora_rank (int): LoRA rank (default: 16)
            lora_alpha (int): LoRA alpha (default: 32)
            lora_dropout (float): LoRA dropout (default: 0.05)
            learning_rate (float): Learning rate (default: 2e-4)
            batch_size (int): Per-device batch size (default: 4)
            grad_accum_steps (int): Gradient accumulation steps (default: 4)
            num_epochs (int): Number of training epochs (default: 1)
            max_iters (int): Max training steps, overrides epochs if set (default: 0 = use epochs)
            max_seq_length (int): Maximum sequence length (default: 2048)
            warmup_ratio (float): LR warmup ratio (default: 0.03)
            save_steps (int): Save checkpoint every N steps (default: 500)
            eval_steps (int): Evaluate every N steps (default: 500)
            load_in_4bit (bool): Use 4-bit quantization (default: true)
            full_finetune (bool): Full fine-tune instead of LoRA (default: false)
        """
        import torch
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset

        self._check_cancel(cancel_flag)

        # --- Parse params ---
        data_dir = Path(params["data_dir"])
        model_name = params["model_name"]
        run_name = params.get("run_name", f"train-{int(time.time())}")
        lora_rank = params.get("lora_rank", 16)
        lora_alpha = params.get("lora_alpha", 32)
        lora_dropout = params.get("lora_dropout", 0.05)
        learning_rate = params.get("learning_rate", 2e-4)
        batch_size = params.get("batch_size", 4)
        grad_accum_steps = params.get("grad_accum_steps", 4)
        num_epochs = params.get("num_epochs", 1)
        max_iters = params.get("max_iters", 0)
        max_seq_length = params.get("max_seq_length", 2048)
        warmup_ratio = params.get("warmup_ratio", 0.03)
        save_steps = params.get("save_steps", 500)
        eval_steps = params.get("eval_steps", 500)
        load_in_4bit = params.get("load_in_4bit", True)
        full_finetune = params.get("full_finetune", False)

        # --- Validate data ---
        train_file = data_dir / "train.jsonl"
        if not train_file.is_file():
            raise InferenceError(f"Training data not found: {train_file}")

        valid_file = data_dir / "valid.jsonl"
        has_valid = valid_file.is_file()

        # Count samples
        train_count = sum(1 for _ in open(train_file))
        valid_count = sum(1 for _ in open(valid_file)) if has_valid else 0
        log.info("Training data: %d train, %d valid samples", train_count, valid_count)

        # --- Output directory ---
        run_dir = TRAINING_ROOT / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        adapter_output = run_dir / "adapter"
        adapter_output.mkdir(parents=True, exist_ok=True)
        log_file = run_dir / "training.log"

        # Write run config
        run_config = {
            "model_name": model_name,
            "data_dir": str(data_dir),
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "num_epochs": num_epochs,
            "max_iters": max_iters,
            "max_seq_length": max_seq_length,
            "train_samples": train_count,
            "valid_samples": valid_count,
            "load_in_4bit": load_in_4bit,
            "full_finetune": full_finetune,
            "started_at": time.time(),
        }
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

        self._check_cancel(cancel_flag)

        # --- Load model ---
        log.info("Loading model: %s (4bit=%s)", model_name, load_in_4bit)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )

        self._check_cancel(cancel_flag)

        # --- Apply LoRA (or skip for full finetune) ---
        if not full_finetune:
            log.info("Applying LoRA: rank=%d, alpha=%d, dropout=%.3f",
                     lora_rank, lora_alpha, lora_dropout)
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                use_gradient_checkpointing="unsloth",
            )

        self._check_cancel(cancel_flag)

        # --- Load dataset ---
        log.info("Loading dataset from %s", data_dir)
        data_files = {"train": str(train_file)}
        if has_valid:
            data_files["validation"] = str(valid_file)

        dataset = load_dataset("json", data_files=data_files)

        self._check_cancel(cancel_flag)

        # --- Configure trainer ---
        effective_steps = max_iters if max_iters > 0 else -1
        effective_epochs = num_epochs if max_iters <= 0 else -1

        trainer_args = SFTConfig(
            output_dir=str(adapter_output),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs if max_iters <= 0 else 1,
            max_steps=effective_steps,
            max_seq_length=max_seq_length,
            warmup_ratio=warmup_ratio,
            save_steps=save_steps,
            eval_strategy="steps" if has_valid else "no",
            eval_steps=eval_steps if has_valid else None,
            logging_steps=50,
            logging_dir=str(run_dir / "logs"),
            save_total_limit=5,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            seed=42,
            report_to="none",
            dataset_text_field=None,  # We use chat template
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            args=trainer_args,
        )

        # --- Train ---
        log.info("Starting training: %d epochs, batch=%d, grad_accum=%d",
                 num_epochs, batch_size, grad_accum_steps)

        train_result = trainer.train()

        self._check_cancel(cancel_flag)

        # --- Save ---
        log.info("Saving adapter to %s", adapter_output)
        if full_finetune:
            trainer.save_model(str(adapter_output))
        else:
            model.save_pretrained(str(adapter_output))

        tokenizer.save_pretrained(str(adapter_output))

        # --- Write training summary ---
        metrics = train_result.metrics
        summary = {
            "run_name": run_name,
            "model_name": model_name,
            "adapter_path": str(adapter_output),
            "train_loss": metrics.get("train_loss"),
            "train_runtime": metrics.get("train_runtime"),
            "train_samples_per_second": metrics.get("train_samples_per_second"),
            "total_steps": metrics.get("total_flos"),
            "epochs_completed": metrics.get("epoch"),
            "finished_at": time.time(),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        # --- Copy final adapter to output_dir for arbiter file serving ---
        output_dir.mkdir(parents=True, exist_ok=True)
        final_adapter = output_dir / "adapter"
        if final_adapter.exists():
            shutil.rmtree(final_adapter)
        shutil.copytree(adapter_output, final_adapter)

        # Write summary to output too
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        log.info("Training complete: loss=%.4f, runtime=%.1fs",
                 metrics.get("train_loss", 0), metrics.get("train_runtime", 0))

        # Cleanup model from GPU
        del model, trainer
        self._cleanup_gpu()

        return {
            "format": "lora-adapter",
            "run_name": run_name,
            "adapter_path": str(adapter_output),
            "train_loss": metrics.get("train_loss"),
            "train_runtime_seconds": metrics.get("train_runtime"),
            "train_samples": train_count,
            "epochs": metrics.get("epoch"),
        }

    def estimate_time(self, params: dict) -> float:
        """Training is long — estimate based on dataset size and iters."""
        # Very rough: ~0.1s per sample per epoch for 8B LoRA on GB10
        data_dir = Path(params.get("data_dir", ""))
        train_file = data_dir / "train.jsonl"
        try:
            sample_count = sum(1 for _ in open(train_file))
        except (FileNotFoundError, OSError):
            sample_count = 10000  # fallback guess

        num_epochs = params.get("num_epochs", 1)
        max_iters = params.get("max_iters", 0)
        batch_size = params.get("batch_size", 4)

        if max_iters > 0:
            # Rough: 0.5s per step
            return max_iters * 500.0
        else:
            steps = (sample_count * num_epochs) / batch_size
            return steps * 500.0  # ~0.5s per step estimate
