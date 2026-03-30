"""LoRA fine-tuning adapter using Unsloth model + standard HF Trainer."""
from __future__ import annotations

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import shutil
import threading
import time
from pathlib import Path

from arbiter.adapters.base import ModelAdapter, InferenceError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

TRAINING_ROOT = Path("/home/darren/training")
TRAINING_ROOT.mkdir(parents=True, exist_ok=True)

CHAT_TEMPLATE_MAP = {
    "llama-3": "llama-3.1", "llama-3.1": "llama-3.1", "llama-3.2": "llama-3.1",
    "qwen": "qwen-2.5", "gemma": "gemma", "phi": "phi-4", "mistral": "mistral",
}

def _detect_chat_template(model_name):
    lower = model_name.lower()
    for key, template in CHAT_TEMPLATE_MAP.items():
        if key in lower:
            return template
    return "llama-3.1"


@register
class LoraTrainAdapter(ModelAdapter):
    model_id = "lora-train"

    def __init__(self):
        self._loaded = False

    def load(self, device="cuda"):
        log.info("Pre-loading training libraries...")
        import unsloth.models._utils as _u
        _u.has_internet = lambda *a, **kw: False
        import torch
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
        from datasets import Dataset
        self._device = device
        self._loaded = True
        log.info("Training libraries ready.")

    def unload(self):
        self._loaded = False
        self._cleanup_gpu()

    def infer(self, params, output_dir, cancel_flag):
        import torch
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
        from datasets import Dataset

        self._check_cancel(cancel_flag)

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
        chat_template = params.get("chat_template", _detect_chat_template(model_name))

        train_file = data_dir / "train.jsonl"
        if not train_file.is_file():
            raise InferenceError(f"Training data not found: {train_file}")
        valid_file = data_dir / "valid.jsonl"
        has_valid = valid_file.is_file()

        train_count = sum(1 for _ in open(train_file))
        valid_count = sum(1 for _ in open(valid_file)) if has_valid else 0
        log.info("Training data: %d train, %d valid", train_count, valid_count)

        run_dir = TRAINING_ROOT / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        adapter_output = run_dir / "adapter"
        adapter_output.mkdir(parents=True, exist_ok=True)

        (run_dir / "config.json").write_text(json.dumps({
            "model_name": model_name, "lora_rank": lora_rank,
            "learning_rate": learning_rate, "batch_size": batch_size,
            "num_epochs": num_epochs, "max_iters": max_iters,
            "train_samples": train_count, "started_at": time.time(),
        }, indent=2))

        self._check_cancel(cancel_flag)

        log.info("Loading model: %s (4bit=%s)", model_name, load_in_4bit)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit, device_map={"": 0},
        )
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._check_cancel(cancel_flag)

        if not full_finetune:
            log.info("Applying LoRA: rank=%d, alpha=%d", lora_rank, lora_alpha)
            model = FastLanguageModel.get_peft_model(
                model, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                use_gradient_checkpointing="unsloth",
            )
        self._check_cancel(cancel_flag)

        # Tokenize in-process — no multiprocessing, no SFTTrainer
        log.info("Tokenizing dataset in-process...")
        def tokenize_jsonl(path):
            input_ids_list, attention_mask_list, labels_list = [], [], []
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    text = tokenizer.apply_chat_template(
                        row["messages"], tokenize=False, add_generation_prompt=False
                    )
                    enc = tokenizer(text, truncation=True, max_length=max_seq_length,
                                    padding=False, return_tensors=None)
                    input_ids_list.append(enc["input_ids"])
                    attention_mask_list.append(enc["attention_mask"])
                    labels_list.append(enc["input_ids"][:])  # labels = input_ids for causal LM
            return Dataset.from_dict({
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
                "labels": labels_list,
            })

        train_dataset = tokenize_jsonl(train_file)
        eval_dataset = tokenize_jsonl(valid_file) if has_valid else None
        log.info("Tokenized: %d train, %d valid", len(train_dataset), len(eval_dataset) if eval_dataset else 0)
        self._check_cancel(cancel_flag)

        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

        training_args = TrainingArguments(
            output_dir=str(adapter_output),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs if max_iters <= 0 else 1,
            max_steps=max_iters if max_iters > 0 else -1,
            warmup_ratio=warmup_ratio,
            save_steps=save_steps,
            eval_strategy="steps" if has_valid else "no",
            eval_steps=eval_steps if has_valid else None,
            logging_steps=10,
            logging_dir=str(run_dir / "logs"),
            save_total_limit=5,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            seed=42,
            report_to="none",
            remove_unused_columns=False,
        )

        log.info("Starting training...")
        trainer = Trainer(
            model=model, tokenizer=tokenizer,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            data_collator=data_collator,
            args=training_args,
        )
        train_result = trainer.train()
        self._check_cancel(cancel_flag)

        log.info("Saving adapter to %s", adapter_output)
        model.save_pretrained(str(adapter_output))
        tokenizer.save_pretrained(str(adapter_output))

        metrics = train_result.metrics
        summary = {
            "run_name": run_name, "model_name": model_name,
            "adapter_path": str(adapter_output),
            "train_loss": metrics.get("train_loss"),
            "train_runtime": metrics.get("train_runtime"),
            "epochs_completed": metrics.get("epoch"),
            "finished_at": time.time(),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        output_dir.mkdir(parents=True, exist_ok=True)
        final = output_dir / "adapter"
        if final.exists():
            shutil.rmtree(final)
        shutil.copytree(adapter_output, final)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        log.info("Training complete: loss=%.4f, runtime=%.1fs",
                 metrics.get("train_loss", 0), metrics.get("train_runtime", 0))
        del model, trainer
        self._cleanup_gpu()
        return {
            "format": "lora-adapter", "run_name": run_name,
            "adapter_path": str(adapter_output),
            "train_loss": metrics.get("train_loss"),
            "train_runtime_seconds": metrics.get("train_runtime"),
            "train_samples": train_count, "epochs": metrics.get("epoch"),
        }

    def estimate_time(self, params):
        max_iters = params.get("max_iters", 0)
        batch_size = params.get("batch_size", 4)
        if max_iters > 0:
            return max_iters * 500.0
        return 3600000.0
