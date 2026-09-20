"""Fine-tuning adapter for LLMs using Unsloth + HuggingFace Trainer with quality-first discipline."""

from __future__ import annotations

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

import importlib
import json
import logging
import shutil
import time
from pathlib import Path

from arbiter.adapters.base import ModelAdapter, InferenceError
from arbiter.adapters.registry import register

log = logging.getLogger(__name__)

TRAINING_ROOT = Path("/home/darren/training")

CHAT_TEMPLATE_MAP = {
    "llama-3": "llama-3.1",
    "llama-3.1": "llama-3.1",
    "llama-3.2": "llama-3.1",
    "qwen": "qwen-2.5",
    "gemma": "gemma",
    "phi": "phi-4",
    "mistral": "mistral",
}


def _detect_chat_template(model_name: str) -> str:
    lower = model_name.lower()
    for key, template in CHAT_TEMPLATE_MAP.items():
        if key in lower:
            return template
    return "qwen-2.5" if "qwen" in lower else "llama-3.1"


@register
class FineTuneAdapter(ModelAdapter):
    model_id = "fine-tune"

    def __init__(self):
        self._loaded = False
        self._device = "cuda"

    def load(self, device="cuda"):
        log.info("Pre-loading fine-tuning dependencies...")
        self._device = device
        self._loaded = True
        log.info("Fine-tuning dependencies ready.")

    def unload(self):
        self._loaded = False
        self._cleanup_gpu()

    def infer(self, params: dict, output_dir: Path, cancel_flag) -> dict:
        import torch

        FastLanguageModel = importlib.import_module("unsloth").FastLanguageModel
        get_chat_template = importlib.import_module(
            "unsloth.chat_templates"
        ).get_chat_template
        from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
        Dataset = importlib.import_module("datasets").Dataset

        self._check_cancel(cancel_flag)

        data_dir = Path(params["data_dir"])
        model_name = params["model_name"]
        run_name = params.get("run_name", f"ft-{int(time.time())}")
        lora_rank = int(params.get("lora_rank", 32))
        lora_alpha = int(params.get("lora_alpha", 64))
        lora_dropout = float(params.get("lora_dropout", 0.05))
        learning_rate = float(params.get("learning_rate", 2e-4))
        lr_scheduler = params.get("lr_scheduler_type", "cosine")
        warmup_ratio = float(params.get("warmup_ratio", 0.05))
        weight_decay = float(params.get("weight_decay", 0.01))
        batch_size = int(params.get("batch_size", 4))
        grad_accum_steps = int(params.get("grad_accum_steps", 4))
        num_epochs = int(params.get("num_epochs", 3))
        max_iters = int(params.get("max_iters", 0))
        max_seq_length = int(params.get("max_seq_length", 2048))
        save_steps = int(params.get("save_steps", 200))
        eval_steps = int(params.get("eval_steps", 200))
        load_in_4bit = bool(params.get("load_in_4bit", True))
        full_finetune = bool(params.get("full_finetune", False))
        mask_prompt = bool(params.get("mask_prompt", True))
        export_merged = bool(params.get("export_merged", True))
        export_format = params.get("export_format", "merged_16bit")
        export_dir_param = params.get("export_dir")
        chat_template = params.get("chat_template") or _detect_chat_template(model_name)

        train_file = data_dir / "train.jsonl"
        if not train_file.is_file():
            raise InferenceError(f"Training data not found: {train_file}")
        valid_file = data_dir / "valid.jsonl"
        has_valid = valid_file.is_file()

        train_count = sum(1 for line in open(train_file) if line.strip())
        valid_count = sum(1 for line in open(valid_file) if line.strip()) if has_valid else 0
        log.info("Fine-tuning data: %d train samples, %d validation samples", train_count, valid_count)

        run_dir = TRAINING_ROOT / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        adapter_output = run_dir / "adapter"
        adapter_output.mkdir(parents=True, exist_ok=True)

        config_record = {
            "model_name": model_name,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "lr_scheduler": lr_scheduler,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "num_epochs": num_epochs,
            "max_iters": max_iters,
            "train_samples": train_count,
            "valid_samples": valid_count,
            "mask_prompt": mask_prompt,
            "export_merged": export_merged,
            "export_format": export_format,
            "started_at": time.time(),
        }
        (run_dir / "config.json").write_text(json.dumps(config_record, indent=2))

        self._check_cancel(cancel_flag)

        os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        try:
            _u = importlib.import_module("unsloth.models._utils")
            setattr(_u, "has_internet", lambda *a, **kw: False)
        except Exception:
            pass

        log.info("Resolving base model: %s (4bit=%s)", model_name, load_in_4bit)
        if os.path.exists(model_name):
            model_path = model_name
        else:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model_name)
        log.info("Loading base model from: %s", model_path)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            device_map={"": 0},
        )
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._check_cancel(cancel_flag)

        if not full_finetune:
            log.info("Applying LoRA: rank=%d, alpha=%d", lora_rank, lora_alpha)
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                use_gradient_checkpointing="unsloth",
            )
        self._check_cancel(cancel_flag)

        log.info("Tokenizing dataset with mask_prompt=%s...", mask_prompt)

        def tokenize_jsonl(path: Path):
            input_ids_list, attention_mask_list, labels_list = [], [], []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    messages = row["messages"]
                    full_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    full_enc = tokenizer(
                        full_text,
                        truncation=True,
                        max_length=max_seq_length,
                        padding=False,
                        return_tensors=None,
                    )
                    input_ids = full_enc["input_ids"]
                    attention_mask = full_enc["attention_mask"]

                    if mask_prompt and len(messages) >= 2:
                        prompt_text = tokenizer.apply_chat_template(
                            messages[:-1], tokenize=False, add_generation_prompt=True
                        )
                        prompt_enc = tokenizer(
                            prompt_text,
                            truncation=True,
                            max_length=max_seq_length,
                            padding=False,
                            return_tensors=None,
                        )
                        prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))
                        labels = [-100] * prompt_len + input_ids[prompt_len:]
                    else:
                        labels = input_ids[:]

                    input_ids_list.append(input_ids)
                    attention_mask_list.append(attention_mask)
                    labels_list.append(labels)

            return Dataset.from_dict(
                {
                    "input_ids": input_ids_list,
                    "attention_mask": attention_mask_list,
                    "labels": labels_list,
                }
            )

        train_dataset = tokenize_jsonl(train_file)
        eval_dataset = tokenize_jsonl(valid_file) if has_valid else None
        log.info(
            "Tokenized dataset: %d train, %d eval",
            len(train_dataset),
            len(eval_dataset) if eval_dataset else 0,
        )
        self._check_cancel(cancel_flag)

        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

        training_args = TrainingArguments(
            output_dir=str(adapter_output),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            num_train_epochs=num_epochs if max_iters <= 0 else 1,
            max_steps=max_iters if max_iters > 0 else -1,
            save_steps=save_steps,
            eval_strategy="steps" if has_valid else "no",
            eval_steps=eval_steps if has_valid else None,
            logging_steps=10,
            logging_dir=str(run_dir / "logs"),
            save_total_limit=3,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            seed=42,
            report_to="none",
            remove_unused_columns=False,
        )

        log.info("Starting fine-tuning...")
        trainer = Trainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            args=training_args,
        )
        train_result = trainer.train()
        self._check_cancel(cancel_flag)

        log.info("Saving LoRA adapter to %s", adapter_output)
        model.save_pretrained(str(adapter_output))
        tokenizer.save_pretrained(str(adapter_output))

        metrics = train_result.metrics
        summary = {
            "run_name": run_name,
            "model_name": model_name,
            "adapter_path": str(adapter_output),
            "train_loss": metrics.get("train_loss"),
            "train_runtime_seconds": metrics.get("train_runtime"),
            "train_samples": train_count,
            "epochs": metrics.get("epoch"),
            "finished_at": time.time(),
        }

        merged_path = None
        if export_merged:
            export_target = (
                Path(export_dir_param)
                if export_dir_param
                else (run_dir / export_format)
            )
            export_target.mkdir(parents=True, exist_ok=True)
            log.info("Exporting merged model to %s (format=%s)...", export_target, export_format)
            if hasattr(model, "save_pretrained_merged"):
                model.save_pretrained_merged(
                    str(export_target),
                    tokenizer,
                    save_method=export_format,
                )
            else:
                unloaded = model.merge_and_unload()
                unloaded.save_pretrained(str(export_target))
                tokenizer.save_pretrained(str(export_target))
            merged_path = str(export_target)
            summary["merged_model_path"] = merged_path
            log.info("Merged model saved successfully at %s", merged_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        final_adapter = output_dir / "adapter"
        if final_adapter.exists():
            shutil.rmtree(final_adapter)
        shutil.copytree(adapter_output, final_adapter)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        log.info(
            "Fine-tuning complete: loss=%.4f, runtime=%.1fs, samples=%d",
            metrics.get("train_loss", 0),
            metrics.get("train_runtime", 0),
            train_count,
        )

        del model, trainer
        self._cleanup_gpu()

        result = {
            "format": "fine-tune",
            "run_name": run_name,
            "adapter_path": str(adapter_output),
            "train_loss": metrics.get("train_loss"),
            "train_runtime_seconds": metrics.get("train_runtime"),
            "train_samples": train_count,
            "epochs": metrics.get("epoch"),
        }
        if merged_path:
            result["merged_model_path"] = merged_path
        return result

    def estimate_time(self, params: dict) -> float:
        max_iters = params.get("max_iters", 0)
        if max_iters > 0:
            return max_iters * 500.0
        return 3600000.0


@register
class LoraTrainAdapter(FineTuneAdapter):
    model_id = "lora-train"
