import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback

from dataset import DerivedMultiTaskDataset, get_default_task_mix, set_seed


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("TRAIN")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_task_mix(raw_value: Optional[str]) -> Dict[str, float]:
    task_mix = get_default_task_mix()
    if not raw_value:
        return task_mix
    try:
        parsed = json.loads(raw_value)
        return {key: float(value) for key, value in parsed.items()}
    except json.JSONDecodeError:
        result = {}
        for chunk in raw_value.split(","):
            if not chunk.strip():
                continue
            key, value = chunk.split("=")
            result[key.strip()] = float(value.strip())
        return result


def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    checkpoints = []
    for candidate in run_dir.glob("checkpoint-*"):
        try:
            step = int(candidate.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, candidate))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def resolve_run_dir(output_dir: str, run_name: Optional[str], resume_from_checkpoint: Optional[str]) -> Path:
    output_root = Path(output_dir)
    if resume_from_checkpoint and resume_from_checkpoint.lower() != "true":
        checkpoint_path = Path(resume_from_checkpoint)
        return checkpoint_path.parent if checkpoint_path.name.startswith("checkpoint-") else checkpoint_path
    if run_name:
        return output_root / run_name
    return output_root / datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_resume_checkpoint(args, run_dir: Path) -> Optional[str]:
    if not args.resume_from_checkpoint:
        return None
    raw_value = args.resume_from_checkpoint.strip()
    if raw_value.lower() == "true":
        latest = find_latest_checkpoint(run_dir)
        if latest is None:
            raise FileNotFoundError(
                "resume_from_checkpoint=True requires an existing run directory with checkpoint-* contents. "
                "Pass --run_name <existing_run> or provide an explicit checkpoint path."
            )
        return str(latest)
    checkpoint_path = Path(raw_value)
    if checkpoint_path.is_dir() and checkpoint_path.name.startswith("checkpoint-"):
        return str(checkpoint_path)
    raise FileNotFoundError(
        "resume_from_checkpoint must be either True or an explicit checkpoint-* directory path."
    )


class EpochProgressCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger
        self.pbar = None
        self.steps_per_epoch = 0
        self.current_loss = None

    def on_train_begin(self, args, state, control, **kwargs):
        epochs = max(1, int(args.num_train_epochs))
        self.steps_per_epoch = max(1, state.max_steps // epochs)
        self.logger.info(
            f"Training: {epochs} epochs, ~{self.steps_per_epoch} steps/epoch (total {state.max_steps})"
        )

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(getattr(state, "epoch", 0) + 1)
        self.pbar = tqdm(
            total=self.steps_per_epoch,
            desc=f"Epoch {epoch}/{int(args.num_train_epochs)}",
            leave=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.update(1)
            if self.current_loss is not None:
                self.pbar.set_postfix(loss=f"{self.current_loss:.4g}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            self.current_loss = logs["loss"]
        if "epoch" in logs and abs(logs["epoch"] - round(logs["epoch"])) < 1e-4:
            parts = [f"step={state.global_step}"]
            for key in ("loss", "learning_rate", "grad_norm", "epoch"):
                value = logs.get(key)
                if isinstance(value, (int, float)):
                    parts.append(f"{key}={value:.4g}")
            self.logger.info("[train] " + " | ".join(parts))

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class DatasetEpochCallback(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.set_epoch(int(state.epoch or 0))


_VISION_KEYS = {
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
}
_PAD_KEYS = {"input_ids", "attention_mask", "labels"}
_FLOAT_KEYS = {"expert_overall_prob", "expert_criterion_probs", "expert_target_mask"}


def custom_collate_fn(features):
    batch = {}
    keys = [key for key in features[0].keys() if key != "task_name"]
    for key in keys:
        tensors = [feature[key] for feature in features if feature.get(key) is not None]
        if not tensors:
            continue
        if key in _VISION_KEYS:
            batch[key] = torch.cat(tensors, dim=0)
            continue
        if key in _PAD_KEYS:
            pad_value = -100 if key == "labels" else 0
            batch[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=pad_value
            )
            continue
        if (
            torch.is_tensor(tensors[0])
            and tensors[0].ndim == 1
            and any(tensor.shape[0] != tensors[0].shape[0] for tensor in tensors[1:])
        ):
            batch[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=0
            )
            continue
        if key in _FLOAT_KEYS:
            batch[key] = torch.stack(tensors, dim=0)
            continue
        if torch.is_tensor(tensors[0]):
            batch[key] = torch.stack(tensors, dim=0)
        else:
            batch[key] = tensors
    batch["task_name"] = [feature["task_name"] for feature in features]
    return batch


def _resolve_model_class(name):
    if "Qwen2.5-VL" in name:
        return Qwen2_5_VLForConditionalGeneration
    if "Qwen2-VL" in name:
        return Qwen2VLForConditionalGeneration
    return AutoModelForCausalLM


def build_model(model_name_or_path, local_files_only, enable_distillation=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = _resolve_model_class(model_name_or_path).from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        use_safetensors=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
        attn_implementation={"": "sdpa", "vision_config": "sdpa"},
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if enable_distillation:
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            text_config = getattr(model.config, "text_config", None)
            if text_config is not None:
                hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Unable to resolve hidden_size for distillation head")
        model.distill_head = torch.nn.Linear(hidden_size, 9)
        model.distill_head.to(next(model.parameters()).device)
    return model


class MultiTaskTrainer(Trainer):
    def __init__(self, *args, distill_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_weight = distill_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        task_names = inputs.pop("task_name", None)
        row_ids = inputs.pop("row_id", None)
        expert_overall = inputs.pop("expert_overall_prob", None)
        expert_criteria = inputs.pop("expert_criterion_probs", None)
        expert_mask = inputs.pop("expert_target_mask", None)

        if self.distill_weight > 0 and hasattr(model, "distill_head"):
            outputs = model(**inputs, output_hidden_states=True)
        else:
            outputs = model(**inputs)

        loss = outputs.loss
        metrics = {}
        if (
            self.distill_weight > 0
            and hasattr(model, "distill_head")
            and expert_mask is not None
            and torch.any(expert_mask > 0)
        ):
            hidden_states = outputs.hidden_states[-1]
            attention_mask = inputs["attention_mask"].float()
            pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            model.distill_head.to(pooled.device)
            distill_logits = model.distill_head(pooled)
            distill_targets = torch.cat([expert_overall, expert_criteria], dim=1).to(distill_logits.device)
            distill_mask = expert_mask.to(distill_logits.device).view(-1, 1)
            distill_loss = F.binary_cross_entropy_with_logits(
                distill_logits,
                distill_targets,
                reduction="none",
            )
            distill_loss = (distill_loss * distill_mask).sum() / distill_mask.sum().clamp_min(1.0)
            loss = loss + self.distill_weight * distill_loss
            metrics["distill_loss"] = float(distill_loss.detach().cpu())

        if metrics:
            self.log(metrics)
        return (loss, outputs) if return_outputs else loss


def build_training_args(args, run_dir):
    return TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=args.batch_size,
        remove_unused_columns=False,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        logging_dir=os.path.join(run_dir, "tensorboard_logs"),
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, args.save_steps),
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        optim="paged_adamw_8bit",
        seed=args.seed,
        report_to="tensorboard",
        disable_tqdm=True,
    )


def resolve_dataset_path(args):
    if args.derived_data_path:
        path = args.derived_data_path
    else:
        path = args.data_path
    if os.path.isdir(path):
        derived = os.path.join(path, "derived.jsonl")
        legacy = os.path.join(path, "holmes_lpcvc_sft.jsonl")
        path = derived if os.path.exists(derived) else legacy
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Student VLM Fine-Tuning")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/ssd4/LPCVC2026/holmes_lpcvc3_multi_teacher/stage1_g31b_v5_full_balanced",
    )
    parser.add_argument("--derived_data_path", type=str, default=None)
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="/ssd4/LPCVC2026/Module-II-Final/prompts",
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--train_mode", type=str, default="multitask_sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--task_mix", type=str, default=None)
    parser.add_argument("--visual_expert_path", type=str, default=None)
    parser.add_argument("--distill_weight", type=float, default=0.0)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    run_dir = resolve_run_dir(args.output_dir, args.run_name, args.resume_from_checkpoint)
    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logger(run_dir)
    logger.info(f"SFT start | model={args.model_name_or_path} | run_dir={run_dir}")
    logger.info(
        f"Config | derived_data={args.derived_data_path or args.data_path} | bs={args.batch_size} | "
        f"epochs={args.epochs} | lr={args.lr} | local_only={args.local_files_only}"
    )

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only
    )
    dataset_path = resolve_dataset_path(args)
    task_mix = parse_task_mix(args.task_mix)
    train_dataset = DerivedMultiTaskDataset(
        dataset_path,
        args.prompt_dir,
        processor=processor,
        task_mix=task_mix,
        expert_targets_path=args.visual_expert_path,
        seed=args.seed,
    )
    logger.info(f"Dataset loaded: {len(train_dataset)} rows from {dataset_path}")
    logger.info(f"Task mix: {task_mix}")

    if len(train_dataset) > 0:
        sample = train_dataset[0]
        shape_info = {
            key: (tuple(value.shape) if torch.is_tensor(value) else type(value).__name__)
            for key, value in sample.items()
            if key != "task_name"
        }
        logger.info(f"Sample keys/shapes: {shape_info} | task={sample['task_name']}")

    enable_distillation = bool(args.visual_expert_path and args.distill_weight > 0)
    model = build_model(
        args.model_name_or_path,
        args.local_files_only,
        enable_distillation=enable_distillation,
    )
    model.print_trainable_parameters()

    training_args = build_training_args(args, run_dir)
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_collate_fn,
        callbacks=[
            EpochProgressCallback(logger),
            DatasetEpochCallback(train_dataset),
        ],
        distill_weight=args.distill_weight,
    )
    trainer.remove_callback(PrinterCallback)

    resume_ckpt = resolve_resume_checkpoint(args, Path(run_dir))
    if resume_ckpt:
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(run_dir)
    processor.save_pretrained(run_dir)
    if enable_distillation and hasattr(model, "distill_head"):
        torch.save(model.distill_head.state_dict(), os.path.join(run_dir, "distill_head.pt"))
    logger.info("Training completed and model saved.")

    summary = {
        "timestamp": Path(run_dir).name,
        "model": args.model_name_or_path,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_mode": args.train_mode,
        "task_mix": task_mix,
        "visual_expert_path": args.visual_expert_path,
        "distill_weight": args.distill_weight,
        "status": "Training Completed",
        "log_dir": str(run_dir),
    }
    with open(os.path.join(run_dir, "experiments_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Experiment summary written.")


if __name__ == "__main__":
    main()
