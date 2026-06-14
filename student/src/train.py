import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback

from dataset import DerivedMultiTaskDataset, get_default_task_mix, set_seed
from detectors.holmes_clip_lora.runtime import DEFAULT_THRESHOLD, default_checkpoint_path
from evaluate import format_eta_seconds, render_markdown_report, run_evaluation
from utils.eval_utils import (
    ensure_split_manifest,
    load_epoch_eval_reports,
    render_checkpoint_ranking_markdown,
    row_ids_for_split,
    summarize_checkpoint_ranking,
)
from utils.model_utils import (
    load_image_text_model,
    load_processor,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


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
    def __init__(self, logger, progress_log_steps: int = 10):
        self.logger = logger
        self.pbar = None
        self.steps_per_epoch = 0
        self.current_loss = None
        self.current_lr = None
        self.current_grad_norm = None
        self.progress_log_steps = max(1, int(progress_log_steps))
        self.train_start_time = None
        self.session_start_step = None

    @staticmethod
    def _compute_grad_norm(model) -> Optional[float]:
        if model is None:
            return None
        sq_norm = 0.0
        found = False
        for param in model.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad = param.grad.detach()
            norm = float(torch.linalg.vector_norm(grad).item())
            sq_norm += norm * norm
            found = True
        if not found:
            return None
        return sq_norm ** 0.5

    def on_train_begin(self, args, state, control, **kwargs):
        epochs = max(1, int(args.num_train_epochs))
        self.steps_per_epoch = max(1, state.max_steps // epochs)
        self.train_start_time = time.time()
        self.session_start_step = None
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

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        grad_norm = self._compute_grad_norm(model)
        if grad_norm is not None:
            self.current_grad_norm = grad_norm

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            self.current_loss = logs["loss"]
        if "learning_rate" in logs:
            self.current_lr = logs["learning_rate"]
        if "grad_norm" in logs and logs["grad_norm"] not in (None, 0):
            self.current_grad_norm = logs["grad_norm"]

        if state.global_step <= 0:
            return

        should_log = state.global_step == state.max_steps or state.global_step % self.progress_log_steps == 0
        if not should_log:
            return

        current_epoch = float(logs.get("epoch", state.epoch or 0.0))
        epoch_index = max(1, int(current_epoch) if float(current_epoch).is_integer() else int(current_epoch) + 1)
        epoch_step = ((state.global_step - 1) % self.steps_per_epoch) + 1
        progress = state.global_step / state.max_steps if state.max_steps else 0.0

        eta_text = "unknown"
        if self.train_start_time and state.global_step > 0:
            if self.session_start_step is None:
                self.session_start_step = max(0, state.global_step - 1)
            session_steps = max(0, state.global_step - self.session_start_step)
            if session_steps >= 5:
                elapsed = max(0.0, time.time() - self.train_start_time)
                sec_per_step = elapsed / session_steps
                remaining = max(0.0, (state.max_steps - state.global_step) * sec_per_step)
                eta_hours = int(remaining // 3600)
                eta_minutes = int((remaining % 3600) // 60)
                eta_text = f"{eta_hours:02d}:{eta_minutes:02d}"
            else:
                eta_text = "warming_up"

        parts = [
            f"epoch={epoch_index}/{int(args.num_train_epochs)}",
            f"epoch_step={epoch_step}/{self.steps_per_epoch}",
            f"global_step={state.global_step}/{state.max_steps}",
            f"progress={progress * 100:.2f}%",
        ]
        if isinstance(self.current_loss, (int, float)):
            parts.append(f"loss={self.current_loss:.4g}")
        if isinstance(self.current_lr, (int, float)):
            parts.append(f"lr={self.current_lr:.4g}")
        if isinstance(self.current_grad_norm, (int, float)):
            parts.append(f"grad_norm={self.current_grad_norm:.4g}")
        parts.append(f"eta={eta_text}")
        self.logger.info("[progress] " + " | ".join(parts))

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class DatasetEpochCallback(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.set_epoch(int(state.epoch or 0))


class EpochEvalCallback(TrainerCallback):
    def __init__(
        self,
        logger,
        model_name: str,
        processor,
        dataset_path: str,
        prompt_dir: str,
        run_dir: Path,
        split_manifest_path: Path,
        detector_checkpoint_path: str,
        detector_clip_weights: str,
        detector_threshold: float,
        local_files_only: bool,
    ):
        self.logger = logger
        self.model_name = model_name
        self.processor = processor
        self.dataset_path = dataset_path
        self.prompt_dir = Path(prompt_dir)
        self.run_dir = Path(run_dir)
        self.split_manifest_path = Path(split_manifest_path)
        self.detector_checkpoint_path = detector_checkpoint_path
        self.detector_clip_weights = detector_clip_weights
        self.detector_threshold = float(detector_threshold)
        self.local_files_only = bool(local_files_only)
        self.last_eval_step = None
        self.evidence_trace_prompt = (self.prompt_dir / "evidence_trace.txt").read_text(encoding="utf-8").strip()
        self.final_json_prompt = (self.prompt_dir / "stage2.txt").read_text(encoding="utf-8").strip()
        self.output_dir = self.run_dir / "training_eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_checkpoint_dir = self.run_dir / "epoch_checkpoints"
        self.epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _model_device(model) -> str:
        return str(next(model.parameters()).device)

    def _log_eval_progress(self, epoch_label: str, payload: dict) -> None:
        self.logger.info(
            "[epoch_eval_progress] epoch=%s | done=%s/%s | acc=%.3f | json_parse=%.3f | trace_parse=%.3f | "
            "sec_per_sample=%.2f | eta=%s",
            epoch_label,
            payload["done"],
            payload["total"],
            float(payload["overall_accuracy"]),
            float(payload["json_parse_rate"]),
            float(payload["trace_json_parse_rate"]),
            float(payload["sec_per_sample"] or 0.0),
            format_eta_seconds(payload["eta_sec"]),
        )

    @staticmethod
    def _epoch_index(epoch_value: Optional[float]) -> int:
        numeric = float(epoch_value or 0.0)
        if numeric <= 0:
            return 1
        rounded = int(numeric)
        return rounded if numeric.is_integer() else max(1, rounded + 1)

    def _snapshot_epoch_adapter(self, model, epoch_label: str, epoch_index: int, global_step: int) -> Path:
        adapter_dir = self.epoch_checkpoint_dir / epoch_label
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        metadata = {
            "epoch_label": epoch_label,
            "epoch_index": int(epoch_index),
            "global_step": int(global_step),
            "base_model": self.model_name,
        }
        (adapter_dir / "epoch_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return adapter_dir

    def _run_eval(self, model, epoch_index: int, global_step: int) -> None:
        epoch_label = f"epoch_{epoch_index:03d}"
        if self.last_eval_step == int(global_step):
            return
        model_was_training = model.training
        model.eval()
        previous_cache = getattr(model.config, "use_cache", None)
        if previous_cache is not None:
            model.config.use_cache = True
        adapter_path = self._snapshot_epoch_adapter(model, epoch_label, epoch_index, global_step)
        try:
            detector_device = self._model_device(model)
            self.logger.info(
                "[epoch_eval] start | epoch=%s | step=%s | detector_device=%s | slice=64 (32 real + 32 fake)",
                epoch_label,
                global_step,
                detector_device,
            )
            eval_args = argparse.Namespace(
                base_model=self.model_name,
                adapter_path=str(adapter_path),
                derived_data_path=self.dataset_path,
                prompt_dir=str(self.prompt_dir),
                max_samples=0,
                eval_slice_count=64,
                eval_slice_seed=42,
                balanced_max_samples=0,
                probe_samples=0,
                max_new_tokens_trace=1536,
                max_new_tokens_json=1024,
                inference_mode="two_stage",
                split="eval",
                split_manifest_path=str(self.split_manifest_path),
                prediction_source="detector_student",
                teacher_reference="derived_target",
                teacher_jsonl_path=str(REPO_ROOT / "teacher" / "stage1_g31b_v5_full_balanced" / "holmes_lpcvc_sft.jsonl"),
                detector_checkpoint_path=self.detector_checkpoint_path,
                detector_clip_weights=self.detector_clip_weights,
                detector_threshold=self.detector_threshold,
                detector_device=detector_device,
                enable_cider=False,
                progress_log_every=8,
                row_ids_path=None,
                predictions_path=str(self.output_dir / f"{epoch_label}.predictions.jsonl"),
                local_files_only=self.local_files_only,
                output_path=None,
            )
            report = run_evaluation(
                eval_args,
                student_bundle={
                    "processor": self.processor,
                    "model": model,
                    "evidence_trace_prompt": self.evidence_trace_prompt,
                    "final_json_prompt": self.final_json_prompt,
                },
                progress_callback=lambda payload: self._log_eval_progress(epoch_label, payload),
            )
        finally:
            if previous_cache is not None:
                model.config.use_cache = previous_cache
            if model_was_training:
                model.train()

        report["epoch_index"] = int(epoch_index)
        report["epoch_label"] = epoch_label
        report["global_step"] = int(global_step)
        report["adapter_path"] = str(adapter_path)
        json_path = self.output_dir / f"{epoch_label}.json"
        md_path = self.output_dir / f"{epoch_label}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        render_markdown_report(epoch_label, report, md_path)
        self.last_eval_step = int(global_step)
        self.logger.info(
            "[epoch_eval] epoch=%s | step=%s | overall_accuracy=%.3f | overall_macro_f1=%.3f | criterion_macro_f1=%.3f | rouge_l=%.3f | report=%s",
            epoch_label,
            global_step,
            float(report.get("overall_accuracy") or 0.0),
            float(report.get("overall_macro_f1") or 0.0),
            float(report.get("criterion_macro_f1") or 0.0),
            float(report.get("rouge_l") or 0.0),
            json_path,
        )

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step <= 0:
            return
        self._run_eval(model, self._epoch_index(state.epoch), int(state.global_step))


_VISION_KEYS = {
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
}
_PAD_KEYS = {"input_ids", "attention_mask", "labels"}
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
        if torch.is_tensor(tensors[0]):
            batch[key] = torch.stack(tensors, dim=0)
        else:
            batch[key] = tensors
    batch["task_name"] = [feature["task_name"] for feature in features]
    return batch


def resolve_lora_target_modules(model) -> List[str]:
    base_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    module_names = {name for name, _ in model.named_modules()}
    resolved = []
    for name in base_targets:
        linear_name = f"{name}.linear"
        if any(module_name.endswith(linear_name) for module_name in module_names):
            resolved.append(linear_name)
        else:
            resolved.append(name)
    return resolved


def ensure_input_gradients(model) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    input_embeddings = None
    if hasattr(model, "get_input_embeddings"):
        try:
            input_embeddings = model.get_input_embeddings()
        except Exception:
            input_embeddings = None

    if input_embeddings is None:
        return

    def _make_inputs_require_grad(module, inputs, output):
        if torch.is_tensor(output):
            output.requires_grad_(True)
        elif isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    item.requires_grad_(True)

    hook_registered = getattr(input_embeddings, "_codex_input_grad_hook", False)
    if not hook_registered:
        input_embeddings.register_forward_hook(_make_inputs_require_grad)
        setattr(input_embeddings, "_codex_input_grad_hook", True)


def build_model(model_name_or_path, local_files_only):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = load_image_text_model(
        model_name_or_path,
        quantization_config=bnb_config,
        local_files_only=local_files_only,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    ensure_input_gradients(model)

    lora_target_modules = resolve_lora_target_modules(model)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    ensure_input_gradients(model)
    return model, lora_target_modules


class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("task_name", None)
        inputs.pop("row_id", None)
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def build_training_args(args, run_dir):
    training_kwargs = dict(
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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        seed=args.seed,
        report_to="tensorboard",
        disable_tqdm=True,
    )
    if args.max_steps is not None and args.max_steps > 0:
        training_kwargs["max_steps"] = args.max_steps
    return TrainingArguments(**training_kwargs)


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
    parser = argparse.ArgumentParser(description="Fine-tune a multitask image-authenticity vision-language model.")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(REPO_ROOT / "teacher" / "stage1_g31b_v5_full_balanced"),
    )
    parser.add_argument("--derived_data_path", type=str, default=None)
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=str(REPO_ROOT / "prompts"),
    )
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "student" / "outputs"))
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--train_mode", type=str, default="multitask_sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--progress_log_steps", type=int, default=10)
    parser.add_argument("--task_mix", type=str, default=None)
    parser.add_argument("--trace_evidence_words", type=int, default=14)
    parser.add_argument("--trace_holmes_span_words", type=int, default=12)
    parser.add_argument("--eval_ratio", type=float, default=0.10)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_manifest_path", type=str, default=None)
    parser.add_argument("--regenerate_split", action="store_true")
    parser.add_argument("--disable_epoch_eval", action="store_true")
    parser.add_argument("--detector_checkpoint_path", type=str, default=default_checkpoint_path())
    parser.add_argument("--detector_clip_weights", type=str, default=None)
    parser.add_argument("--detector_threshold", type=float, default=DEFAULT_THRESHOLD)
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
        f"epochs={args.epochs} | max_steps={args.max_steps} | lr={args.lr} | local_only={args.local_files_only}"
    )
    if not args.disable_epoch_eval and (
        not args.detector_clip_weights or not Path(args.detector_clip_weights).exists()
    ):
        raise FileNotFoundError(
            "epoch-level detector_student evaluation requires --detector_clip_weights <ViT-L-14-336px.pt>."
        )

    processor = load_processor(args.model_name_or_path, local_files_only=args.local_files_only)
    dataset_path = resolve_dataset_path(args)
    split_manifest_path, split_manifest = ensure_split_manifest(
        Path(dataset_path),
        eval_ratio=args.eval_ratio,
        seed=args.split_seed,
        manifest_path=Path(args.split_manifest_path) if args.split_manifest_path else None,
        regenerate=args.regenerate_split,
    )
    train_row_ids = row_ids_for_split(split_manifest, "train") or set()
    eval_row_ids = row_ids_for_split(split_manifest, "eval") or set()
    task_mix = parse_task_mix(args.task_mix)
    train_dataset = DerivedMultiTaskDataset(
        dataset_path,
        args.prompt_dir,
        processor=processor,
        task_mix=task_mix,
        trace_evidence_words=args.trace_evidence_words,
        trace_holmes_span_words=args.trace_holmes_span_words,
        seed=args.seed,
        allowed_row_ids=train_row_ids,
    )
    logger.info(f"Dataset loaded: {len(train_dataset)} train rows from {dataset_path}")
    logger.info(
        "Split manifest: %s | train_rows=%s | eval_rows=%s | eval_ratio=%.3f | split_seed=%s",
        split_manifest_path,
        len(train_row_ids),
        len(eval_row_ids),
        args.eval_ratio,
        args.split_seed,
    )
    logger.info(f"Task mix: {task_mix}")

    if len(train_dataset) > 0:
        sample = train_dataset[0]
        shape_info = {
            key: (tuple(value.shape) if torch.is_tensor(value) else type(value).__name__)
            for key, value in sample.items()
            if key != "task_name"
        }
        logger.info(f"Sample keys/shapes: {shape_info} | task={sample['task_name']}")

    model, lora_target_modules = build_model(args.model_name_or_path, args.local_files_only)
    logger.info(f"LoRA target modules: {lora_target_modules}")
    model.print_trainable_parameters()

    training_args = build_training_args(args, run_dir)
    callbacks = [
        EpochProgressCallback(logger, progress_log_steps=args.progress_log_steps),
        DatasetEpochCallback(train_dataset),
    ]
    if not args.disable_epoch_eval:
        callbacks.append(
            EpochEvalCallback(
                logger,
                model_name=args.model_name_or_path,
                processor=processor,
                dataset_path=dataset_path,
                prompt_dir=args.prompt_dir,
                run_dir=Path(run_dir),
                split_manifest_path=split_manifest_path,
                detector_checkpoint_path=args.detector_checkpoint_path,
                detector_clip_weights=args.detector_clip_weights,
                detector_threshold=args.detector_threshold,
                local_files_only=args.local_files_only,
            )
        )
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_collate_fn,
        callbacks=callbacks,
    )
    trainer.remove_callback(PrinterCallback)

    resume_ckpt = resolve_resume_checkpoint(args, Path(run_dir))
    if resume_ckpt:
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(run_dir)
    processor.save_pretrained(run_dir)
    logger.info("Training completed and model saved.")

    ranking_json_path = Path(run_dir) / "training_eval" / "checkpoint_ranking.json"
    ranking_summary = summarize_checkpoint_ranking(load_epoch_eval_reports(Path(run_dir) / "training_eval"))
    if not args.disable_epoch_eval:
        ranking_md_path = ranking_json_path.with_suffix(".md")
        ranking_json_path.write_text(json.dumps(ranking_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        render_checkpoint_ranking_markdown(ranking_summary, ranking_md_path)
        logger.info("Checkpoint ranking written: %s", ranking_json_path)

    summary = {
        "timestamp": Path(run_dir).name,
        "model": args.model_name_or_path,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "train_mode": args.train_mode,
        "task_mix": task_mix,
        "trace_evidence_words": args.trace_evidence_words,
        "trace_holmes_span_words": args.trace_holmes_span_words,
        "eval_ratio": args.eval_ratio,
        "split_seed": args.split_seed,
        "split_manifest_path": str(split_manifest_path),
        "train_rows": len(train_row_ids),
        "eval_rows": len(eval_row_ids),
        "epoch_eval_enabled": not args.disable_epoch_eval,
        "epoch_eval_prediction_source": "detector_student" if not args.disable_epoch_eval else None,
        "epoch_eval_split": "eval",
        "detector_checkpoint_path": args.detector_checkpoint_path,
        "detector_clip_weights": args.detector_clip_weights,
        "detector_threshold": args.detector_threshold,
        "best_epoch_checkpoint": ranking_summary.get("best_checkpoint_path"),
        "checkpoint_ranking_path": str(ranking_json_path) if not args.disable_epoch_eval else None,
        "status": "Training Completed",
        "log_dir": str(run_dir),
    }
    with open(os.path.join(run_dir, "experiments_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Experiment summary written.")


if __name__ == "__main__":
    main()
