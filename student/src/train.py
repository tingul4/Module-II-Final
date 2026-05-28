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
import torch.nn.functional as F
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
from inference import apply_expert_fusion, generate_trace_payload
from model_utils import load_image_text_model, load_processor, prepare_generation_inputs
from task_utils import safe_json_loads


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
        self.last_logged_step = 0
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
        self.last_logged_step = 0
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

        should_log = (
            state.global_step == 1
            or state.global_step == state.max_steps
            or state.global_step - self.last_logged_step >= self.progress_log_steps
        )
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
        self.last_logged_step = state.global_step

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class DatasetEpochCallback(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.set_epoch(int(state.epoch or 0))


def load_eval_rows(dataset_path: str, max_rows: int) -> List[dict]:
    rows = []
    reals = []
    fakes = []
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            target = row.get("final_json_target", {})
            label = target.get("overall_likelihood", "")
            if label == "Real":
                reals.append(row)
            elif label == "AI-Generated":
                fakes.append(row)
            if len(reals) >= max_rows and len(fakes) >= max_rows:
                break
    combined = []
    limit_each = max(1, max_rows // 2)
    combined.extend(reals[:limit_each])
    combined.extend(fakes[:limit_each])
    if len(combined) < max_rows:
        seen = {int(item.get("row_id", -1)) for item in combined}
        with open(dataset_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if len(combined) >= max_rows:
                    break
                row = json.loads(line)
                row_id = int(row.get("row_id", -1))
                if row_id in seen:
                    continue
                combined.append(row)
    return combined[:max_rows]


def resolve_eval_image_path(dataset_path: str, row: dict) -> str:
    image_path = Path(str(row["image"]))
    if image_path.is_absolute() and image_path.exists():
        return str(image_path)
    image_root = row.get("image_root")
    if image_root:
        candidate = Path(str(image_root)) / str(row["image"])
        if candidate.exists():
            return str(candidate)
    return str(Path(dataset_path).parent / str(row["image"]))


def render_training_eval_html(step: int, samples: List[dict], output_path: Path) -> None:
    rows = []
    for sample in samples:
        rows.append(
            "<tr>"
            f"<td>{sample['row_id']}</td>"
            f"<td>{sample['gold_overall']}</td>"
            f"<td>{sample['pred_overall']}</td>"
            f"<td>{'yes' if sample['final_json_parse_ok'] else 'no'}</td>"
            f"<td>{sample['final_error']}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Training Eval Step {step}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
    pre {{ white-space: pre-wrap; margin: 0; }}
  </style>
</head>
<body>
  <h1>Training Eval Step {step}</h1>
  <table>
    <thead>
      <tr><th>Row</th><th>GT</th><th>Pred</th><th>Parse OK</th><th>Parse Error</th></tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


class FixedStepEvalCallback(TrainerCallback):
    def __init__(
        self,
        logger,
        processor,
        dataset_path: str,
        prompt_dir: str,
        run_dir: Path,
        eval_steps: int = 500,
        max_rows: int = 4,
        max_new_tokens_trace: int = 1024,
        max_new_tokens_json: int = 768,
        expert_path: Optional[str] = None,
        fusion_alpha: float = 0.8,
    ):
        self.logger = logger
        self.processor = processor
        self.dataset_path = dataset_path
        self.prompt_dir = Path(prompt_dir)
        self.run_dir = Path(run_dir)
        self.eval_steps = max(1, int(eval_steps))
        self.max_rows = max(1, int(max_rows))
        self.max_new_tokens_trace = int(max_new_tokens_trace)
        self.max_new_tokens_json = int(max_new_tokens_json)
        self.expert_path = expert_path
        self.fusion_alpha = float(fusion_alpha)
        self.rows = load_eval_rows(dataset_path, self.max_rows)
        self.last_eval_step = 0
        self.trace_prompt = (self.prompt_dir / "evidence_trace.txt").read_text(encoding="utf-8").strip()
        self.final_prompt = (self.prompt_dir / "stage2.txt").read_text(encoding="utf-8").strip()
        self.output_dir = self.run_dir / "training_eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate(self, model, image_path: str, prompt_text: str, max_new_tokens: int) -> str:
        inputs = prepare_generation_inputs(self.processor, image_path, prompt_text)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        return output_text[0]

    def _run_eval(self, model, step: int) -> None:
        model_was_training = model.training
        model.eval()
        previous_cache = getattr(model.config, "use_cache", None)
        if previous_cache is not None:
            model.config.use_cache = True

        samples = []
        parse_ok = 0
        trace_parse_ok = 0
        overall_ok = 0
        try:
            for row in self.rows:
                image_path = resolve_eval_image_path(self.dataset_path, row)
                trace_text, trace_json, trace_error, trace_for_final, trace_retry_used = generate_trace_payload(
                    model,
                    self.processor,
                    image_path,
                    self.trace_prompt,
                    self.max_new_tokens_trace,
                )
                trace_parse_ok += int(bool(trace_json))
                final_prompt = (
                    f"{self.final_prompt}\n\n"
                    "Here is the structured evidence trace for this image:\n"
                    f"{trace_for_final}\n\n"
                    "Use the trace to synthesize the final structured decision JSON."
                )
                final_text = self._generate(model, image_path, final_prompt, self.max_new_tokens_json)
                final_json, final_error = safe_json_loads(final_text)
                if final_json and self.expert_path:
                    final_json = apply_expert_fusion(final_json, self.expert_path, image_path, self.fusion_alpha)
                parse_ok += int(bool(final_json))
                gold = row["final_json_target"]
                pred_overall = final_json.get("overall_likelihood") if final_json else ""
                overall_ok += int(pred_overall == gold.get("overall_likelihood"))
                samples.append(
                    {
                        "row_id": int(row.get("row_id", -1)),
                        "image": row["image"],
                        "gold_overall": gold.get("overall_likelihood", ""),
                        "pred_overall": pred_overall,
                        "trace_parse_ok": bool(trace_json),
                        "trace_retry_used": trace_retry_used,
                        "trace_error": trace_error,
                        "final_json_parse_ok": bool(final_json),
                        "final_error": final_error,
                        "pred_trace_text": trace_text,
                        "pred_final_text": final_text,
                        "gold_final_json": gold,
                    }
                )
        finally:
            if previous_cache is not None:
                model.config.use_cache = previous_cache
            if model_was_training:
                model.train()

        summary = {
            "step": step,
            "samples": len(samples),
            "trace_json_parse_rate": trace_parse_ok / len(samples) if samples else 0.0,
            "final_json_parse_rate": parse_ok / len(samples) if samples else 0.0,
            "overall_accuracy": overall_ok / len(samples) if samples else 0.0,
            "rows": samples,
        }
        json_path = self.output_dir / f"step_{step:06d}.json"
        html_path = self.output_dir / f"step_{step:06d}.html"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        render_training_eval_html(step, samples, html_path)
        self.logger.info(
            "[sample_eval] step=%s | samples=%s | trace_json_parse_rate=%.3f | final_json_parse_rate=%.3f | overall_accuracy=%.3f | report=%s",
            step,
            len(samples),
            summary["trace_json_parse_rate"],
            summary["final_json_parse_rate"],
            summary["overall_accuracy"],
            json_path,
        )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step <= 0:
            return
        if state.global_step - self.last_eval_step < self.eval_steps and state.global_step != state.max_steps:
            return
        self.last_eval_step = state.global_step
        self._run_eval(model, state.global_step)


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


def build_model(model_name_or_path, local_files_only, enable_distillation=False):
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
    return model, lora_target_modules


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
    parser.add_argument("--sample_eval_steps", type=int, default=500)
    parser.add_argument("--sample_eval_rows", type=int, default=4)
    parser.add_argument("--sample_eval_max_new_tokens_trace", type=int, default=1024)
    parser.add_argument("--sample_eval_max_new_tokens_json", type=int, default=768)
    parser.add_argument("--task_mix", type=str, default=None)
    parser.add_argument("--visual_expert_path", type=str, default=None)
    parser.add_argument("--distill_weight", type=float, default=0.0)
    parser.add_argument("--trace_evidence_words", type=int, default=14)
    parser.add_argument("--trace_holmes_span_words", type=int, default=12)
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

    processor = load_processor(args.model_name_or_path, local_files_only=args.local_files_only)
    dataset_path = resolve_dataset_path(args)
    task_mix = parse_task_mix(args.task_mix)
    train_dataset = DerivedMultiTaskDataset(
        dataset_path,
        args.prompt_dir,
        processor=processor,
        task_mix=task_mix,
        expert_targets_path=args.visual_expert_path,
        trace_evidence_words=args.trace_evidence_words,
        trace_holmes_span_words=args.trace_holmes_span_words,
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
    model, lora_target_modules = build_model(
        args.model_name_or_path,
        args.local_files_only,
        enable_distillation=enable_distillation,
    )
    logger.info(f"LoRA target modules: {lora_target_modules}")
    model.print_trainable_parameters()

    training_args = build_training_args(args, run_dir)
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_collate_fn,
        callbacks=[
            EpochProgressCallback(logger, progress_log_steps=args.progress_log_steps),
            DatasetEpochCallback(train_dataset),
            FixedStepEvalCallback(
                logger,
                processor,
                dataset_path=dataset_path,
                prompt_dir=args.prompt_dir,
                run_dir=Path(run_dir),
                eval_steps=args.sample_eval_steps,
                max_rows=args.sample_eval_rows,
                max_new_tokens_trace=args.sample_eval_max_new_tokens_trace,
                max_new_tokens_json=args.sample_eval_max_new_tokens_json,
                expert_path=args.visual_expert_path if args.distill_weight > 0 else None,
            ),
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
        "max_steps": args.max_steps,
        "lr": args.lr,
        "train_mode": args.train_mode,
        "task_mix": task_mix,
        "visual_expert_path": args.visual_expert_path,
        "distill_weight": args.distill_weight,
        "trace_evidence_words": args.trace_evidence_words,
        "trace_holmes_span_words": args.trace_holmes_span_words,
        "sample_eval_steps": args.sample_eval_steps,
        "sample_eval_rows": args.sample_eval_rows,
        "sample_eval_max_new_tokens_trace": args.sample_eval_max_new_tokens_trace,
        "sample_eval_max_new_tokens_json": args.sample_eval_max_new_tokens_json,
        "status": "Training Completed",
        "log_dir": str(run_dir),
    }
    with open(os.path.join(run_dir, "experiments_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Experiment summary written.")


if __name__ == "__main__":
    main()
