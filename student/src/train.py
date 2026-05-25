import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
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
from qwen_vl_utils import process_vision_info

from dataset import HolmesSFTDataset, set_seed


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("TRAIN")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def append_to_experiments_md(
    summary, md_path="/ssd4/LPCVC2026/student/.context/04_EXPERIMENTS.md"
):
    if not os.path.exists(md_path):
        return

    block = (
        f"\n**Exp {summary['timestamp']} (SFT Run):**\n"
        f"- **Date**: [{summary['timestamp'][:8]}]\n"
        f"- **Task**: Student VLM Fine-Tuning using `{summary['model']}`\n"
        f"- **Seed**: {summary['seed']}\n"
        f"- **Batch Size**: {summary['batch_size']}, "
        f"**Epochs**: {summary['epochs']}, **LR**: {summary['lr']}\n"
        f"- **Status**: Trained\n"
        f"- **Logs/Outputs**: `{summary['log_dir']}`\n"
        f"- **Result & Observation**: Model successfully initialized and loaded with LoRA.\n"
        f"- **Conclusion**: Training pipeline established.\n\n---\n"
    )
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(block)


class EpochProgressCallback(TrainerCallback):
    """Per-epoch tqdm bar. No per-step metric spam — HF logs once per epoch."""

    def __init__(self, logger):
        self.logger = logger
        self.pbar = None
        self.steps_per_epoch = 0
        self.current_loss = None

    def on_train_begin(self, args, state, control, **kwargs):
        epochs = max(1, int(args.num_train_epochs))
        self.steps_per_epoch = max(1, state.max_steps // epochs)
        self.logger.info(
            f"Training: {epochs} epochs, ~{self.steps_per_epoch} steps/epoch "
            f"(total {state.max_steps})"
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

        # Capture loss to display on progress bar
        if "loss" in logs:
            self.current_loss = logs["loss"]

        # Only emit a terse summary when epoch drops
        if "epoch" in logs and abs(logs["epoch"] - round(logs["epoch"])) < 1e-4:
            parts = [f"step={state.global_step}"]
            for k in ("loss", "learning_rate", "grad_norm", "epoch"):
                v = logs.get(k)
                if isinstance(v, (int, float)):
                    parts.append(f"{k}={v:.4g}")
            self.logger.info("[train] " + " | ".join(parts))

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            self.pbar.close()
        self.logger.info("Training finished.")

class SampleGenerationCallback(TrainerCallback):
    """
    定期生成推論結果，確保模型在訓練過程中沒有壞掉 (崩潰/亂碼)。
    每隔一定 step (例如半個 epoch) 執行一次。
    """
    def __init__(self, logger, processor, image_path, prompt, generate_steps=500):
        self.logger = logger
        self.processor = processor
        self.image_path = image_path
        self.prompt = prompt
        self.generate_steps = generate_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.generate_steps == 0 and state.global_step > 0:
            self.logger.info(f"=== [Step {state.global_step}] Generating validation sample ===")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.image_path},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            # 使用模型原本所在的 device
            device = next(model.parameters()).device
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)
            
            model.eval()
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            
            self.logger.info(f"[Inference Result]:\n{output_text[0]}\n======================")
            model.train()


# ---------------------------------------------------------------------------
# Data collation
# ---------------------------------------------------------------------------

_VISION_KEYS = {
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
}
_PAD_KEYS = {"input_ids", "attention_mask", "labels"}


def custom_collate_fn(features):
    batch = {}
    keys = features[0].keys()

    for key in keys:
        tensors = [f[key] for f in features if f.get(key) is not None]
        if not tensors:
            continue

        if key in _VISION_KEYS:
            batch[key] = torch.cat(tensors, dim=0)
            continue

        # 自動判斷是否需要 Padding: 
        # 如果是 1D tensor 且長度不一致，或者是在 _PAD_KEYS 中
        shapes = [t.shape for t in tensors if torch.is_tensor(t)]
        is_varying_length = len(set(shapes)) > 1
        
        if key in _PAD_KEYS or (len(shapes) > 0 and tensors[0].dim() == 1 and is_varying_length):
            pad_value = -100 if key == "labels" else 0
            batch[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=pad_value
            )
        elif torch.is_tensor(tensors[0]):
            if tensors[0].dim() > 0:
                batch[key] = torch.stack(tensors, dim=0)
            else:
                batch[key] = torch.stack(tensors, dim=0)
        else:
            batch[key] = torch.tensor(tensors)

    # 確保 labels 存在
    if "input_ids" in batch and "labels" not in batch:
        batch["labels"] = batch["input_ids"].clone()
        if "attention_mask" in batch:
            batch["labels"][batch["attention_mask"] == 0] = -100
    
    return batch


# ---------------------------------------------------------------------------
# Model / args builders
# ---------------------------------------------------------------------------


def _resolve_model_class(name):
    if "Qwen2.5-VL" in name:
        return Qwen2_5_VLForConditionalGeneration
    if "Qwen2-VL" in name:
        return Qwen2VLForConditionalGeneration
    return AutoModelForCausalLM


def build_model(model_name_or_path, local_files_only):
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
        # Top-level attn_implementation gets overridden by vision_config,
        # so force sdpa on both sub-configs explicitly.
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
    return model


def build_training_args(args, run_dir, dataset_len=None):
    if dataset_len is not None:
        total_steps = dataset_len // (args.batch_size * 8)
        save_steps = max(1, total_steps // 2)
    else:
        save_steps = 100

    return TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        logging_dir=os.path.join(run_dir, "tensorboard_logs"),
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
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


def parse_args():
    p = argparse.ArgumentParser(description="Student VLM Fine-Tuning")
    p.add_argument(
        "--model_name_or_path", type=str, default="Qwen/Qwen2-VL-2B-Instruct"
    )
    p.add_argument(
        "--data_path",
        type=str,
        default="/ssd4/LPCVC2026/holmes_lpcvc3_multi_teacher/stage1_g31b_v5_full_balanced",
    )
    p.add_argument(
        "--prompt_dir",
        type=str,
        default="/ssd4/LPCVC2026/Qualcomm_tools/26LPCVC_Track3_Sample_Solution/dataset/prompts/",
    )
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval_steps", type=int, default=None, help="Number of steps between sample generations.")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Checkpoint folder, or 'True' to auto-resume from output_dir's latest.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def reserve_vram(fraction=0.85):
    """
    Allocates a large chunk of VRAM and deletes it immediately.
    PyTorch's caching allocator will keep this memory reserved for the current process,
    preventing other processes from taking it and causing OOM later.
    """
    if not torch.cuda.is_available():
        return

    device = torch.cuda.current_device()
    free_mem, _ = torch.cuda.mem_get_info(device)
    reserve_bytes = int(free_mem * fraction)

    try:
        # Pre-allocate using uint8 (1 byte per element)
        block = torch.empty(reserve_bytes, dtype=torch.uint8, device="cuda")
        del block  # Free it back to PyTorch's memory cache, NOT the OS
    except RuntimeError as e:
        pass


def main():
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logger(run_dir)

    logger.info(f"SFT start | model={args.model_name_or_path} | run_dir={run_dir}")
    logger.info(
        f"Config | data={args.data_path} | bs={args.batch_size} | "
        f"epochs={args.epochs} | lr={args.lr} | local_only={args.local_files_only}"
    )

    # Processor
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only
    )
    logger.info("Processor loaded.")

    # Dataset
    dataset_path = args.data_path
    if os.path.isdir(dataset_path):
        dataset_path = os.path.join(dataset_path, "holmes_lpcvc_sft.jsonl")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    train_dataset = HolmesSFTDataset(dataset_path, args.prompt_dir, processor=processor)
    logger.info(f"Dataset loaded: {len(train_dataset)} samples from {dataset_path}")

    # 取得一個 Stage 1 prompt 用於 validation 回報
    sample_val_prompt = "Is this image real or fake? Think step-by-step before giving a conclusion. Please analyze based on the following three aspects: Edge & Boundary Integrity, Texture & Resolution Coherence, Material & Object Detail Fidelity."
    if train_dataset.prompts.get("stage1"):
        sample_val_prompt = train_dataset.prompts["stage1"][0]

    if len(train_dataset) > 0:
        sample = train_dataset[0]
        shape_info = {
            k: (
                tuple(v.shape)
                if torch.is_tensor(v) or isinstance(v, np.ndarray)
                else type(v).__name__
            )
            for k, v in sample.items()
        }
        logger.info(f"Sample keys/shapes: {shape_info}")

    # Model
    logger.info("Building model with 4-bit quant + LoRA adapters...")
    model = build_model(args.model_name_or_path, args.local_files_only)
    model.print_trainable_parameters()

    # Train
    training_args = build_training_args(args, run_dir, dataset_len=len(train_dataset))
    
    # 決定多少步生一次 (例如每半個 epoch)
    if args.eval_steps is not None:
        generate_steps = args.eval_steps
    else:
        total_steps = len(train_dataset) // (args.batch_size * 8)
        generate_steps = max(10, total_steps // 2)

    val_image_path = "/ssd4/LPCVC2026/student/teacher_dataset_04211726/images/1_fake/code_lcm-lora-sdv1-5_train2017_000000061697.jpg"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_collate_fn,
        callbacks=[
            EpochProgressCallback(logger),
            SampleGenerationCallback(logger, processor, val_image_path, sample_val_prompt, generate_steps)
        ],
    )
    trainer.remove_callback(PrinterCallback)

    resume_ckpt = None
    if args.resume_from_checkpoint:
        resume_ckpt = (
            True
            if args.resume_from_checkpoint.lower() == "true"
            else args.resume_from_checkpoint
        )
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(run_dir)
    processor.save_pretrained(run_dir)
    logger.info("Training completed and model saved.")

    # Summary
    summary = {
        "timestamp": timestamp,
        "model": args.model_name_or_path,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "status": "Training Completed",
        "log_dir": run_dir,
    }
    with open(os.path.join(run_dir, "experiments_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    append_to_experiments_md(summary)
    logger.info("Experiment summary written.")


if __name__ == "__main__":
    main()
