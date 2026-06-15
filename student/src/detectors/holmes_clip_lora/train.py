import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from detectors.holmes_clip_lora.clip_model import CLIPBinaryModel
from detectors.holmes_clip_lora.config import DetectorConfig
from detectors.holmes_clip_lora.data import (
    build_train_transform,
    create_filelist_loader,
    create_imagefolder_loader,
    load_derived_filelist,
    summarize_binary_labels,
)
from detectors.holmes_clip_lora.validate import validate
from utils.eval_utils import ensure_split_manifest, row_ids_for_split


def parse_args():
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description="Train the Holmes CLIP LoRA detector.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(repo_root / "student" / "outputs" / "detectors" / "holmes_clip_lora_vitl14_336" / "config_train.yaml"),
    )
    parser.add_argument("--derived_data_path", type=str, default=None)
    parser.add_argument("--split_manifest_path", type=str, default=None)
    parser.add_argument("--eval_ratio", type=float, default=None)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--regenerate_split", action="store_true")
    parser.add_argument("--train_dataroot", type=str, default=None)
    parser.add_argument("--test_dataroot", type=str, default=None)
    parser.add_argument("--clip_weights", type=str, default=None)
    parser.add_argument("--checkpoints_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"holmes_clip_lora_train:{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def format_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_device(config: DetectorConfig) -> str:
    gpu_ids = str(config.get("training.gpu_ids", "0"))
    return f"cuda:{gpu_ids.split(',')[0].strip()}" if torch.cuda.is_available() else "cpu"


def create_loaders(config: DetectorConfig, model: CLIPBinaryModel):
    derived_data_path = config.get("data.derived_data_path")
    num_workers = int(config.get("system.num_threads", 8))
    train_batch_size = int(config.get("training.batch_size", 32))
    val_batch_size = int(config.get("testing.batch_size", 64))
    train_transform = build_train_transform(model.preprocess)

    if derived_data_path:
        dataset_path = Path(str(derived_data_path))
        split_manifest_raw = config.get("data.split_manifest_path")
        split_manifest_path, split_manifest = ensure_split_manifest(
            dataset_path,
            eval_ratio=float(config.get("data.eval_ratio", 0.1)),
            seed=int(config.get("training.split_seed", config.get("training.seed", 100))),
            manifest_path=Path(split_manifest_raw) if split_manifest_raw else None,
            regenerate=bool(config.get("data.regenerate_split", False)),
        )
        train_rows = load_derived_filelist(dataset_path, allowed_row_ids=row_ids_for_split(split_manifest, "train"))
        val_rows = load_derived_filelist(dataset_path, allowed_row_ids=row_ids_for_split(split_manifest, "eval"))
        train_loader = create_filelist_loader(
            train_rows,
            train_transform,
            batch_size=train_batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        val_loader = create_filelist_loader(
            val_rows,
            model.preprocess,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        metadata = {
            "data_source": "derived_jsonl",
            "derived_data_path": str(dataset_path),
            "split_manifest_path": str(split_manifest_path),
            "train_counts": summarize_binary_labels(train_rows),
            "val_counts": summarize_binary_labels(val_rows),
        }
        return train_loader, val_loader, metadata

    train_root = Path(str(config.get("data.train_dataroot"))) / str(config.get("data.train_split", "dataset"))
    val_root = Path(str(config.get("data.train_dataroot"))) / str(config.get("data.val_split", "val"))
    train_loader = create_imagefolder_loader(
        train_root,
        train_transform,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = create_imagefolder_loader(
        val_root,
        model.preprocess,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    metadata = {
        "data_source": "imagefolder",
        "train_root": str(train_root),
        "val_root": str(val_root),
        "train_counts": {},
        "val_counts": {},
    }
    return train_loader, val_loader, metadata


def checkpoint_payload(model: CLIPBinaryModel, epoch: int, global_step: int, metrics: dict, run_config: dict) -> dict:
    return {
        "fc": model.fc.state_dict(),
        "lora": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
        "run_config": run_config,
    }


def load_checkpoint_for_eval(
    checkpoint_path: Path,
    *,
    clip_weights: str,
    model_name: str,
    device: str,
) -> CLIPBinaryModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = CLIPBinaryModel(model_name, clip_weights)
    model.fc.load_state_dict(checkpoint["fc"])
    model.load_state_dict(checkpoint["lora"], strict=False)
    return model.to(device)


def main():
    args = parse_args()
    config = DetectorConfig(args.config)
    config.apply_overrides(args)
    if args.regenerate_split:
        config.set("data.regenerate_split", True)

    clip_weights = str(config.get("models.pretrained.clip_weights", ""))
    if not clip_weights or not Path(clip_weights).exists():
        raise FileNotFoundError("CLIP base weights are required for detector training")

    checkpoints_dir = Path(str(config.get("models.checkpoints_dir", "./checkpoints_clip_lora")))
    run_name = str(config.get("training.name", "clip_lora_holmes"))
    run_dir = checkpoints_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "resolved_config.yaml")

    logger = setup_logger(run_dir / str(config.get("logging.log_file", "training.log")))
    history_path = run_dir / "metrics_history.jsonl"
    if history_path.exists():
        history_path.unlink()

    seed = int(config.get("training.seed", 100))
    set_seed(seed)

    device = resolve_device(config)
    model_name = str(config.get("training.modelname", "CLIP:ViT-L/14@336px")).replace("CLIP:", "")
    model = CLIPBinaryModel(model_name, clip_weights)
    train_loader, val_loader, data_meta = create_loaders(config, model)

    for name, param in model.named_parameters():
        param.requires_grad = name in {"fc.weight", "fc.bias"} or "lora_" in name
    model = model.to(device)

    train_batch_size = int(config.get("training.batch_size", 32))
    val_batch_size = int(config.get("testing.batch_size", 64))
    epochs = int(config.get("training.niter", 5))
    num_workers = int(config.get("system.num_threads", 8))
    eval_threshold = float(config.get("testing.threshold", 0.5))
    log_interval = max(1, int(config.get("training.log_interval", 50)))
    max_steps = int(config.get("training.max_steps", 0) or 0)

    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config.get("training.lr", 1e-4)),
        betas=(float(config.get("training.beta1", 0.9)), 0.999),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("training loader is empty")
    total_planned_steps = steps_per_epoch * epochs
    if max_steps > 0:
        total_planned_steps = min(total_planned_steps, max_steps)

    logger.info(
        "Detector training start | device=%s | data_source=%s | epochs=%s | train_batch_size=%s | "
        "val_batch_size=%s | steps_per_epoch=%s | total_planned_steps=%s | num_workers=%s | threshold=%.3f",
        device,
        data_meta["data_source"],
        epochs,
        train_batch_size,
        val_batch_size,
        steps_per_epoch,
        total_planned_steps,
        num_workers,
        eval_threshold,
    )
    logger.info("Dataset summary | details=%s", json.dumps(data_meta, ensure_ascii=False, sort_keys=True))

    best_metrics = {"macro_f1": -1.0, "accuracy": -1.0, "average_precision": -1.0}
    best_checkpoint_path = None
    global_step = 0
    total_seen = 0
    start_time = time.monotonic()
    stop_training = False

    run_config = {
        "clip_weights": clip_weights,
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
        "epochs": epochs,
        "lr": float(config.get("training.lr", 1e-4)),
        "num_workers": num_workers,
        "threshold": eval_threshold,
        "max_steps": max_steps,
        **data_meta,
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_seen = 0
        epoch_start_time = time.monotonic()

        for epoch_step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device).float()

            logits = model(images).squeeze(1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = int(labels.size(0))
            batch_loss = float(loss.item())
            epoch_loss += batch_loss * batch_size
            epoch_seen += batch_size
            total_seen += batch_size
            global_step += 1

            should_log = (
                global_step == 1
                or global_step % log_interval == 0
                or global_step == total_planned_steps
                or epoch_step == steps_per_epoch
            )
            if should_log:
                elapsed = time.monotonic() - start_time
                steps_per_second = global_step / elapsed if elapsed > 0 else 0.0
                samples_per_second = total_seen / elapsed if elapsed > 0 else 0.0
                remaining_steps = max(total_planned_steps - global_step, 0)
                eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else None
                avg_loss = epoch_loss / max(epoch_seen, 1)
                progress_pct = 100.0 * global_step / max(total_planned_steps, 1)
                logger.info(
                    "epoch=%s/%s | epoch_step=%s/%s | global_step=%s/%s | progress=%.2f%% | "
                    "batch_loss=%.6f | avg_loss=%.6f | lr=%.7f | samples_per_sec=%.2f | eta=%s",
                    epoch + 1,
                    epochs,
                    epoch_step,
                    steps_per_epoch,
                    global_step,
                    total_planned_steps,
                    progress_pct,
                    batch_loss,
                    avg_loss,
                    optimizer.param_groups[0]["lr"],
                    samples_per_second,
                    format_seconds(eta_seconds),
                )
                append_jsonl(
                    history_path,
                    {
                        "event": "train_step",
                        "epoch": epoch + 1,
                        "epoch_step": epoch_step,
                        "global_step": global_step,
                        "total_planned_steps": total_planned_steps,
                        "progress_pct": round(progress_pct, 4),
                        "batch_loss": batch_loss,
                        "avg_loss": avg_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                        "samples_per_sec": samples_per_second,
                        "eta_seconds": eta_seconds,
                    },
                )

            if max_steps > 0 and global_step >= max_steps:
                stop_training = True
                break

        epoch_avg_loss = epoch_loss / max(epoch_seen, 1)
        live_metrics = validate(model, val_loader, device, threshold=eval_threshold, scan_thresholds=True)
        live_best_threshold_metrics = live_metrics.pop("best_threshold_metrics", {})
        epoch_elapsed = time.monotonic() - epoch_start_time
        payload = checkpoint_payload(
            model,
            epoch + 1,
            global_step,
            {**live_metrics, "best_threshold_metrics": live_best_threshold_metrics},
            run_config,
        )
        last_checkpoint_path = run_dir / "model_epoch_last.pth"
        torch.save(payload, last_checkpoint_path)

        reloaded_model = load_checkpoint_for_eval(
            last_checkpoint_path,
            clip_weights=clip_weights,
            model_name=model_name,
            device=device,
        )
        checkpoint_metrics = validate(reloaded_model, val_loader, device, threshold=eval_threshold, scan_thresholds=True)
        checkpoint_best_threshold_metrics = checkpoint_metrics.pop("best_threshold_metrics", {})
        del reloaded_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            "epoch=%s/%s | train_loss=%.6f | live_val_acc=%.4f | live_val_macro_f1=%.4f | "
            "checkpoint_val_acc=%.4f | checkpoint_val_macro_f1=%.4f | checkpoint_val_ap=%.4f | "
            "checkpoint_real_recall=%.4f | checkpoint_fake_recall=%.4f | val_threshold=%.3f | "
            "checkpoint_best_f1_threshold=%.2f | checkpoint_best_f1=%.4f | epoch_time=%s",
            epoch + 1,
            epochs,
            epoch_avg_loss,
            live_metrics["accuracy"],
            live_metrics["macro_f1"],
            checkpoint_metrics["accuracy"],
            checkpoint_metrics["macro_f1"],
            checkpoint_metrics["average_precision"],
            checkpoint_metrics["real_recall"],
            checkpoint_metrics["fake_recall"],
            eval_threshold,
            float(checkpoint_best_threshold_metrics.get("threshold", eval_threshold)),
            float(checkpoint_best_threshold_metrics.get("macro_f1", checkpoint_metrics["macro_f1"])),
            format_seconds(epoch_elapsed),
        )
        append_jsonl(
            history_path,
            {
                "event": "epoch_eval",
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": epoch_avg_loss,
                "epoch_seconds": epoch_elapsed,
                "live_val_metrics": live_metrics,
                "live_best_threshold_metrics": live_best_threshold_metrics,
                "checkpoint_val_metrics": checkpoint_metrics,
                "checkpoint_best_threshold_metrics": checkpoint_best_threshold_metrics,
            },
        )

        rank = (
            checkpoint_metrics["macro_f1"],
            checkpoint_metrics["accuracy"],
            checkpoint_metrics["average_precision"],
        )
        current_best_rank = (
            best_metrics["macro_f1"],
            best_metrics["accuracy"],
            best_metrics["average_precision"],
        )
        if rank > current_best_rank:
            best_metrics = dict(checkpoint_metrics)
            best_checkpoint_path = run_dir / (
                f"model_best_f1_{checkpoint_metrics['macro_f1']:.4f}_acc_{checkpoint_metrics['accuracy']:.4f}_epoch_{epoch + 1}.pth"
            )
            best_payload = checkpoint_payload(
                model,
                epoch + 1,
                global_step,
                {
                    **checkpoint_metrics,
                    "best_threshold_metrics": checkpoint_best_threshold_metrics,
                    "live_val_metrics": live_metrics,
                    "live_best_threshold_metrics": live_best_threshold_metrics,
                },
                run_config,
            )
            torch.save(best_payload, best_checkpoint_path)
            logger.info("best_checkpoint_updated | path=%s", best_checkpoint_path)

        if stop_training:
            logger.info("Max steps reached at global_step=%s; stopping after epoch %s validation.", global_step, epoch + 1)
            break

    elapsed = time.monotonic() - start_time
    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
        "best_metrics": best_metrics,
        "clip_weights": clip_weights,
        "device": device,
        "global_step": global_step,
        "planned_steps": total_planned_steps,
        "elapsed_seconds": elapsed,
        "resolved_config_path": str(run_dir / "resolved_config.yaml"),
        "history_path": str(history_path),
        **data_meta,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Detector training completed | elapsed=%s | summary=%s", format_seconds(elapsed), summary)


if __name__ == "__main__":
    main()
