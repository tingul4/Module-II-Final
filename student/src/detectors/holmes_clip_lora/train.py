import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from detectors.holmes_clip_lora.config import DetectorConfig
from detectors.holmes_clip_lora.data import build_train_transform, create_imagefolder_loader
from detectors.holmes_clip_lora.clip_model import CLIPBinaryModel
from detectors.holmes_clip_lora.validate import validate


def parse_args():
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description="Train the Holmes CLIP LoRA detector from a YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(repo_root / "student" / "outputs" / "detectors" / "holmes_clip_lora_vitl14_336" / "config_train.yaml"),
    )
    parser.add_argument("--train_dataroot", type=str, default=None)
    parser.add_argument("--test_dataroot", type=str, default=None)
    parser.add_argument("--clip_weights", type=str, default=None)
    parser.add_argument("--checkpoints_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"holmes_clip_lora_train:{log_path}")
    logger.setLevel(logging.INFO)
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


def main():
    args = parse_args()
    config = DetectorConfig(args.config)
    config.apply_overrides(args)

    clip_weights = str(config.get("models.pretrained.clip_weights", ""))
    if not clip_weights or not Path(clip_weights).exists():
        raise FileNotFoundError("CLIP base weights are required for detector training")

    train_root = Path(str(config.get("data.train_dataroot"))) / str(config.get("data.train_split", "dataset"))
    val_root = Path(str(config.get("data.train_dataroot"))) / str(config.get("data.val_split", "val"))
    checkpoints_dir = Path(str(config.get("models.checkpoints_dir", "./checkpoints_clip_lora")))
    run_name = str(config.get("training.name", "clip_lora_holmes"))
    run_dir = checkpoints_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "resolved_config.yaml")

    logger = setup_logger(run_dir / str(config.get("logging.log_file", "training.log")))
    seed = int(config.get("training.seed", 100))
    set_seed(seed)

    gpu_ids = str(config.get("training.gpu_ids", "0"))
    device = f"cuda:{gpu_ids.split(',')[0].strip()}" if torch.cuda.is_available() else "cpu"
    logger.info("Detector training start | train_root=%s | val_root=%s | device=%s", train_root, val_root, device)

    model = CLIPBinaryModel(str(config.get("training.modelname", "CLIP:ViT-L/14@336px")).replace("CLIP:", ""), clip_weights)
    train_transform = build_train_transform(model.preprocess)
    train_loader = create_imagefolder_loader(
        train_root,
        train_transform,
        batch_size=int(config.get("training.batch_size", 32)),
        shuffle=True,
        num_workers=int(config.get("system.num_threads", 8)),
    )
    val_loader = create_imagefolder_loader(
        val_root,
        model.preprocess,
        batch_size=int(config.get("testing.batch_size", 64)),
        shuffle=False,
        num_workers=int(config.get("system.num_threads", 8)),
    )

    for name, param in model.named_parameters():
        param.requires_grad = name in {"fc.weight", "fc.bias"} or "lora_" in name
    model = model.to(device)

    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config.get("training.lr", 1e-4)),
        betas=(float(config.get("training.beta1", 0.9)), 0.999),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_acc = -1.0
    best_ap = -1.0
    epochs = int(config.get("training.niter", 5))
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            logits = model(images).squeeze(1)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
            seen += int(labels.size(0))

        metrics = validate(model, val_loader, device, threshold=0.5)
        avg_loss = running_loss / max(seen, 1)
        logger.info(
            "epoch=%s/%s | train_loss=%.6f | val_acc=%.4f | val_macro_f1=%.4f | val_ap=%.4f",
            epoch + 1,
            epochs,
            avg_loss,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["average_precision"],
        )
        checkpoint = {
            "fc": model.fc.state_dict(),
            "lora": model.state_dict(),
            "epoch": epoch + 1,
            "metrics": metrics,
        }
        torch.save(checkpoint, run_dir / "model_epoch_last.pth")
        if (metrics["accuracy"], metrics["average_precision"]) > (best_acc, best_ap):
            best_acc = metrics["accuracy"]
            best_ap = metrics["average_precision"]
            best_path = run_dir / f"model_epoch_{best_acc:.2f}_{best_ap:.2f}.pth"
            torch.save(checkpoint, best_path)

    summary = {
        "run_dir": str(run_dir),
        "best_accuracy": best_acc,
        "best_average_precision": best_ap,
        "clip_weights": clip_weights,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Detector training completed | summary=%s", summary)


if __name__ == "__main__":
    main()
