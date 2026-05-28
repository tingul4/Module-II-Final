import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from task_utils import CRITERIA, FAKE_LABEL


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-8 or y_std < 1e-8:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else 0.0


def rgb_image_to_features(image: Image.Image, patch_grid: int = 4) -> np.ndarray:
    image = image.convert("RGB").resize((256, 256))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)

    gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    grad = gx + gy

    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(fft))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    low = mag[cy - 16 : cy + 16, cx - 16 : cx + 16]
    high = np.concatenate(
        [
            mag[: cy - 32, :].reshape(-1),
            mag[cy + 32 :, :].reshape(-1),
            mag[cy - 32 : cy + 32, : cx - 32].reshape(-1),
            mag[cy - 32 : cy + 32, cx + 32 :].reshape(-1),
        ]
    )

    features: List[float] = []
    for channel_idx in range(3):
        channel = arr[:, :, channel_idx]
        features.extend(
            [
                float(channel.mean()),
                float(channel.std()),
                float(np.percentile(channel, 10)),
                float(np.percentile(channel, 90)),
            ]
        )

    features.extend(
        [
            float(gray.mean()),
            float(gray.std()),
            float(grad.mean()),
            float(grad.std()),
            float((grad > grad.mean()).mean()),
            float(low.mean()),
            float(low.std()),
            float(high.mean()) if high.size else 0.0,
            float(high.std()) if high.size else 0.0,
            float((high.mean() + 1e-6) / (low.mean() + 1e-6)) if high.size else 0.0,
        ]
    )

    patch_h = gray.shape[0] // patch_grid
    patch_w = gray.shape[1] // patch_grid
    patch_means = []
    patch_stds = []
    patch_grads = []
    for py in range(patch_grid):
        for px in range(patch_grid):
            patch = gray[py * patch_h : (py + 1) * patch_h, px * patch_w : (px + 1) * patch_w]
            patch_grad = grad[py * patch_h : (py + 1) * patch_h, px * patch_w : (px + 1) * patch_w]
            patch_means.append(float(patch.mean()))
            patch_stds.append(float(patch.std()))
            patch_grads.append(float(patch_grad.mean()))

    features.extend(
        [
            float(np.mean(patch_means)),
            float(np.std(patch_means)),
            float(np.mean(patch_stds)),
            float(np.std(patch_stds)),
            float(np.mean(patch_grads)),
            float(np.std(patch_grads)),
            float(np.max(patch_grads)),
            float(np.min(patch_grads)),
        ]
    )

    channel_corr_rg = _safe_corrcoef(arr[:, :, 0].reshape(-1), arr[:, :, 1].reshape(-1))
    channel_corr_rb = _safe_corrcoef(arr[:, :, 0].reshape(-1), arr[:, :, 2].reshape(-1))
    channel_corr_gb = _safe_corrcoef(arr[:, :, 1].reshape(-1), arr[:, :, 2].reshape(-1))
    features.extend([channel_corr_rg, channel_corr_rb, channel_corr_gb])

    feature_array = np.asarray(features, dtype=np.float32)
    return np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)


class VisualExpertModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def iter_derived_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_targets(row: Dict[str, object]) -> np.ndarray:
    final_target = row["final_json_target"]
    per_criterion = final_target["per_criterion"]
    overall = 1.0 if final_target["overall_likelihood"] == FAKE_LABEL else 0.0
    criterion_targets = [float(int(item.get("aigc score", 0) or 0)) for item in per_criterion]
    return np.asarray([overall] + criterion_targets, dtype=np.float32)


def resolve_image_path(dataset_path: Path, image_value: str) -> Path:
    return dataset_path.parent / image_value


def resolve_image_path_from_row(dataset_path: Path, row: Dict[str, object]) -> Path:
    image_value = str(row["image"])
    image_path = Path(image_value)
    if image_path.is_absolute() and image_path.exists():
        return image_path
    image_root = row.get("image_root")
    if image_root:
        candidate = Path(str(image_root)) / image_value
        if candidate.exists():
            return candidate
    candidate = resolve_image_path(dataset_path, image_value)
    if candidate.exists():
        return candidate
    return candidate


def load_training_arrays(derived_jsonl: Path, max_samples: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    features: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    image_keys: List[str] = []
    for row_idx, row in enumerate(iter_derived_rows(derived_jsonl)):
        if max_samples and row_idx >= max_samples:
            break
        image_path = resolve_image_path_from_row(derived_jsonl, row)
        with Image.open(image_path) as image:
            features.append(rgb_image_to_features(image))
        targets.append(build_targets(row))
        image_keys.append(str(row["image"]))
    return np.stack(features), np.stack(targets), image_keys


def train_visual_expert(
    derived_jsonl: Path,
    output_root: Path,
    epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
    batch_size: int = 128,
    max_samples: int = 0,
) -> Dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    features, targets, image_keys = load_training_arrays(derived_jsonl, max_samples=max_samples)
    perm = np.random.permutation(len(features))
    split = max(1, math.floor(len(features) * 0.9))
    train_idx = perm[:split]
    val_idx = perm[split:] if split < len(features) else perm[:1]

    mean = features[train_idx].mean(axis=0)
    std = features[train_idx].std(axis=0) + 1e-6
    features = (features - mean) / std

    model = VisualExpertModel(input_dim=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_x = torch.tensor(features[train_idx], dtype=torch.float32)
    train_y = torch.tensor(targets[train_idx], dtype=torch.float32)
    val_x = torch.tensor(features[val_idx], dtype=torch.float32)
    val_y = torch.tensor(targets[val_idx], dtype=torch.float32)

    best_state = None
    best_val_loss = float("inf")
    for _ in range(epochs):
        model.train()
        order = torch.randperm(train_x.size(0))
        for start in range(0, train_x.size(0), batch_size):
            batch_idx = order[start : start + batch_size]
            logits = model(train_x[batch_idx])
            loss = loss_fn(logits, train_y[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x), val_y).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "expert.pt"
    logits_path = output_root / "dataset_logits.jsonl"
    metadata_path = output_root / "metadata.json"

    with torch.no_grad():
        all_logits = model(torch.tensor(features, dtype=torch.float32)).numpy()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": features.shape[1],
            "hidden_dim": 128,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "criteria": CRITERIA,
        },
        checkpoint_path,
    )

    with logits_path.open("w", encoding="utf-8") as handle:
        for image_key, logits in zip(image_keys, all_logits):
            row = {
                "image": image_key,
                "overall_logit": float(logits[0]),
                "criterion_logits": [float(value) for value in logits[1:]],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "derived_jsonl": str(derived_jsonl),
        "checkpoint_path": str(checkpoint_path),
        "logits_path": str(logits_path),
        "val_loss": best_val_loss,
        "feature_dim": int(features.shape[1]),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def load_visual_expert(checkpoint_path: Path) -> Tuple[VisualExpertModel, np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = VisualExpertModel(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload.get("hidden_dim", 128)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    std = np.asarray(payload["feature_std"], dtype=np.float32)
    return model, mean, std


def predict_visual_expert(checkpoint_path: Path, image: Image.Image) -> Tuple[float, List[float]]:
    model, mean, std = load_visual_expert(checkpoint_path)
    features = rgb_image_to_features(image)
    features = (features - mean) / std
    with torch.no_grad():
        logits = model(torch.tensor(features[None, :], dtype=torch.float32))[0]
        probs = torch.sigmoid(logits).tolist()
    return float(probs[0]), [float(value) for value in probs[1:]]
