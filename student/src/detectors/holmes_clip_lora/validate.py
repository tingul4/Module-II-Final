from typing import Dict

import torch

from detectors.holmes_clip_lora.metrics import compute_binary_metrics


def validate(model, data_loader, device: str, threshold: float = 0.5) -> Dict[str, float]:
    y_true = []
    y_score = []
    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images.to(device))
            y_score.extend(logits.sigmoid().flatten().detach().cpu().tolist())
            y_true.extend(labels.detach().cpu().flatten().tolist())
    return compute_binary_metrics([int(item) for item in y_true], [float(item) for item in y_score], threshold)

