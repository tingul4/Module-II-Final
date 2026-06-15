from typing import Dict, Iterable, List, Tuple

import torch

from detectors.holmes_clip_lora.metrics import compute_binary_metrics


def collect_scores(model, data_loader, device: str) -> Tuple[List[int], List[float]]:
    y_true = []
    y_score = []
    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images.to(device))
            y_score.extend(logits.sigmoid().flatten().detach().cpu().tolist())
            y_true.extend(labels.detach().cpu().flatten().tolist())
    return [int(item) for item in y_true], [float(item) for item in y_score]


def find_best_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    thresholds: Iterable[float] | None = None,
) -> Dict[str, float]:
    y_true = list(y_true)
    y_score = list(y_score)
    thresholds = list(thresholds) if thresholds is not None else [step / 100.0 for step in range(5, 96)]

    best_threshold = 0.5
    best_metrics = compute_binary_metrics(y_true, y_score, best_threshold)
    best_rank = (
        best_metrics["macro_f1"],
        best_metrics["accuracy"],
        best_metrics["average_precision"],
        -abs(best_threshold - 0.5),
    )
    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, y_score, threshold)
        rank = (
            metrics["macro_f1"],
            metrics["accuracy"],
            metrics["average_precision"],
            -abs(float(threshold) - 0.5),
        )
        if rank > best_rank:
            best_rank = rank
            best_threshold = float(threshold)
            best_metrics = metrics
    return {
        "threshold": float(best_threshold),
        **best_metrics,
    }


def validate(
    model,
    data_loader,
    device: str,
    threshold: float = 0.5,
    *,
    scan_thresholds: bool = False,
) -> Dict[str, float]:
    y_true, y_score = collect_scores(model, data_loader, device)
    metrics = compute_binary_metrics(y_true, y_score, threshold)
    metrics["num_samples"] = len(y_true)
    if scan_thresholds:
        metrics["best_threshold_metrics"] = find_best_threshold(y_true, y_score)
    return metrics
