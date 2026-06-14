from typing import Dict, Iterable, List


def average_precision(y_true: List[int], y_score: List[float]) -> float:
    positives = sum(y_true)
    if positives == 0:
        return 0.0
    order = sorted(range(len(y_true)), key=lambda idx: y_score[idx], reverse=True)
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0
    for idx in order:
        if y_true[idx]:
            tp += 1
        else:
            fp += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / positives
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def macro_f1_from_binary_labels(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)

    def class_f1(target: int) -> float:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == target and yp == target)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != target and yp == target)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == target and yp != target)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return (class_f1(0) + class_f1(1)) / 2.0


def compute_binary_metrics(y_true: List[int], y_score: List[float], threshold: float) -> Dict[str, float]:
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return {
        "accuracy": (tp + tn) / len(y_true) if y_true else 0.0,
        "macro_f1": macro_f1_from_binary_labels(y_true, y_pred),
        "average_precision": average_precision(y_true, y_score),
        "real_recall": tn / (tn + fp) if (tn + fp) else 0.0,
        "fake_recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "pred_fake_count": sum(y_pred),
        "pred_real_count": len(y_pred) - sum(y_pred),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

