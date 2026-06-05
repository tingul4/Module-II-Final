import argparse
import html
import json
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "student" / "outputs" / "gemma4_lightweight_eval"
DEFAULT_DETECTOR = DEFAULT_OUTPUT_ROOT / "detector_clip_lora_gpu.jsonl"
DEFAULT_GEMMA = DEFAULT_OUTPUT_ROOT / "predictions" / "ckpt4000_single_stage.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare detector-only and detector+Gemma fusion on the lightweight slice.")
    parser.add_argument("--detector_scores_path", type=str, default=str(DEFAULT_DETECTOR))
    parser.add_argument("--gemma_predictions_path", type=str, default=str(DEFAULT_GEMMA))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output_prefix", type=str, default="detector_fusion_analysis")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision_fake = tp / (tp + fp) if (tp + fp) else 0.0
    precision_real = tn / (tn + fn) if (tn + fn) else 0.0
    fake_f1 = 2 * precision_fake * fake_recall / (precision_fake + fake_recall) if (precision_fake + fake_recall) else 0.0
    real_f1 = 2 * precision_real * real_recall / (precision_real + real_recall) if (precision_real + real_recall) else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": (real_f1 + fake_f1) / 2.0,
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "pred_fake_count": sum(y_pred),
        "pred_real_count": len(y_pred) - sum(y_pred),
    }


def gemma_positive_flag(row: Dict[str, object]) -> int:
    final_json = row.get("final_json") or {}
    entries = final_json.get("per_criterion") or []
    return 1 if any(int(entry.get("aigc score", 0) or 0) for entry in entries if isinstance(entry, dict)) else 0


def grid_search(detector_rows: List[Dict[str, object]], gemma_rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_row_id = {int(row["row_id"]): row for row in gemma_rows}
    joined = []
    for detector in detector_rows:
        gemma = by_row_id.get(int(detector["row_id"]))
        if gemma is None:
            continue
        joined.append(
            {
                "row_id": int(detector["row_id"]),
                "gold_label": int(detector["gold_label"]),
                "detector_score": float(detector["detector_score"]),
                "gemma_positive_flag": gemma_positive_flag(gemma),
            }
        )
    if not joined:
        raise ValueError("No overlapping row_ids between detector and gemma predictions")

    best_detector = None
    for step in range(101):
        threshold = step / 100.0
        y_true = [row["gold_label"] for row in joined]
        y_pred = [1 if row["detector_score"] >= threshold else 0 for row in joined]
        metrics = compute_metrics(y_true, y_pred)
        candidate = {"threshold": threshold, **metrics}
        if best_detector is None or (candidate["accuracy"], candidate["macro_f1"]) > (best_detector["accuracy"], best_detector["macro_f1"]):
            best_detector = candidate

    best_fusion = None
    for detector_weight_step in range(0, 21):
        detector_weight = detector_weight_step / 20.0
        gemma_weight = 1.0 - detector_weight
        for threshold_step in range(101):
            threshold = threshold_step / 100.0
            y_true = [row["gold_label"] for row in joined]
            y_pred = []
            for row in joined:
                score = detector_weight * row["detector_score"] + gemma_weight * row["gemma_positive_flag"]
                y_pred.append(1 if score >= threshold else 0)
            metrics = compute_metrics(y_true, y_pred)
            candidate = {
                "threshold": threshold,
                "detector_weight": detector_weight,
                "gemma_weight": gemma_weight,
                **metrics,
            }
            if best_fusion is None or (candidate["accuracy"], candidate["macro_f1"]) > (best_fusion["accuracy"], best_fusion["macro_f1"]):
                best_fusion = candidate

    gemma_metrics = compute_metrics(
        [row["gold_label"] for row in joined],
        [row["gemma_positive_flag"] for row in joined],
    )

    return {
        "sample_count": len(joined),
        "gemma_positive_flag_baseline": gemma_metrics,
        "best_detector_only": best_detector,
        "best_detector_gemma_flag_fusion": best_fusion,
    }


def render_html(report: Dict[str, object]) -> str:
    sections = []
    for title, payload in (
        ("Gemma Positive-Flag Baseline", report["gemma_positive_flag_baseline"]),
        ("Best Detector Only", report["best_detector_only"]),
        ("Best Detector + Gemma Flag Fusion", report["best_detector_gemma_flag_fusion"]),
    ):
        rows = []
        for key, value in payload.items():
            rows.append(
                "<tr>"
                f"<th>{html.escape(str(key))}</th>"
                f"<td>{html.escape(f'{value:.6f}' if isinstance(value, float) else str(value))}</td>"
                "</tr>"
            )
        sections.append(f"<h2>{html.escape(title)}</h2><table>{''.join(rows)}</table>")
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Detector Fusion Analysis</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;}table{border-collapse:collapse;margin-bottom:24px;}th,td{border:1px solid #ccc;padding:8px 10px;text-align:left;}th{background:#f4f4f4;}</style>"
        "</head><body>"
        f"<h1>Detector Fusion Analysis</h1><p>sample_count={report['sample_count']}</p>"
        + "".join(sections)
        + "</body></html>"
    )


def main():
    args = parse_args()
    detector_rows = load_jsonl(Path(args.detector_scores_path))
    gemma_rows = load_jsonl(Path(args.gemma_predictions_path))
    report = grid_search(detector_rows, gemma_rows)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_json = output_root / f"{args.output_prefix}.json"
    output_html = output_root / f"{args.output_prefix}.html"
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_html.write_text(render_html(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
