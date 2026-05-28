import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image

from inference import apply_expert_fusion, generate_text, generate_trace_payload, load_model, load_prompts
from task_utils import CRITERIA, compact_json_dumps, compact_trace_payload, safe_json_loads


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a multitask image-authenticity student model.")
    parser.add_argument("--base_model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--derived_data_path", type=str, required=True)
    parser.add_argument("--prompt_dir", type=str, default=str(REPO_ROOT / "prompts"))
    parser.add_argument("--expert_path", type=str, default=None)
    parser.add_argument("--fusion_alpha", type=float, default=0.8)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--probe_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens_trace", type=int, default=1536)
    parser.add_argument("--max_new_tokens_json", type=int, default=1024)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def render_html_report(title: str, report: dict, output_path: Path):
    rows = []
    for criterion, metrics in report.get("per_criterion_f1", {}).items():
        rows.append(
            "<tr>"
            f"<td>{criterion}</td>"
            f"<td>{metrics.get('precision', 0.0):.3f}</td>"
            f"<td>{metrics.get('recall', 0.0):.3f}</td>"
            f"<td>{metrics.get('f1', 0.0):.3f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #ddd; padding: 12px; }}
    .value {{ font-size: 24px; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="metrics">
    <div class="metric"><div>Final JSON Parse</div><div class="value">{report.get('json_parse_rate', 0.0):.3f}</div></div>
    <div class="metric"><div>Trace JSON Parse</div><div class="value">{report.get('trace_json_parse_rate', 0.0):.3f}</div></div>
    <div class="metric"><div>Overall Accuracy</div><div class="value">{report.get('overall_accuracy', 0.0):.3f}</div></div>
    <div class="metric"><div>Macro F1</div><div class="value">{report.get('macro_f1', 0.0):.3f}</div></div>
    <div class="metric"><div>Support Type Accuracy</div><div class="value">{report.get('support_type_accuracy', 0.0):.3f}</div></div>
    <div class="metric"><div>Taxonomy Accuracy</div><div class="value">{report.get('taxonomy_accuracy', 0.0):.3f}</div></div>
    <div class="metric"><div>Consistency Score</div><div class="value">{report.get('consistency_score', 0.0):.3f}</div></div>
    <div class="metric"><div>Real False Positive Rate</div><div class="value">{report.get('real_false_positive_rate', 0.0):.3f}</div></div>
    <div class="metric"><div>Blank Probe Parse</div><div class="value">{report.get('blank_probe', {}).get('json_parse_rate', 0.0):.3f}</div></div>
  </div>
  <table>
    <thead>
      <tr><th>Criterion</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve_image_path(dataset_path: Path, row: dict) -> str:
    image_path = Path(row["image"])
    if image_path.is_absolute() and image_path.exists():
        return str(image_path)
    image_root = row.get("image_root")
    if image_root:
        candidate = Path(str(image_root)) / row["image"]
        if candidate.exists():
            return str(candidate)
    return str(dataset_path.parent / row["image"])


def normalize_criterion_entries(payload: dict) -> dict:
    entries = payload.get("per_criterion", []) if isinstance(payload, dict) else []
    normalized = {}
    if not isinstance(entries, list):
        return normalized
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        criterion = entry.get("criterion")
        if criterion in CRITERIA:
            normalized[criterion] = entry
    return normalized


def evaluate_prediction(pred: dict, gold: dict, counts: dict):
    counts["samples"] += 1
    counts["json_parse_ok"] += int(bool(pred))
    if gold.get("overall_likelihood") == "Real":
        counts["real_samples"] += 1
    if not pred:
        return
    counts["overall_correct"] += int(pred.get("overall_likelihood") == gold.get("overall_likelihood"))
    pred_entries = normalize_criterion_entries(pred)
    gold_entries = gold.get("per_criterion", [])
    if gold.get("overall_likelihood") == "Real" and any(
        int(item.get("aigc score", 0) or 0) for item in pred_entries.values()
    ):
        counts["real_false_positive_samples"] += 1
    for gold_entry in gold_entries:
        pred_entry = pred_entries.get(gold_entry["criterion"], {})
        pred_score = 1 if int(pred_entry.get("aigc score", 0) or 0) else 0
        gold_score = 1 if int(gold_entry.get("aigc score", 0) or 0) else 0
        criterion = gold_entry["criterion"]
        if pred_score and gold_score:
            counts["tp"][criterion] += 1
        elif pred_score and not gold_score:
            counts["fp"][criterion] += 1
        elif gold_score and not pred_score:
            counts["fn"][criterion] += 1


def evaluate_trace_prediction(pred: dict, gold: dict, counts: dict):
    if not pred:
        return
    counts["trace_json_parse_ok"] += 1
    pred_entries = normalize_criterion_entries(pred)
    for gold_entry in gold.get("per_criterion", []):
        pred_entry = pred_entries.get(gold_entry["criterion"], {})
        counts["support_total"] += 1
        counts["support_correct"] += int(pred_entry.get("support_type") == gold_entry.get("support_type"))
        counts["taxonomy_total"] += 1
        counts["taxonomy_correct"] += int(
            pred_entry.get("artifact_taxonomy") == gold_entry.get("artifact_taxonomy")
        )
        counts["consistency_total"] += 1
        pred_consistent = int(pred_entry.get("score", 0) or 0) == 0 or (
            pred_entry.get("evidence") not in {"", None} and pred_entry.get("support_type") != "unsupported"
        )
        gold_consistent = not bool(gold_entry.get("artifact_score_conflict"))
        counts["consistency_correct"] += int(pred_consistent == gold_consistent)


def finalize_metrics(counts: dict):
    per_criterion_f1 = {}
    macro_f1_values = []
    for criterion in CRITERIA:
        tp = counts["tp"][criterion]
        fp = counts["fp"][criterion]
        fn = counts["fn"][criterion]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_criterion_f1[criterion] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        macro_f1_values.append(f1)
    return {
        "json_parse_rate": counts["json_parse_ok"] / counts["samples"] if counts["samples"] else 0.0,
        "trace_json_parse_rate": counts["trace_json_parse_ok"] / counts["samples"] if counts["samples"] else 0.0,
        "overall_accuracy": counts["overall_correct"] / counts["samples"] if counts["samples"] else 0.0,
        "macro_f1": sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else 0.0,
        "per_criterion_f1": per_criterion_f1,
        "support_type_accuracy": counts["support_correct"] / counts["support_total"] if counts["support_total"] else 0.0,
        "taxonomy_accuracy": counts["taxonomy_correct"] / counts["taxonomy_total"] if counts["taxonomy_total"] else 0.0,
        "consistency_score": counts["consistency_correct"] / counts["consistency_total"] if counts["consistency_total"] else 0.0,
    }


def main():
    args = parse_args()
    dataset_path = Path(args.derived_data_path)
    rows = list(iter_rows(dataset_path))[: args.max_samples]
    processor, model = load_model(args.base_model, args.adapter_path, local_files_only=args.local_files_only)
    evidence_trace_prompt, final_json_prompt = load_prompts(args.prompt_dir)

    counts = {
        "samples": 0,
        "json_parse_ok": 0,
        "trace_json_parse_ok": 0,
        "overall_correct": 0,
        "tp": {criterion: 0 for criterion in CRITERIA},
        "fp": {criterion: 0 for criterion in CRITERIA},
        "fn": {criterion: 0 for criterion in CRITERIA},
        "support_total": 0,
        "support_correct": 0,
        "taxonomy_total": 0,
        "taxonomy_correct": 0,
        "consistency_total": 0,
        "consistency_correct": 0,
        "real_samples": 0,
        "real_false_positive_samples": 0,
    }

    blank_probe = {"parse_ok": 0, "overall_accuracy": 0, "samples": 0}
    shuffle_probe = {"overall_accuracy": 0, "samples": 0}
    oracle_probe = {"student_overall_accuracy": 0, "oracle_overall_accuracy": 0, "samples": 0}

    blank_image = Image.new("RGB", (512, 512), color=(0, 0, 0))
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        blank_image.save(handle.name)
        shuffled_rows = rows[1:] + rows[:1]
        for idx, row in enumerate(rows):
            image_path = resolve_image_path(dataset_path, row)
            trace_text, trace_json, _, trace_for_final, _ = generate_trace_payload(
                model,
                processor,
                image_path,
                evidence_trace_prompt,
                args.max_new_tokens_trace,
            )
            final_prompt = (
                f"{final_json_prompt}\n\n"
                "Here is the structured evidence trace for this image:\n"
                f"{trace_for_final}\n\n"
                "Use the trace to synthesize the final structured decision JSON."
            )
            final_text = generate_text(model, processor, image_path, final_prompt, args.max_new_tokens_json)
            pred_json, _ = safe_json_loads(final_text)
            if pred_json and args.expert_path:
                pred_json = apply_expert_fusion(pred_json, args.expert_path, image_path, args.fusion_alpha)
            evaluate_prediction(pred_json, row["final_json_target"], counts)
            evaluate_trace_prediction(trace_json, row["evidence_trace_target"], counts)

            if idx < args.probe_samples:
                blank_trace, blank_trace_json, _, blank_trace_for_final, _ = generate_trace_payload(
                    model,
                    processor,
                    handle.name,
                    evidence_trace_prompt,
                    args.max_new_tokens_trace,
                )
                blank_final = generate_text(
                    model,
                    processor,
                    handle.name,
                    f"{final_json_prompt}\n\nHere is the structured evidence trace for this image:\n{blank_trace_for_final}\n\nUse the trace to synthesize the final structured decision JSON.",
                    args.max_new_tokens_json,
                )
                blank_json, _ = safe_json_loads(blank_final)
                blank_probe["samples"] += 1
                blank_probe["parse_ok"] += int(bool(blank_json))
                blank_probe["overall_accuracy"] += int(
                    blank_json.get("overall_likelihood") == row["final_json_target"]["overall_likelihood"]
                ) if blank_json else 0

                shuffle_row = shuffled_rows[idx]
                shuffle_image_path = resolve_image_path(dataset_path, shuffle_row)
                shuffle_trace, shuffle_trace_json, _, shuffle_trace_for_final, _ = generate_trace_payload(
                    model,
                    processor,
                    shuffle_image_path,
                    evidence_trace_prompt,
                    args.max_new_tokens_trace,
                )
                shuffle_final = generate_text(
                    model,
                    processor,
                    shuffle_image_path,
                    f"{final_json_prompt}\n\nHere is the structured evidence trace for this image:\n{shuffle_trace_for_final}\n\nUse the trace to synthesize the final structured decision JSON.",
                    args.max_new_tokens_json,
                )
                shuffle_json, _ = safe_json_loads(shuffle_final)
                shuffle_probe["samples"] += 1
                shuffle_probe["overall_accuracy"] += int(
                    shuffle_json.get("overall_likelihood") == row["final_json_target"]["overall_likelihood"]
                ) if shuffle_json else 0

                oracle_final = generate_text(
                    model,
                    processor,
                    image_path,
                    (
                        f"{final_json_prompt}\n\n"
                        "Here is the structured evidence trace for this image:\n"
                        f"{compact_json_dumps(compact_trace_payload(row['evidence_trace_target']))}\n\n"
                        "Use the trace to synthesize the final structured decision JSON."
                    ),
                    args.max_new_tokens_json,
                )
                oracle_json, _ = safe_json_loads(oracle_final)
                oracle_probe["samples"] += 1
                oracle_probe["student_overall_accuracy"] += int(
                    pred_json.get("overall_likelihood") == row["final_json_target"]["overall_likelihood"]
                ) if pred_json else 0
                oracle_probe["oracle_overall_accuracy"] += int(
                    oracle_json.get("overall_likelihood") == row["final_json_target"]["overall_likelihood"]
                ) if oracle_json else 0

    report = finalize_metrics(counts)
    report["real_false_positive_rate"] = (
        counts["real_false_positive_samples"] / counts["real_samples"] if counts["real_samples"] else 0.0
    )
    report["blank_probe"] = {
        "json_parse_rate": blank_probe["parse_ok"] / blank_probe["samples"] if blank_probe["samples"] else 0.0,
        "overall_accuracy": blank_probe["overall_accuracy"] / blank_probe["samples"] if blank_probe["samples"] else 0.0,
    }
    report["shuffle_probe"] = {
        "overall_accuracy": shuffle_probe["overall_accuracy"] / shuffle_probe["samples"] if shuffle_probe["samples"] else 0.0
    }
    report["oracle_vs_student_trace_probe"] = {
        "student_overall_accuracy": (
            oracle_probe["student_overall_accuracy"] / oracle_probe["samples"] if oracle_probe["samples"] else 0.0
        ),
        "oracle_overall_accuracy": (
            oracle_probe["oracle_overall_accuracy"] / oracle_probe["samples"] if oracle_probe["samples"] else 0.0
        ),
    }

    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        render_html_report(output_path.stem, report, output_path.with_suffix(".html"))
    print(output)


if __name__ == "__main__":
    main()
