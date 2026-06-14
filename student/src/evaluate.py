import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

from detectors.holmes_clip_lora.metrics import macro_f1_from_binary_labels
from detectors.holmes_clip_lora.runtime import (
    DEFAULT_THRESHOLD,
    default_checkpoint_path,
    load_detector,
    prediction_payload,
    score_single_image,
)
from utils.eval_utils import (
    compute_explanatory_metrics,
    ensure_split_manifest,
    format_metric_value,
    load_rows_for_split,
    select_balanced_rows,
    render_origin_response_surface,
    load_teacher_predictions,
    render_explanation_surface,
    row_ids_for_split,
)
from utils.task_utils import (
    CRITERIA,
    canonicalize_criterion,
    compact_json_dumps,
    compact_trace_payload,
    normalize_final_prediction_payload,
    safe_json_loads,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an image-authenticity student model.")
    parser.add_argument("--base_model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--derived_data_path", type=str, required=True)
    parser.add_argument("--prompt_dir", type=str, default=str(REPO_ROOT / "prompts"))
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--probe_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens_trace", type=int, default=1536)
    parser.add_argument("--max_new_tokens_json", type=int, default=1024)
    parser.add_argument("--inference_mode", choices=("two_stage", "single_stage"), default="two_stage")
    parser.add_argument("--split", choices=("train", "eval", "all"), default="eval")
    parser.add_argument("--split_manifest_path", type=str, default=None)
    parser.add_argument(
        "--prediction_source",
        choices=("student", "teacher", "teacher_origin", "detector", "detector_student"),
        default="student",
    )
    parser.add_argument("--teacher_reference", choices=("derived_target", "origin_response"), default="derived_target")
    parser.add_argument(
        "--teacher_jsonl_path",
        type=str,
        default=str(REPO_ROOT / "teacher" / "stage1_g31b_v5_full_balanced" / "holmes_lpcvc_sft.jsonl"),
    )
    parser.add_argument("--detector_checkpoint_path", type=str, default=default_checkpoint_path())
    parser.add_argument("--detector_clip_weights", type=str, default=None)
    parser.add_argument("--detector_threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--detector_device", type=str, default=None)
    parser.add_argument("--enable_cider", action="store_true")
    parser.add_argument("--row_ids_path", type=str, default=None)
    parser.add_argument("--eval_slice_count", type=int, default=0)
    parser.add_argument("--eval_slice_seed", type=int, default=42)
    parser.add_argument("--balanced_max_samples", type=int, default=0)
    parser.add_argument("--progress_log_every", type=int, default=10)
    parser.add_argument("--predictions_path", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def render_markdown_report(title: str, report: dict, output_path: Path):
    metric_specs = [
        ("Prediction Source", report.get("prediction_source")),
        ("Inference Mode", report.get("inference_mode")),
        ("Split", report.get("split")),
        ("Eval Slice Count", report.get("eval_slice_count")),
        ("Eval Slice Seed", report.get("eval_slice_seed")),
        ("Teacher Reference", report.get("teacher_reference")),
        ("Sample Count", report.get("sample_count")),
        ("Wall Time (sec)", report.get("wall_time_sec")),
        ("Sec / Sample", report.get("sec_per_sample")),
        ("Detector Threshold", report.get("detector_threshold")),
    ]
    classification_specs = [
        ("Final JSON Parse", report.get("json_parse_rate")),
        ("Trace JSON Parse", report.get("trace_json_parse_rate")),
        ("Overall Accuracy", report.get("overall_accuracy")),
        ("Overall Macro F1", report.get("overall_macro_f1")),
        ("Criterion Macro F1", report.get("criterion_macro_f1")),
        ("Average Precision", report.get("average_precision")),
        ("Real False Positive Rate", report.get("real_false_positive_rate")),
        ("Support Type Accuracy", report.get("support_type_accuracy")),
        ("Taxonomy Accuracy", report.get("taxonomy_accuracy")),
        ("Consistency Score", report.get("consistency_score")),
    ]
    explanation_specs = [
        ("BLEU-1", report.get("bleu_1")),
        ("ROUGE-L", report.get("rouge_l")),
        ("METEOR", report.get("meteor")),
        ("CIDEr", report.get("cider")),
        ("Blank Probe Parse", (report.get("blank_probe") or {}).get("json_parse_rate")),
    ]
    lines = [
        f"# {title}",
        "",
        "## Run Context",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for label, value in metric_specs:
        lines.append(f"| {label} | {format_metric_value(value)} |")
    lines.extend(
        [
            "",
            "## Classification Metrics",
            "",
            "| Metric | Value |",
            "| --- | --- |",
        ]
    )
    for label, value in classification_specs:
        lines.append(f"| {label} | {format_metric_value(value)} |")
    lines.extend(
        [
            "",
            "## Explanation Metrics",
            "",
            "| Metric | Value |",
            "| --- | --- |",
        ]
    )
    for label, value in explanation_specs:
        lines.append(f"| {label} | {format_metric_value(value)} |")

    per_criterion = report.get("per_criterion_f1")
    if isinstance(per_criterion, dict):
        lines.extend(
            [
                "",
                "## Per-Criterion F1",
                "",
                "| Criterion | Precision | Recall | F1 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for criterion, metrics in per_criterion.items():
            lines.append(
                f"| {criterion} | {format_metric_value(metrics.get('precision'))} | "
                f"{format_metric_value(metrics.get('recall'))} | {format_metric_value(metrics.get('f1'))} |"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_selected_row_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"row_ids file not found: {file_path}")
    if file_path.suffix.lower() == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("row_ids json must be a list")
        return {str(item) for item in payload}
    return {line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_eval_rows(
    dataset_path: Path,
    max_samples: int,
    row_ids_path: str | None,
    split_row_ids: Optional[set[int]],
    eval_slice_count: int = 0,
    eval_slice_seed: int = 42,
):
    selected_row_ids = load_selected_row_ids(row_ids_path)
    rows = load_rows_for_split(
        dataset_path,
        split_row_ids=split_row_ids,
        selected_row_ids=selected_row_ids or None,
        max_samples=0 if eval_slice_count > 0 else max_samples,
    )
    if eval_slice_count > 0:
        return select_balanced_rows(rows, eval_slice_count, seed=eval_slice_seed)
    return rows


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
        criterion = canonicalize_criterion(entry.get("criterion"))
        if criterion in CRITERIA:
            normalized[criterion] = entry
    return normalized


def evaluate_prediction(pred: dict, gold: dict, counts: dict, *, parsed_ok: bool = True):
    counts["samples"] += 1
    counts["json_parse_ok"] += int(parsed_ok)
    gold_label = 1 if gold.get("overall_likelihood") == "AI-Generated" else 0
    counts["overall_gold_labels"].append(gold_label)
    if gold.get("overall_likelihood") == "Real":
        counts["real_samples"] += 1
    if not pred:
        return
    pred_label = 1 if pred.get("overall_likelihood") == "AI-Generated" else 0
    counts["overall_pred_labels"].append(pred_label)
    counts["overall_correct"] += int(pred.get("overall_likelihood") == gold.get("overall_likelihood"))
    pred_entries = normalize_criterion_entries(pred)
    if gold.get("overall_likelihood") == "Real" and pred.get("overall_likelihood") == "AI-Generated":
        counts["real_false_positive_samples"] += 1
    for gold_entry in gold.get("per_criterion", []):
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
        counts["taxonomy_correct"] += int(pred_entry.get("artifact_taxonomy") == gold_entry.get("artifact_taxonomy"))
        counts["consistency_total"] += 1
        pred_consistent = int(pred_entry.get("score", 0) or 0) == 0 or (
            pred_entry.get("evidence") not in {"", None} and pred_entry.get("support_type") != "unsupported"
        )
        gold_consistent = not bool(gold_entry.get("artifact_score_conflict"))
        counts["consistency_correct"] += int(pred_consistent == gold_consistent)


def finalize_student_metrics(counts: dict, inference_mode: str):
    per_criterion_f1 = {}
    criterion_macro_f1_values = []
    for criterion in CRITERIA:
        tp = counts["tp"][criterion]
        fp = counts["fp"][criterion]
        fn = counts["fn"][criterion]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_criterion_f1[criterion] = {"precision": precision, "recall": recall, "f1": f1}
        criterion_macro_f1_values.append(f1)
    trace_parse_rate = None
    if inference_mode == "two_stage":
        trace_parse_rate = counts["trace_json_parse_ok"] / counts["samples"] if counts["samples"] else 0.0
    return {
        "json_parse_rate": counts["json_parse_ok"] / counts["samples"] if counts["samples"] else 0.0,
        "trace_json_parse_rate": trace_parse_rate,
        "overall_accuracy": counts["overall_correct"] / counts["samples"] if counts["samples"] else 0.0,
        "overall_macro_f1": (
            macro_f1_from_binary_labels(counts["overall_gold_labels"], counts["overall_pred_labels"])
            if counts["overall_gold_labels"] and len(counts["overall_gold_labels"]) == len(counts["overall_pred_labels"])
            else 0.0
        ),
        "criterion_macro_f1": (
            sum(criterion_macro_f1_values) / len(criterion_macro_f1_values) if criterion_macro_f1_values else 0.0
        ),
        "average_precision": None,
        "per_criterion_f1": per_criterion_f1,
        "support_type_accuracy": counts["support_correct"] / counts["support_total"] if counts["support_total"] else 0.0,
        "taxonomy_accuracy": counts["taxonomy_correct"] / counts["taxonomy_total"] if counts["taxonomy_total"] else 0.0,
        "consistency_score": counts["consistency_correct"] / counts["consistency_total"] if counts["consistency_total"] else 0.0,
        "real_false_positive_rate": (
            counts["real_false_positive_samples"] / counts["real_samples"] if counts["real_samples"] else 0.0
        ),
    }


def build_final_prompt(final_json_prompt: str, trace_for_final: str):
    return (
        f"{final_json_prompt}\n\n"
        "Here is the structured evidence trace for this image:\n"
        f"{trace_for_final}\n\n"
        "Use the trace to synthesize the final structured decision JSON."
    )


def record_prediction(predictions_handle, payload: dict):
    if predictions_handle is None:
        return
    predictions_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    predictions_handle.flush()


def overlay_detector_label(student_final_json: dict, detector_meta: dict) -> dict:
    normalized = normalize_final_prediction_payload(student_final_json)
    payload = dict(normalized)
    payload["student_overall_likelihood"] = normalized.get("overall_likelihood")
    payload["overall_likelihood"] = detector_meta["detector_label"]
    payload["detector_score"] = detector_meta["detector_score"]
    payload["detector_threshold"] = detector_meta["detector_threshold"]
    payload["detector_label"] = detector_meta["detector_label"]
    return payload


def init_student_counts() -> dict:
    return {
        "samples": 0,
        "json_parse_ok": 0,
        "trace_json_parse_ok": 0,
        "overall_correct": 0,
        "overall_gold_labels": [],
        "overall_pred_labels": [],
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


def format_eta_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _teacher_reference_for_args(args) -> str:
    if args.prediction_source == "teacher_origin":
        return "origin_response"
    return args.teacher_reference


def _student_bundle_from_args(args):
    from inference import load_model, load_prompts

    processor, model = load_model(args.base_model, args.adapter_path, local_files_only=args.local_files_only)
    evidence_trace_prompt, final_json_prompt = load_prompts(args.prompt_dir)
    return {
        "processor": processor,
        "model": model,
        "evidence_trace_prompt": evidence_trace_prompt,
        "final_json_prompt": final_json_prompt,
    }


def run_evaluation(args, student_bundle: Optional[dict] = None, progress_callback=None) -> dict:
    dataset_path = Path(args.derived_data_path)
    split_manifest_path, split_manifest = ensure_split_manifest(
        dataset_path,
        manifest_path=Path(args.split_manifest_path) if args.split_manifest_path else None,
    )
    split_row_ids = row_ids_for_split(split_manifest, args.split)
    rows = load_eval_rows(
        dataset_path,
        args.max_samples,
        args.row_ids_path,
        split_row_ids,
        eval_slice_count=max(int(getattr(args, "eval_slice_count", 0)), int(getattr(args, "balanced_max_samples", 0))),
        eval_slice_seed=int(getattr(args, "eval_slice_seed", 42)),
    )

    processor = None
    model = None
    evidence_trace_prompt = ""
    final_json_prompt = ""
    teacher_predictions = {}
    detector_bundle = None
    teacher_reference = _teacher_reference_for_args(args)

    if args.prediction_source in {"student", "detector_student"}:
        if student_bundle is None and not args.adapter_path:
            raise ValueError("--adapter_path is required when prediction_source uses the student model")
        if student_bundle is None:
            student_bundle = _student_bundle_from_args(args)
        from inference import generate_text, generate_trace_payload

        processor = student_bundle["processor"]
        model = student_bundle["model"]
        evidence_trace_prompt = student_bundle["evidence_trace_prompt"]
        final_json_prompt = student_bundle["final_json_prompt"]
    if args.prediction_source in {"teacher", "teacher_origin"}:
        teacher_predictions = load_teacher_predictions(
            Path(args.teacher_jsonl_path),
            {int(row["row_id"]) for row in rows},
        )
    if args.prediction_source in {"detector", "detector_student"}:
        detector_bundle = load_detector(
            args.detector_checkpoint_path,
            args.detector_clip_weights,
            threshold=args.detector_threshold,
            device=args.detector_device,
        )

    predictions_handle = None
    if args.predictions_path:
        predictions_path = Path(args.predictions_path)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_handle = predictions_path.open("w", encoding="utf-8")

    counts = init_student_counts()
    reference_texts = []
    hypothesis_texts = []
    detector_row_ids = []
    detector_gold = []
    detector_scores = []

    blank_probe = {"parse_ok": 0, "overall_accuracy": 0, "samples": 0}
    shuffle_probe = {"overall_accuracy": 0, "samples": 0}
    oracle_probe = {"student_overall_accuracy": 0, "oracle_overall_accuracy": 0, "samples": 0}

    start_time = time.time()
    total_rows = len(rows)
    progress_log_every = max(1, int(getattr(args, "progress_log_every", 10)))
    progress_desc = f"eval[{args.prediction_source}:{args.split}]"
    print(
        f"[eval] start | source={args.prediction_source} | split={args.split} | "
        f"samples={total_rows} | detector_device={getattr(args, 'detector_device', None)} | "
        f"eval_slice_count={max(int(getattr(args, 'eval_slice_count', 0)), int(getattr(args, 'balanced_max_samples', 0)))} | "
        f"eval_slice_seed={int(getattr(args, 'eval_slice_seed', 42))}"
    )
    progress_bar = tqdm(total=total_rows, desc=progress_desc, leave=True) if total_rows else None

    def emit_progress(done: int):
        elapsed = time.time() - start_time
        sec_per_sample = elapsed / done if done else None
        eta_sec = sec_per_sample * (total_rows - done) if sec_per_sample is not None else None
        payload = {
            "done": done,
            "total": total_rows,
            "elapsed_sec": elapsed,
            "sec_per_sample": sec_per_sample,
            "eta_sec": eta_sec,
            "json_parse_rate": counts["json_parse_ok"] / counts["samples"] if counts["samples"] else 0.0,
            "trace_json_parse_rate": counts["trace_json_parse_ok"] / counts["samples"] if counts["samples"] else 0.0,
            "overall_accuracy": counts["overall_correct"] / counts["samples"] if counts["samples"] else 0.0,
        }
        if progress_bar is not None:
            progress_bar.set_postfix(
                acc=f"{payload['overall_accuracy']:.3f}",
                json=f"{payload['json_parse_rate']:.3f}",
                trace=f"{payload['trace_json_parse_rate']:.3f}",
                sec=f"{payload['sec_per_sample']:.1f}" if payload["sec_per_sample"] is not None else "n/a",
                eta=format_eta_seconds(payload["eta_sec"]),
            )
        if progress_callback is not None:
            progress_callback(payload)
        else:
            sec_per_sample_text = f"{payload['sec_per_sample']:.1f}" if payload["sec_per_sample"] is not None else "n/a"
            print(
                f"[eval_progress] done={done}/{total_rows} | "
                f"acc={payload['overall_accuracy']:.3f} | "
                f"json_parse={payload['json_parse_rate']:.3f} | "
                f"trace_parse={payload['trace_json_parse_rate']:.3f} | "
                f"sec_per_sample={sec_per_sample_text} | "
                f"eta={format_eta_seconds(payload['eta_sec'])}"
            )

    blank_image = Image.new("RGB", (512, 512), color=(0, 0, 0))
    try:
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            blank_image.save(handle.name)
            shuffled_rows = rows[1:] + rows[:1]
            for idx, row in enumerate(rows):
                row_id = int(row.get("row_id"))
                image_path = resolve_image_path(dataset_path, row)
                gold_final_json = row["final_json_target"]
                gold_label = 1 if gold_final_json["overall_likelihood"] == "AI-Generated" else 0
                trace_text = ""
                trace_json = {}
                trace_parse_error = ""
                trace_retry_used = False
                final_text = ""
                pred_json = {}
                parse_error = ""
                student_final_json = None
                detector_meta = None

                if args.prediction_source in {"teacher", "teacher_origin"}:
                    teacher_payload = teacher_predictions[row_id]
                    pred_json = teacher_payload["final_json"]
                    trace_json = teacher_payload["evidence_trace"]
                    final_text = json.dumps(pred_json, ensure_ascii=False)
                    trace_text = json.dumps(trace_json, ensure_ascii=False)
                    evaluate_trace_prediction(trace_json, row["evidence_trace_target"], counts)
                    evaluate_prediction(pred_json, gold_final_json, counts)
                    if teacher_reference == "origin_response":
                        reference_texts.append(
                            render_origin_response_surface(
                                str(row.get("original_response", "")),
                                gold_final_json["overall_likelihood"],
                            )
                        )
                    else:
                        reference_texts.append(render_explanation_surface(gold_final_json))
                    hypothesis_texts.append(render_explanation_surface(pred_json))
                elif args.prediction_source == "detector":
                    score = score_single_image(detector_bundle, image_path)
                    detector_meta = prediction_payload(score, detector_bundle.threshold)
                    pred_json = {"overall_likelihood": detector_meta["detector_label"]}
                    detector_row_ids.append(row_id)
                    detector_gold.append(gold_label)
                    detector_scores.append(score)
                else:
                    if args.inference_mode == "two_stage":
                        trace_text, trace_json, trace_parse_error, trace_for_final, trace_retry_used = generate_trace_payload(
                            model,
                            processor,
                            image_path,
                            evidence_trace_prompt,
                            args.max_new_tokens_trace,
                        )
                        final_text = generate_text(
                            model,
                            processor,
                            image_path,
                            build_final_prompt(final_json_prompt, trace_for_final),
                            args.max_new_tokens_json,
                        )
                        pred_json, parse_error = safe_json_loads(final_text)
                        evaluate_trace_prediction(trace_json, row["evidence_trace_target"], counts)
                    else:
                        final_text = generate_text(
                            model,
                            processor,
                            image_path,
                            final_json_prompt,
                            args.max_new_tokens_json,
                        )
                        pred_json, parse_error = safe_json_loads(final_text)

                    if pred_json:
                        student_final_json = normalize_final_prediction_payload(pred_json)
                        pred_json = student_final_json
                    if args.prediction_source == "detector_student":
                        score = score_single_image(detector_bundle, image_path)
                        detector_meta = prediction_payload(score, detector_bundle.threshold)
                        detector_row_ids.append(row_id)
                        detector_gold.append(gold_label)
                        detector_scores.append(score)
                        if pred_json:
                            pred_json = overlay_detector_label(student_final_json, detector_meta)
                        else:
                            pred_json = {"overall_likelihood": detector_meta["detector_label"], "per_criterion": []}

                    evaluate_prediction(pred_json, gold_final_json, counts, parsed_ok=bool(student_final_json))
                    if student_final_json:
                        reference_texts.append(render_explanation_surface(gold_final_json))
                        hypothesis_texts.append(render_explanation_surface(student_final_json))
                    else:
                        reference_texts.append(render_explanation_surface(gold_final_json))
                        hypothesis_texts.append("")

                    if (
                        args.prediction_source == "student"
                        and args.inference_mode == "two_stage"
                        and idx < args.probe_samples
                    ):
                        _, _, _, blank_trace_for_final, _ = generate_trace_payload(
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
                            build_final_prompt(final_json_prompt, blank_trace_for_final),
                            args.max_new_tokens_json,
                        )
                        blank_json, _ = safe_json_loads(blank_final)
                        blank_probe["samples"] += 1
                        blank_probe["parse_ok"] += int(bool(blank_json))
                        blank_probe["overall_accuracy"] += int(
                            blank_json.get("overall_likelihood") == gold_final_json["overall_likelihood"]
                        ) if blank_json else 0

                        shuffle_row = shuffled_rows[idx]
                        shuffle_image_path = resolve_image_path(dataset_path, shuffle_row)
                        _, _, _, shuffle_trace_for_final, _ = generate_trace_payload(
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
                            build_final_prompt(final_json_prompt, shuffle_trace_for_final),
                            args.max_new_tokens_json,
                        )
                        shuffle_json, _ = safe_json_loads(shuffle_final)
                        shuffle_probe["samples"] += 1
                        shuffle_probe["overall_accuracy"] += int(
                            shuffle_json.get("overall_likelihood") == gold_final_json["overall_likelihood"]
                        ) if shuffle_json else 0

                        oracle_final = generate_text(
                            model,
                            processor,
                            image_path,
                            build_final_prompt(
                                final_json_prompt,
                                compact_json_dumps(compact_trace_payload(row["evidence_trace_target"])),
                            ),
                            args.max_new_tokens_json,
                        )
                        oracle_json, _ = safe_json_loads(oracle_final)
                        oracle_probe["samples"] += 1
                        oracle_probe["student_overall_accuracy"] += int(
                            pred_json.get("overall_likelihood") == gold_final_json["overall_likelihood"]
                        ) if pred_json else 0
                        oracle_probe["oracle_overall_accuracy"] += int(
                            oracle_json.get("overall_likelihood") == gold_final_json["overall_likelihood"]
                        ) if oracle_json else 0

                overall_correct = bool(pred_json and pred_json.get("overall_likelihood") == gold_final_json["overall_likelihood"])
                error_type = "ok"
                if args.prediction_source == "detector":
                    if not overall_correct:
                        error_type = "overall_mismatch"
                elif not student_final_json and args.prediction_source == "detector_student":
                    error_type = "final_json_parse_error"
                elif not pred_json:
                    error_type = "final_json_parse_error"
                elif args.inference_mode == "two_stage" and args.prediction_source in {"student", "detector_student"} and not trace_json:
                    error_type = "trace_json_parse_error"
                elif not overall_correct:
                    error_type = "overall_mismatch"

                record_prediction(
                    predictions_handle,
                    {
                        "row_id": row_id,
                        "image_path": image_path,
                        "prediction_source": args.prediction_source,
                        "inference_mode": args.inference_mode if args.prediction_source != "detector" else "detector_only",
                        "overall_correct": overall_correct,
                        "error_type": error_type,
                        "final_json_parse_ok": (
                            bool(student_final_json) if args.prediction_source in {"student", "detector_student"} else None
                        ),
                        "trace_parse_ok": (
                            bool(trace_json) if args.prediction_source != "detector" and args.inference_mode == "two_stage" else None
                        ),
                        "final_json_text": final_text or None,
                        "student_final_json": student_final_json,
                        "final_json": pred_json or None,
                        "parse_error": parse_error or None,
                        "evidence_trace_text": trace_text or None,
                        "evidence_trace_json": trace_json or None,
                        "evidence_trace_parse_error": trace_parse_error or None,
                        "evidence_trace_retry_used": trace_retry_used,
                        "detector": detector_meta,
                        "gold_final_json": gold_final_json,
                        "gold_evidence_trace": row["evidence_trace_target"],
                    },
                )
                if progress_bar is not None:
                    progress_bar.update(1)
                done = idx + 1
                if done == total_rows or done % progress_log_every == 0:
                    emit_progress(done)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if predictions_handle is not None:
        predictions_handle.close()

    wall_time_sec = time.time() - start_time
    if args.prediction_source == "detector":
        from detectors.holmes_clip_lora.runtime import evaluate_scores

        detector_report = evaluate_scores(detector_row_ids, detector_gold, detector_scores, detector_bundle.threshold)
        report = {
            "json_parse_rate": None,
            "trace_json_parse_rate": None,
            "overall_accuracy": detector_report["accuracy"],
            "overall_macro_f1": detector_report["macro_f1"],
            "criterion_macro_f1": None,
            "average_precision": detector_report["average_precision"],
            "per_criterion_f1": None,
            "support_type_accuracy": None,
            "taxonomy_accuracy": None,
            "consistency_score": None,
            "bleu_1": None,
            "rouge_l": None,
            "meteor": None,
            "cider": None,
            "prediction_source": args.prediction_source,
            "inference_mode": "detector_only",
            "split": args.split,
            "split_manifest_path": str(split_manifest_path),
            "sample_count": len(detector_row_ids),
            "wall_time_sec": wall_time_sec,
            "sec_per_sample": wall_time_sec / len(detector_row_ids) if detector_row_ids else None,
            "real_false_positive_rate": 1.0 - detector_report["real_recall"],
            "predictions_path": args.predictions_path,
            "blank_probe": None,
            "shuffle_probe": None,
            "oracle_vs_student_trace_probe": None,
            "detector_threshold": detector_bundle.threshold,
            "detector_checkpoint_path": args.detector_checkpoint_path,
            "teacher_reference": None,
            "adapter_path": None,
        }
    else:
        metrics_mode = "two_stage" if args.prediction_source in {"teacher", "teacher_origin"} else args.inference_mode
        report = finalize_student_metrics(counts, metrics_mode)
        report.update(compute_explanatory_metrics(reference_texts, hypothesis_texts, enable_cider=args.enable_cider))
        report["prediction_source"] = args.prediction_source
        report["inference_mode"] = (
            "teacher_baseline" if args.prediction_source in {"teacher", "teacher_origin"} else args.inference_mode
        )
        report["split"] = args.split
        report["split_manifest_path"] = str(split_manifest_path)
        report["sample_count"] = counts["samples"]
        report["wall_time_sec"] = wall_time_sec
        report["sec_per_sample"] = wall_time_sec / counts["samples"] if counts["samples"] else None
        report["predictions_path"] = args.predictions_path
        report["detector_threshold"] = detector_bundle.threshold if detector_bundle is not None else None
        report["detector_checkpoint_path"] = args.detector_checkpoint_path if detector_bundle is not None else None
        report["teacher_reference"] = teacher_reference if args.prediction_source in {"teacher", "teacher_origin"} else None
        report["adapter_path"] = args.adapter_path if args.prediction_source in {"student", "detector_student"} else None
        report["eval_slice_count"] = max(int(getattr(args, "eval_slice_count", 0)), int(getattr(args, "balanced_max_samples", 0)))
        report["eval_slice_seed"] = int(getattr(args, "eval_slice_seed", 42))
        if args.prediction_source == "student" and args.inference_mode == "two_stage":
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
        else:
            report["blank_probe"] = None
            report["shuffle_probe"] = None
            report["oracle_vs_student_trace_probe"] = None

    if args.prediction_source in {"teacher", "teacher_origin"} and "teacher_reference" not in report:
        report["teacher_reference"] = teacher_reference
    if args.prediction_source in {"student", "detector_student"} and "adapter_path" not in report:
        report["adapter_path"] = args.adapter_path
    if "eval_slice_count" not in report:
        report["eval_slice_count"] = max(int(getattr(args, "eval_slice_count", 0)), int(getattr(args, "balanced_max_samples", 0)))
        report["eval_slice_seed"] = int(getattr(args, "eval_slice_seed", 42))

    print(
        f"[eval] completed | source={args.prediction_source} | split={args.split} | "
        f"samples={report.get('sample_count')} | acc={format_metric_value(report.get('overall_accuracy'))} | "
        f"overall_macro_f1={format_metric_value(report.get('overall_macro_f1'))} | "
        f"criterion_macro_f1={format_metric_value(report.get('criterion_macro_f1'))} | "
        f"bleu_1={format_metric_value(report.get('bleu_1'))} | "
        f"rouge_l={format_metric_value(report.get('rouge_l'))} | "
        f"meteor={format_metric_value(report.get('meteor'))}"
    )
    return report


def main():
    args = parse_args()
    report = run_evaluation(args)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        render_markdown_report(output_path.stem, report, output_path.with_suffix(".md"))
    print(output)


if __name__ == "__main__":
    main()
