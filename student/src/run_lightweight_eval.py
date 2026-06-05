import argparse
import html
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "teacher" / "derived_deterministic_v1" / "derived.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "student" / "outputs" / "gemma4_lightweight_eval"
DEFAULT_CKPT4000 = REPO_ROOT / "student" / "outputs" / "gemma4_e2b_round1_20260527" / "checkpoint-4000"
DEFAULT_CKPT6015 = REPO_ROOT / "student" / "outputs" / "gemma4_e2b_round1_20260527" / "checkpoint-6015"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the lightweight Gemma evaluation suite.")
    parser.add_argument("--base_model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--derived_data_path", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--prompt_dir", type=str, default=str(REPO_ROOT / "prompts"))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens_trace", type=int, default=1536)
    parser.add_argument("--max_new_tokens_json", type=int, default=1024)
    parser.add_argument("--probe_samples", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--gpu_devices", type=str, default="")
    parser.add_argument("--python_executable", type=str, default=default_python_executable())
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def default_python_executable():
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidate = Path(virtual_env) / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return sys.executable


def metric_cell(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_job_specs():
    return [
        {
            "name": "ckpt4000_two_stage",
            "adapter_path": str(DEFAULT_CKPT4000),
            "inference_mode": "two_stage",
        },
        {
            "name": "ckpt4000_single_stage",
            "adapter_path": str(DEFAULT_CKPT4000),
            "inference_mode": "single_stage",
        },
        {
            "name": "ckpt6015_two_stage",
            "adapter_path": str(DEFAULT_CKPT6015),
            "inference_mode": "two_stage",
        },
    ]


def sample_balanced_row_ids(dataset_path: Path, sample_size: int, seed: int):
    if sample_size <= 0 or sample_size % 2 != 0:
        raise ValueError("sample_size must be a positive even number")
    buckets = {"Real": [], "AI-Generated": []}
    for row in iter_rows(dataset_path):
        label = row["final_json_target"]["overall_likelihood"]
        if label in buckets:
            buckets[label].append(str(row["row_id"]))
    per_label = sample_size // 2
    rng = random.Random(seed)
    selected = []
    for label in ("Real", "AI-Generated"):
        row_ids = list(buckets[label])
        rng.shuffle(row_ids)
        if len(row_ids) < per_label:
            raise ValueError(f"not enough rows for label={label}: need {per_label}, have {len(row_ids)}")
        selected.extend(row_ids[:per_label])
    selected.sort()
    return selected


def write_eval_slice(output_root: Path, dataset_path: Path, sample_size: int, seed: int):
    output_root.mkdir(parents=True, exist_ok=True)
    row_ids = sample_balanced_row_ids(dataset_path, sample_size, seed)
    row_ids_path = output_root / "eval_slice_row_ids.txt"
    row_ids_path.write_text("\n".join(row_ids) + "\n", encoding="utf-8")
    manifest = {
        "derived_data_path": str(dataset_path),
        "sample_size": sample_size,
        "seed": seed,
        "row_ids_path": str(row_ids_path),
        "label_counts": {"Real": sample_size // 2, "AI-Generated": sample_size // 2},
    }
    (output_root / "eval_slice_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return row_ids_path, manifest


def build_eval_command(args, job: dict, row_ids_path: Path, output_root: Path):
    report_path = output_root / "eval_reports" / f"{job['name']}.json"
    predictions_path = output_root / "predictions" / f"{job['name']}.jsonl"
    command = [
        args.python_executable,
        str(REPO_ROOT / "student" / "src" / "evaluate.py"),
        "--base_model",
        args.base_model,
        "--adapter_path",
        job["adapter_path"],
        "--derived_data_path",
        args.derived_data_path,
        "--prompt_dir",
        args.prompt_dir,
        "--row_ids_path",
        str(row_ids_path),
        "--inference_mode",
        job["inference_mode"],
        "--probe_samples",
        str(args.probe_samples),
        "--max_new_tokens_trace",
        str(args.max_new_tokens_trace),
        "--max_new_tokens_json",
        str(args.max_new_tokens_json),
        "--predictions_path",
        str(predictions_path),
        "--output_path",
        str(report_path),
    ]
    if args.local_files_only:
        command.append("--local_files_only")
    return command, report_path, predictions_path


def launch_jobs(args, row_ids_path: Path, output_root: Path):
    (output_root / "eval_reports").mkdir(parents=True, exist_ok=True)
    (output_root / "predictions").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    gpu_devices = [item.strip() for item in args.gpu_devices.split(",") if item.strip()]
    max_parallel = max(1, min(args.jobs, len(gpu_devices) if gpu_devices else args.jobs))
    pending = build_job_specs()
    running = []
    results = []

    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            command, report_path, predictions_path = build_eval_command(args, job, row_ids_path, output_root)
            log_path = output_root / "logs" / f"{job['name']}.log"
            env = os.environ.copy()
            gpu = None
            if gpu_devices:
                gpu = gpu_devices[(len(results) + len(running)) % len(gpu_devices)]
                env["CUDA_VISIBLE_DEVICES"] = gpu
            log_handle = log_path.open("w", encoding="utf-8")
            start_time = time.time()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            running.append(
                {
                    "job": job,
                    "process": process,
                    "log_handle": log_handle,
                    "log_path": log_path,
                    "report_path": report_path,
                    "predictions_path": predictions_path,
                    "gpu": gpu,
                    "start_time": start_time,
                    "command": command,
                }
            )
        if not running:
            break
        time.sleep(2)
        still_running = []
        for item in running:
            returncode = item["process"].poll()
            if returncode is None:
                still_running.append(item)
                continue
            item["log_handle"].close()
            results.append(
                {
                    "name": item["job"]["name"],
                    "inference_mode": item["job"]["inference_mode"],
                    "adapter_path": item["job"]["adapter_path"],
                    "gpu": item["gpu"],
                    "command": item["command"],
                    "returncode": returncode,
                    "runtime_sec": time.time() - item["start_time"],
                    "log_path": str(item["log_path"]),
                    "report_path": str(item["report_path"]),
                    "predictions_path": str(item["predictions_path"]),
                    "status": "ok" if returncode == 0 else "failed",
                }
            )
        running = still_running
    return sorted(results, key=lambda item: item["name"])


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_summary_html(output_root: Path, job_results: list[dict]):
    metrics_by_job = {}
    for result in job_results:
        report_path = Path(result["report_path"])
        if result["returncode"] == 0 and report_path.exists():
            metrics_by_job[result["name"]] = load_json(report_path)

    rows = []
    for result in job_results:
        report = metrics_by_job.get(result["name"], {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(result['name'])}</td>"
            f"<td>{html.escape(result['status'])}</td>"
            f"<td>{metric_cell(report.get('json_parse_rate'))}</td>"
            f"<td>{metric_cell(report.get('trace_json_parse_rate'))}</td>"
            f"<td>{metric_cell(report.get('overall_accuracy'))}</td>"
            f"<td>{metric_cell(report.get('macro_f1'))}</td>"
            f"<td>{metric_cell(report.get('wall_time_sec'))}</td>"
            f"<td>{metric_cell(report.get('sec_per_sample'))}</td>"
            "</tr>"
        )

    def delta_row(label: str, baseline_name: str, candidate_name: str):
        baseline = metrics_by_job.get(baseline_name, {})
        candidate = metrics_by_job.get(candidate_name, {})
        if not baseline or not candidate:
            return (
                "<tr>"
                f"<td>{html.escape(label)}</td><td>{html.escape(baseline_name)}</td><td>{html.escape(candidate_name)}</td>"
                "<td colspan='4'>missing report</td></tr>"
            )
        pairs = [
            ("Accuracy", "overall_accuracy"),
            ("Macro F1", "macro_f1"),
            ("JSON Parse", "json_parse_rate"),
            ("Trace Parse", "trace_json_parse_rate"),
        ]
        cells = []
        for _, key in pairs:
            b = baseline.get(key)
            c = candidate.get(key)
            if b is None or c is None:
                cells.append("N/A")
            else:
                cells.append(f"{c - b:+.3f}")
        return (
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(baseline_name)}</td>"
            f"<td>{html.escape(candidate_name)}</td>"
            f"<td>{cells[0]}</td><td>{cells[1]}</td><td>{cells[2]}</td><td>{cells[3]}</td>"
            "</tr>"
        )

    delta_rows = [
        delta_row("Stage Ablation", "ckpt4000_two_stage", "ckpt4000_single_stage"),
        delta_row("Checkpoint Ablation", "ckpt4000_two_stage", "ckpt6015_two_stage"),
    ]

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Lightweight Eval Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Lightweight Eval Summary</h1>
  <h2>Job Metrics</h2>
  <table>
    <thead>
      <tr><th>Job</th><th>Status</th><th>JSON Parse</th><th>Trace Parse</th><th>Accuracy</th><th>Macro F1</th><th>Wall Time</th><th>Sec / Sample</th></tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Comparison Deltas</h2>
  <table>
    <thead>
      <tr><th>Comparison</th><th>Baseline</th><th>Candidate</th><th>Accuracy</th><th>Macro F1</th><th>JSON Parse</th><th>Trace Parse</th></tr>
    </thead>
    <tbody>{''.join(delta_rows)}</tbody>
  </table>
</body>
</html>
"""
    output_path = output_root / "eval_reports" / "comparison_summary.html"
    output_path.write_text(html_text, encoding="utf-8")


def choose_case_rows(predictions: list[dict]):
    selected = []
    used_row_ids = set()

    def add_first(predicate, case_label):
        for row in predictions:
            row_id = row.get("row_id")
            if row_id in used_row_ids:
                continue
            if predicate(row):
                row["case_label"] = case_label
                selected.append(row)
                used_row_ids.add(row_id)
                return

    add_first(
        lambda row: row.get("overall_correct") and row["gold_final_json"]["overall_likelihood"] == "Real",
        "Success: Real",
    )
    add_first(
        lambda row: row.get("overall_correct") and row["gold_final_json"]["overall_likelihood"] == "AI-Generated",
        "Success: AI-Generated",
    )
    failure_count = 0
    for row in predictions:
        row_id = row.get("row_id")
        if row_id in used_row_ids:
            continue
        if not row.get("overall_correct"):
            failure_count += 1
            row["case_label"] = f"Failure #{failure_count}"
            selected.append(row)
            used_row_ids.add(row_id)
        if len(selected) >= 4:
            break
    return selected[:4]


def render_case_analysis(output_root: Path, predictions_path: Path):
    predictions = load_jsonl(predictions_path)
    selected = choose_case_rows(predictions)
    json_path = output_root / "eval_reports" / "case_analysis.json"
    json_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    cards = []
    for row in selected:
        pred_final_json = json.dumps(row.get("final_json"), ensure_ascii=False, indent=2)
        gold_final_json = json.dumps(row["gold_final_json"], ensure_ascii=False, indent=2)
        pred_trace_json = json.dumps(row.get("evidence_trace_json"), ensure_ascii=False, indent=2)
        gold_trace_json = json.dumps(row["gold_evidence_trace"], ensure_ascii=False, indent=2)
        cards.append(
            "<section class='case'>"
            f"<h2>{html.escape(row.get('case_label', 'Case'))}</h2>"
            f"<p><strong>row_id:</strong> {html.escape(str(row.get('row_id')))}</p>"
            f"<p><strong>image:</strong> {html.escape(str(row.get('image_path')))}</p>"
            f"<p><strong>error_type:</strong> {html.escape(str(row.get('error_type')))}</p>"
            f"<img src=\"{html.escape(str(row.get('image_path')))}\" alt=\"case image\" />"
            "<h3>Predicted Final JSON</h3>"
            f"<pre>{html.escape(pred_final_json)}</pre>"
            "<h3>Gold Final JSON</h3>"
            f"<pre>{html.escape(gold_final_json)}</pre>"
            "<h3>Predicted Evidence Trace Text</h3>"
            f"<pre>{html.escape(str(row.get('evidence_trace_text')))}</pre>"
            "<h3>Predicted Evidence Trace JSON</h3>"
            f"<pre>{html.escape(pred_trace_json)}</pre>"
            "<h3>Gold Evidence Trace</h3>"
            f"<pre>{html.escape(gold_trace_json)}</pre>"
            "</section>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Case Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    .case {{ border: 1px solid #ddd; padding: 16px; margin-bottom: 24px; }}
    img {{ max-width: 360px; display: block; margin: 12px 0; }}
    pre {{ white-space: pre-wrap; background: #f7f7f7; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Case Analysis</h1>
  {''.join(cards)}
</body>
</html>
"""
    (output_root / "eval_reports" / "case_analysis.html").write_text(html_text, encoding="utf-8")


def main():
    args = parse_args()
    dataset_path = Path(args.derived_data_path)
    output_root = Path(args.output_root)
    row_ids_path, manifest = write_eval_slice(output_root, dataset_path, args.sample_size, args.seed)
    job_results = launch_jobs(args, row_ids_path, output_root)
    runtime_summary = {
        "manifest": manifest,
        "jobs": job_results,
    }
    (output_root / "runtime_summary.json").write_text(
        json.dumps(runtime_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    render_summary_html(output_root, job_results)

    baseline_predictions_path = output_root / "predictions" / "ckpt4000_two_stage.jsonl"
    if baseline_predictions_path.exists():
        render_case_analysis(output_root, baseline_predictions_path)


if __name__ == "__main__":
    main()
