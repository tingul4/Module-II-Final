import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Render an HTML comparison report from two eval JSON reports.")
    parser.add_argument("--baseline_json", type=str, required=True)
    parser.add_argument("--candidate_json", type=str, required=True)
    parser.add_argument("--baseline_label", type=str, default="Baseline")
    parser.add_argument("--candidate_label", type=str, default="Candidate")
    parser.add_argument("--title", type=str, default="Evaluation Comparison Report")
    parser.add_argument("--output_html", type=str, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
def metric_row(name: str, baseline: dict, candidate: dict, key: str) -> str:
    b = float(baseline.get(key, 0.0) or 0.0)
    c = float(candidate.get(key, 0.0) or 0.0)
    delta = c - b
    return (
        "<tr>"
        f"<td>{name}</td><td>{b:.3f}</td><td>{c:.3f}</td><td>{delta:+.3f}</td>"
        "</tr>"
    )


def main():
    args = parse_args()
    baseline = load_json(Path(args.baseline_json))
    candidate = load_json(Path(args.candidate_json))

    metric_rows = [
        metric_row("Final JSON Parse", baseline, candidate, "json_parse_rate"),
        metric_row("Trace JSON Parse", baseline, candidate, "trace_json_parse_rate"),
        metric_row("Overall Accuracy", baseline, candidate, "overall_accuracy"),
        metric_row("Macro F1", baseline, candidate, "macro_f1"),
        metric_row("Support Type Accuracy", baseline, candidate, "support_type_accuracy"),
        metric_row("Taxonomy Accuracy", baseline, candidate, "taxonomy_accuracy"),
        metric_row("Consistency Score", baseline, candidate, "consistency_score"),
        metric_row("Real False Positive Rate", baseline, candidate, "real_false_positive_rate"),
    ]

    criterion_rows = []
    criteria = set(baseline.get("per_criterion_f1", {}).keys()) | set(candidate.get("per_criterion_f1", {}).keys())
    for criterion in sorted(criteria):
        b = baseline.get("per_criterion_f1", {}).get(criterion, {}).get("f1", 0.0)
        c = candidate.get("per_criterion_f1", {}).get(criterion, {}).get("f1", 0.0)
        criterion_rows.append(
            "<tr>"
            f"<td>{criterion}</td><td>{b:.3f}</td><td>{c:.3f}</td><td>{c - b:+.3f}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{args.title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .meta {{ margin-bottom: 20px; }}
  </style>
</head>
<body>
  <h1>{args.title}</h1>
  <div class="meta">
    <div><strong>{args.baseline_label}</strong>: <code>{args.baseline_json}</code></div>
    <div><strong>{args.candidate_label}</strong>: <code>{args.candidate_json}</code></div>
  </div>
  <h2>Headline Metrics</h2>
  <table>
    <thead>
      <tr><th>Metric</th><th>{args.baseline_label}</th><th>{args.candidate_label}</th><th>Delta</th></tr>
    </thead>
    <tbody>{''.join(metric_rows)}</tbody>
  </table>
  <h2>Per-Criterion F1</h2>
  <table>
    <thead>
      <tr><th>Criterion</th><th>{args.baseline_label}</th><th>{args.candidate_label}</th><th>Delta</th></tr>
    </thead>
    <tbody>{''.join(criterion_rows)}</tbody>
  </table>
</body>
</html>
"""
    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
