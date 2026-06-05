#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(REPO_ROOT / "student" / "src"))

from task_utils import (
    consistency_target_from_trace,
    evidence_trace_from_step2,
    format_final_prediction_json,
    json_dumps,
    quality_flags_from_trace,
    taxonomy_target_from_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic multitask supervision from existing Holmes-derived teacher labels.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=REPO_ROOT / "teacher" / "stage1_g31b_v5_full_balanced" / "holmes_lpcvc_sft.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "teacher" / "derived_deterministic_v1",
    )
    return parser.parse_args()


def iter_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve_teacher_step2(row: Dict[str, object]) -> Dict[str, object]:
    step2_internal = row.get("step2_internal")
    if isinstance(step2_internal, dict) and step2_internal.get("per_criterion_draft"):
        return step2_internal
    step2_draft = row.get("step2_draft")
    if isinstance(step2_draft, dict) and step2_draft.get("per_criterion_draft"):
        return step2_draft
    return {}


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    derived_path = output_root / "derived.jsonl"
    manifest_path = output_root / "manifest.json"

    total_rows = 0
    total_flags = 0
    label_counts: Dict[str, int] = {}
    example_final = {}
    example_trace = {}

    with derived_path.open("w", encoding="utf-8") as out_handle:
        for row_id, row in enumerate(iter_rows(args.input_jsonl)):
            step2_source = resolve_teacher_step2(row)
            final_json_target = format_final_prediction_json(step2_source)
            evidence_trace_target = evidence_trace_from_step2(step2_source)
            taxonomy_target = taxonomy_target_from_trace(evidence_trace_target)
            consistency_target = consistency_target_from_trace(evidence_trace_target)
            quality_flags = quality_flags_from_trace(evidence_trace_target)
            overall = final_json_target["overall_likelihood"]
            label_counts[overall] = label_counts.get(overall, 0) + 1

            derived_row = {
                "row_id": row_id,
                "image": row["image"],
                "image_root": os.fspath(args.input_jsonl.parent),
                "source": row.get("source", "holmes_sft"),
                "original_query": row.get("original_query", ""),
                "original_response": row.get("original_response", ""),
                "step1_target": row.get("step1_target", ""),
                "final_json_target": final_json_target,
                "evidence_trace_target": evidence_trace_target,
                "taxonomy_target": taxonomy_target,
                "consistency_target": consistency_target,
                "quality_flags": quality_flags,
            }
            out_handle.write(json.dumps(derived_row, ensure_ascii=False) + "\n")
            total_rows += 1
            total_flags += len(quality_flags)
            if not example_final:
                example_final = final_json_target
                example_trace = evidence_trace_target

    manifest = {
        "input_jsonl": os.fspath(args.input_jsonl),
        "output_jsonl": os.fspath(derived_path),
        "rows": total_rows,
        "label_counts": label_counts,
        "quality_flag_count": total_flags,
        "fields": [
            "row_id",
            "image",
            "image_root",
            "source",
            "original_query",
            "original_response",
            "step1_target",
            "final_json_target",
            "evidence_trace_target",
            "taxonomy_target",
            "consistency_target",
            "quality_flags",
        ],
        "examples": {
            "final_json_target": json_dumps(example_final),
            "evidence_trace_target": json_dumps(example_trace),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
