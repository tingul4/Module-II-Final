import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from utils.task_utils import (
    CRITERIA,
    DEFAULT_EVIDENCE,
    clean_text,
    criterion_lookup_key,
    evidence_trace_from_step2,
    format_final_prediction_json,
    normalize_final_prediction_payload,
)


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_HOLMES_SECTION_PATTERN = re.compile(r"\n\s*\d+\.\s+\*\*(.*?)\*\*:\s*(.+?)(?=\n\s*\d+\.\s+\*\*|\Z)", re.S)
_NEGATIVE_RESPONSE_MARKERS = (
    "not assessable",
    "does not apply",
    "not applicable",
    "n/a",
    "no discernible",
    "no visible",
    "no signs",
    "no text",
    "no symbols",
    "no faces",
    "no face",
    "no human",
    "natural",
    "realistic",
    "consistent",
    "well-defined",
)
_CHECKPOINT_SORT_FIELDS = (
    "overall_accuracy",
    "overall_macro_f1",
    "json_parse_rate",
    "rouge_l",
    "meteor",
)
_HOLMES_ALIAS_MAP = {
    "lighting": "Lighting & Shadows Consistency",
    "shadows": "Lighting & Shadows Consistency",
    "lighting shadows consistency": "Lighting & Shadows Consistency",
    "shadows lighting": "Lighting & Shadows Consistency",
    "edges": "Edges & Boundaries",
    "boundaries": "Edges & Boundaries",
    "edges boundaries": "Edges & Boundaries",
    "texture": "Texture & Resolution",
    "resolution": "Texture & Resolution",
    "texture resolution": "Texture & Resolution",
    "clarity": "Texture & Resolution",
    "perspective": "Perspective & Spatial Relationships",
    "perspective spatial relationships": "Perspective & Spatial Relationships",
    "distortion": "Perspective & Spatial Relationships",
    "physical laws": "Physical & Common Sense Logic",
    "physical common sense logic": "Physical & Common Sense Logic",
    "common sense": "Physical & Common Sense Logic",
    "text": "Text & Symbols",
    "symbols": "Text & Symbols",
    "text symbols": "Text & Symbols",
    "ocr": "Text & Symbols",
    "faces": "Human & Biological Structure Integrity",
    "body structure": "Human & Biological Structure Integrity",
    "human biological structure integrity": "Human & Biological Structure Integrity",
    "human structure": "Human & Biological Structure Integrity",
    "biological structure": "Human & Biological Structure Integrity",
    "material": "Material & Object Details",
    "object details": "Material & Object Details",
    "material object details": "Material & Object Details",
}


def iter_jsonl_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def default_split_manifest_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(f"{dataset_path.stem}_split.json")


def resolve_row_id(row: Dict[str, object], fallback: int) -> int:
    return int(row.get("row_id", fallback))


def summarize_dataset_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    row_ids = [resolve_row_id(row, idx) for idx, row in enumerate(rows)]
    label_counts = Counter(row["final_json_target"]["overall_likelihood"] for row in rows)
    return {
        "dataset_rows": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "row_id_min": min(row_ids) if row_ids else None,
        "row_id_max": max(row_ids) if row_ids else None,
        "row_id_checksum": sum(row_ids),
    }


def _eval_count(label_rows: List[int], eval_ratio: float) -> int:
    if not label_rows:
        return 0
    raw_count = int(round(len(label_rows) * float(eval_ratio)))
    return max(1, min(len(label_rows) - 1, raw_count)) if len(label_rows) > 1 else 1


def build_split_manifest(
    rows: Sequence[Dict[str, object]],
    dataset_path: Path,
    eval_ratio: float,
    seed: int,
) -> Dict[str, object]:
    buckets: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        row_id = resolve_row_id(row, idx)
        label = str(row["final_json_target"]["overall_likelihood"])
        buckets.setdefault(label, []).append(row_id)

    rng = random.Random(seed)
    train_row_ids: List[int] = []
    eval_row_ids: List[int] = []
    train_label_counts: Dict[str, int] = {}
    eval_label_counts: Dict[str, int] = {}

    for label in sorted(buckets):
        row_ids = list(buckets[label])
        rng.shuffle(row_ids)
        eval_count = _eval_count(row_ids, eval_ratio)
        eval_part = sorted(row_ids[:eval_count])
        train_part = sorted(row_ids[eval_count:])
        eval_row_ids.extend(eval_part)
        train_row_ids.extend(train_part)
        eval_label_counts[label] = len(eval_part)
        train_label_counts[label] = len(train_part)

    train_row_ids.sort()
    eval_row_ids.sort()

    source_summary = summarize_dataset_rows(rows)
    return {
        "split_manifest_version": 1,
        "dataset_path": str(dataset_path),
        "eval_ratio": float(eval_ratio),
        "seed": int(seed),
        "source_dataset": source_summary,
        "train_label_counts": train_label_counts,
        "eval_label_counts": eval_label_counts,
        "train_row_ids": train_row_ids,
        "eval_row_ids": eval_row_ids,
        "drift_summary": {
            "row_partition_ok": len(set(train_row_ids) & set(eval_row_ids)) == 0,
            "train_rows": len(train_row_ids),
            "eval_rows": len(eval_row_ids),
            "total_rows": len(train_row_ids) + len(eval_row_ids),
        },
    }


def ensure_split_manifest(
    dataset_path: Path,
    eval_ratio: float = 0.10,
    seed: int = 42,
    manifest_path: Optional[Path] = None,
    regenerate: bool = False,
) -> Tuple[Path, Dict[str, object]]:
    manifest_path = manifest_path or default_split_manifest_path(dataset_path)
    rows = list(iter_jsonl_rows(dataset_path))
    current_summary = summarize_dataset_rows(rows)

    if manifest_path.exists() and not regenerate:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("seed") == int(seed)
            and abs(float(manifest.get("eval_ratio", -1.0)) - float(eval_ratio)) < 1e-12
            and manifest.get("source_dataset") == current_summary
        ):
            return manifest_path, manifest

    manifest = build_split_manifest(rows, dataset_path, eval_ratio=eval_ratio, seed=seed)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path, manifest


def row_ids_for_split(manifest: Dict[str, object], split: str) -> Optional[set[int]]:
    if split == "all":
        return None
    key = f"{split}_row_ids"
    return {int(item) for item in manifest.get(key, [])}


def select_balanced_rows(
    rows: Sequence[Dict[str, object]],
    max_rows: int,
    *,
    seed: int = 42,
) -> List[Dict[str, object]]:
    if max_rows <= 0:
        return []
    if max_rows % 2 != 0:
        raise ValueError("balanced slice count must be even so real/fake counts stay matched")
    reals = [row for row in rows if row["final_json_target"]["overall_likelihood"] == "Real"]
    fakes = [row for row in rows if row["final_json_target"]["overall_likelihood"] == "AI-Generated"]
    limit_each = max_rows // 2
    if len(reals) < limit_each or len(fakes) < limit_each:
        raise ValueError(
            f"balanced slice count {max_rows} requires at least {limit_each} rows per class; "
            f"got real={len(reals)}, fake={len(fakes)}"
        )
    rng = random.Random(seed)
    reals = list(reals)
    fakes = list(fakes)
    rng.shuffle(reals)
    rng.shuffle(fakes)
    combined = reals[:limit_each] + fakes[:limit_each]
    rng.shuffle(combined)
    return combined


def load_rows_for_split(
    dataset_path: Path,
    split_row_ids: Optional[set[int]] = None,
    selected_row_ids: Optional[set[str]] = None,
    max_samples: int = 0,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    wanted = {str(item) for item in selected_row_ids} if selected_row_ids else None
    for idx, row in enumerate(iter_jsonl_rows(dataset_path)):
        row_id = resolve_row_id(row, idx)
        if split_row_ids is not None and row_id not in split_row_ids:
            continue
        if wanted is not None and str(row_id) not in wanted:
            continue
        rows.append(row)
        if wanted is None and max_samples > 0 and len(rows) >= max_samples:
            break
    return rows


def resolve_teacher_step2_payload(row: Dict[str, object]) -> Dict[str, object]:
    step2_internal = row.get("step2_internal")
    if isinstance(step2_internal, dict) and step2_internal.get("per_criterion_draft"):
        return step2_internal
    step2_draft = row.get("step2_draft")
    if isinstance(step2_draft, dict) and step2_draft.get("per_criterion_draft"):
        return step2_draft
    return {}


def load_teacher_predictions(
    teacher_jsonl_path: Path,
    wanted_row_ids: set[int],
) -> Dict[int, Dict[str, object]]:
    predictions: Dict[int, Dict[str, object]] = {}
    if not wanted_row_ids:
        return predictions
    for idx, row in enumerate(iter_jsonl_rows(teacher_jsonl_path)):
        if idx not in wanted_row_ids:
            continue
        step2_payload = resolve_teacher_step2_payload(row)
        final_json = normalize_final_prediction_payload(format_final_prediction_json(step2_payload))
        evidence_trace = evidence_trace_from_step2(step2_payload)
        predictions[idx] = {
            "final_json": final_json,
            "evidence_trace": evidence_trace,
            "raw_teacher_row": row,
        }
        if len(predictions) >= len(wanted_row_ids):
            break
    missing = sorted(wanted_row_ids - set(predictions))
    if missing:
        raise ValueError(f"teacher rows missing for derived row_ids: {missing[:10]}")
    return predictions


def _canonicalize_holmes_criterion(name: str) -> Optional[str]:
    cleaned = clean_text(name)
    if cleaned in CRITERIA:
        return cleaned
    return _HOLMES_ALIAS_MAP.get(criterion_lookup_key(cleaned))


def _extract_holmes_sections(response: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    for raw_name, raw_body in _HOLMES_SECTION_PATTERN.findall("\n" + (response or "")):
        canonical = _canonicalize_holmes_criterion(raw_name)
        if canonical is None or canonical in sections:
            continue
        cleaned = clean_text(raw_body)
        if cleaned:
            sections[canonical] = cleaned
    return sections


def _infer_implied_holmes_sentence(response: str, criterion: str) -> str:
    lowered_response = clean_text(response).casefold()
    if not lowered_response:
        return ""
    aliases = [alias for alias, canonical in _HOLMES_ALIAS_MAP.items() if canonical == criterion]
    sentences = [clean_text(sentence) for sentence in re.split(r"(?<=[.!?])\s+", response) if clean_text(sentence)]
    for alias in aliases:
        pattern = r"(?<!\w)" + re.escape(alias.casefold()) + r"(?!\w)"
        if not re.search(pattern, lowered_response):
            continue
        for sentence in sentences:
            if re.search(pattern, sentence.casefold()):
                return sentence
        return alias
    return ""


def render_origin_response_surface(response: str, overall_likelihood: str) -> str:
    sections = _extract_holmes_sections(response)
    lines = []
    fake_label = clean_text(overall_likelihood) == "AI-Generated"
    for criterion in CRITERIA:
        evidence = sections.get(criterion) or _infer_implied_holmes_sentence(response, criterion) or DEFAULT_EVIDENCE
        lowered = evidence.casefold()
        score = 1 if fake_label and evidence != DEFAULT_EVIDENCE and not any(
            marker in lowered for marker in _NEGATIVE_RESPONSE_MARKERS
        ) else 0
        lines.append(f"{criterion} | score={score} | evidence={evidence}")
    lines.append(f"overall_likelihood | {overall_likelihood}")
    return "\n".join(lines)


def render_explanation_surface(final_json: Dict[str, object]) -> str:
    normalized = normalize_final_prediction_payload(final_json or {})
    lines = []
    for entry in normalized.get("per_criterion", []):
        criterion = str(entry.get("criterion", ""))
        score = 1 if int(entry.get("aigc score", 0) or 0) else 0
        evidence = clean_text(entry.get("evidence", "")) or DEFAULT_EVIDENCE
        lines.append(f"{criterion} | score={score} | evidence={evidence}")
    lines.append(f"overall_likelihood | {normalized.get('overall_likelihood', 'Unknown')}")
    return "\n".join(lines)


def tokenize_text(text: str) -> List[str]:
    return _TOKEN_RE.findall(clean_text(text).lower())


def _bleu_1_builtin(reference_tokens: List[str], hypothesis_tokens: List[str]) -> float:
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    ref_counts = Counter(reference_tokens)
    hyp_counts = Counter(hypothesis_tokens)
    overlap = sum(min(count, ref_counts[token]) for token, count in hyp_counts.items())
    precision = overlap / len(hypothesis_tokens)
    brevity_penalty = 1.0
    if len(hypothesis_tokens) < len(reference_tokens):
        brevity_penalty = math.exp(1.0 - (len(reference_tokens) / max(len(hypothesis_tokens), 1)))
    return brevity_penalty * precision


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for idx, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[idx]))
        prev = curr
    return prev[-1]


def _rouge_l_builtin(reference_tokens: List[str], hypothesis_tokens: List[str]) -> float:
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, hypothesis_tokens)
    precision = lcs / len(hypothesis_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _chunks(matches: List[int]) -> int:
    if not matches:
        return 0
    count = 1
    for prev, curr in zip(matches, matches[1:]):
        if curr != prev + 1:
            count += 1
    return count


def _meteor_builtin(reference_tokens: List[str], hypothesis_tokens: List[str]) -> float:
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    ref_positions: Dict[str, List[int]] = {}
    for idx, token in enumerate(reference_tokens):
        ref_positions.setdefault(token, []).append(idx)
    matches = []
    used_positions = set()
    for token in hypothesis_tokens:
        for pos in ref_positions.get(token, []):
            if pos not in used_positions:
                used_positions.add(pos)
                matches.append(pos)
                break
    match_count = len(matches)
    if match_count == 0:
        return 0.0
    precision = match_count / len(hypothesis_tokens)
    recall = match_count / len(reference_tokens)
    f_mean = (10 * precision * recall) / (recall + 9 * precision) if (recall + 9 * precision) else 0.0
    penalty = 0.5 * ((_chunks(matches) / match_count) ** 3)
    return (1.0 - penalty) * f_mean


def _load_bleu_backend():
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        return sentence_bleu, SmoothingFunction
    except Exception:
        return None, None


def _load_rouge_backend():
    try:
        from rouge_score import rouge_scorer

        return rouge_scorer
    except Exception:
        return None


def _load_meteor_backend():
    try:
        from nltk.translate.meteor_score import single_meteor_score

        return single_meteor_score
    except Exception:
        return None


def compute_explanatory_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    enable_cider: bool = False,
) -> Dict[str, object]:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have identical length")

    sentence_bleu, smoothing_cls = _load_bleu_backend()
    rouge_backend = _load_rouge_backend()
    meteor_backend = _load_meteor_backend()

    bleu_scores: List[float] = []
    rouge_scores: List[float] = []
    meteor_scores: List[float] = []
    metric_backends = {
        "bleu_1": "nltk" if sentence_bleu else "builtin",
        "rouge_l": "rouge_score" if rouge_backend else "builtin",
        "meteor": "nltk" if meteor_backend else "builtin",
    }

    rouge_scorer = rouge_backend.RougeScorer(["rougeL"], use_stemmer=True) if rouge_backend else None
    smoother = smoothing_cls().method1 if smoothing_cls else None

    for reference, hypothesis in zip(references, hypotheses):
        ref_tokens = tokenize_text(reference)
        hyp_tokens = tokenize_text(hypothesis)

        if sentence_bleu:
            bleu_scores.append(
                float(sentence_bleu([ref_tokens], hyp_tokens, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smoother))
            )
        else:
            bleu_scores.append(_bleu_1_builtin(ref_tokens, hyp_tokens))

        if rouge_scorer:
            rouge_scores.append(float(rouge_scorer.score(reference, hypothesis)["rougeL"].fmeasure))
        else:
            rouge_scores.append(_rouge_l_builtin(ref_tokens, hyp_tokens))

        if meteor_backend:
            try:
                meteor_scores.append(float(meteor_backend(ref_tokens, hyp_tokens)))
            except LookupError:
                metric_backends["meteor"] = "builtin"
                meteor_scores.append(_meteor_builtin(ref_tokens, hyp_tokens))
        else:
            meteor_scores.append(_meteor_builtin(ref_tokens, hyp_tokens))

    report: Dict[str, object] = {
        "bleu_1": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        "rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
        "meteor": sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0,
        "metric_backends": metric_backends,
    }

    if enable_cider:
        try:
            from pycocoevalcap.cider.cider import Cider
        except Exception as exc:
            raise RuntimeError(
                "CIDEr requires pycocoevalcap. Install it separately before using --enable_cider."
            ) from exc
        refs = {idx: [references[idx]] for idx in range(len(references))}
        hyps = {idx: [hypotheses[idx]] for idx in range(len(hypotheses))}
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(refs, hyps)
        report["cider"] = float(cider_score)
        report["metric_backends"]["cider"] = "pycocoevalcap"
    else:
        report["cider"] = None

    return report


def checkpoint_rank_key(report: Dict[str, object]) -> Tuple[float, float, float, float, float]:
    values = []
    for field in _CHECKPOINT_SORT_FIELDS:
        value = report.get(field)
        values.append(float(value) if isinstance(value, (int, float)) else float("-inf"))
    return tuple(values)


def sort_checkpoint_reports(entries: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(entries, key=lambda entry: checkpoint_rank_key(entry.get("report", {})), reverse=True)


def load_epoch_eval_reports(training_eval_dir: Path) -> List[Dict[str, object]]:
    entries = []
    for path in sorted(training_eval_dir.glob("epoch_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        entries.append(
            {
                "report_path": str(path),
                "markdown_path": str(path.with_suffix(".md")),
                "report": report,
            }
        )
    return entries


def summarize_checkpoint_ranking(entries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    ranked = sort_checkpoint_reports(entries)
    best = ranked[0]["report"] if ranked else None
    summary_entries = []
    for rank, entry in enumerate(ranked, start=1):
        report = entry["report"]
        summary_entries.append(
            {
                "rank": rank,
                "report_path": entry["report_path"],
                "markdown_path": entry["markdown_path"],
                "epoch_label": report.get("epoch_label"),
                "epoch_index": report.get("epoch_index"),
                "adapter_path": report.get("adapter_path"),
                "overall_accuracy": report.get("overall_accuracy"),
                "overall_macro_f1": report.get("overall_macro_f1"),
                "criterion_macro_f1": report.get("criterion_macro_f1"),
                "json_parse_rate": report.get("json_parse_rate"),
                "rouge_l": report.get("rouge_l"),
                "meteor": report.get("meteor"),
            }
        )
    return {
        "ranking_fields": list(_CHECKPOINT_SORT_FIELDS),
        "best_checkpoint_path": best.get("adapter_path") if best else None,
        "best_report_path": ranked[0]["report_path"] if ranked else None,
        "entries": summary_entries,
    }


def render_checkpoint_ranking_markdown(summary: Dict[str, object], output_path: Path) -> None:
    lines = [
        "# Checkpoint Ranking",
        "",
        f"Best checkpoint: `{summary.get('best_checkpoint_path') or 'N/A'}`",
        "",
        "| Rank | Epoch | Overall Acc | Overall Macro F1 | Criterion Macro F1 | JSON Parse | ROUGE-L | METEOR | Adapter |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in summary.get("entries", []):
        lines.append(
            "| {rank} | {epoch} | {acc} | {overall_f1} | {criterion_f1} | {parse} | {rouge} | {meteor} | {adapter} |".format(
                rank=entry.get("rank"),
                epoch=entry.get("epoch_label") or entry.get("epoch_index") or "N/A",
                acc=format_metric_value(entry.get("overall_accuracy")),
                overall_f1=format_metric_value(entry.get("overall_macro_f1")),
                criterion_f1=format_metric_value(entry.get("criterion_macro_f1")),
                parse=format_metric_value(entry.get("json_parse_rate")),
                rouge=format_metric_value(entry.get("rouge_l")),
                meteor=format_metric_value(entry.get("meteor")),
                adapter=entry.get("adapter_path") or "N/A",
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
