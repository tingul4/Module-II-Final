import json
import re
from typing import Dict, List, Tuple


CRITERIA: List[str] = [
    "Lighting & Shadows Consistency",
    "Edges & Boundaries",
    "Texture & Resolution",
    "Perspective & Spatial Relationships",
    "Physical & Common Sense Logic",
    "Text & Symbols",
    "Human & Biological Structure Integrity",
    "Material & Object Details",
]

DEFAULT_EVIDENCE = "Not assessable due to lack of relevant content"
REAL_LABEL = "Real"
FAKE_LABEL = "AI-Generated"
UNCERTAIN_LABEL = "Uncertain"

NEGATIVE_EVIDENCE_MARKERS = (
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

ARTIFACT_TAXONOMY_RULES = {
    "Lighting & Shadows Consistency": [
        ("shadow_mismatch", ("shadow", "shadows", "light source", "lighting", "illumination")),
        ("highlight_mismatch", ("highlight", "glare", "specular", "reflection")),
    ],
    "Edges & Boundaries": [
        ("edge_discontinuity", ("edge", "boundary", "outline", "halo", "cutout", "blending")),
        ("shape_breakage", ("discontinuity", "broken", "misaligned", "fragmented")),
    ],
    "Texture & Resolution": [
        ("texture_repetition", ("repetitive", "repetition", "pattern", "uniform", "duplicated")),
        ("resolution_mismatch", ("blur", "blurry", "pixelation", "low-resolution", "oversharp", "crisp")),
        ("noise_artifact", ("noise", "grain", "artifact", "compression")),
    ],
    "Perspective & Spatial Relationships": [
        ("perspective_distortion", ("perspective", "distorted", "warped", "geometry")),
        ("scale_mismatch", ("scale", "proportion", "proportions", "too large", "too small")),
        ("spatial_inconsistency", ("misaligned", "position", "spacing", "depth")),
    ],
    "Physical & Common Sense Logic": [
        ("physical_violation", ("physics", "gravity", "floating", "impossible", "collision")),
        ("object_relation_error", ("relationship", "interaction", "support", "logic", "common sense")),
    ],
    "Text & Symbols": [
        ("text_gibberish", ("gibberish", "garbled", "unreadable", "nonsensical", "misspelled")),
        ("symbol_error", ("symbol", "logo", "sign", "letter", "character")),
    ],
    "Human & Biological Structure Integrity": [
        ("anatomy_error", ("finger", "hand", "arm", "leg", "limb", "face", "eye", "teeth", "anatomy")),
        ("pose_structure_error", ("pose", "joint", "body", "proportion", "biological")),
    ],
    "Material & Object Details": [
        ("material_rendering_error", ("surface", "material", "fabric", "skin", "fur", "plastic", "metallic")),
        ("surface_repetition", ("repetitive", "pattern", "sheen", "gloss", "grain", "texture")),
    ],
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def compact_json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def normalize_label(label: str) -> str:
    text = clean_text(label)
    if text == FAKE_LABEL:
        return FAKE_LABEL
    if text == REAL_LABEL:
        return REAL_LABEL
    return UNCERTAIN_LABEL


def normalize_final_prediction_payload(payload: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(payload, dict):
        return payload

    entry_map: Dict[str, Dict[str, object]] = {}
    any_positive = False
    for entry in payload.get("per_criterion", []):
        if not isinstance(entry, dict):
            continue
        criterion = canonicalize_criterion(entry.get("criterion", ""))
        if criterion not in CRITERIA or criterion in entry_map:
            continue
        score = 1 if int(entry.get("aigc score", 0) or 0) else 0
        evidence = clean_text(entry.get("evidence", "")) or DEFAULT_EVIDENCE
        normalized_entry = dict(entry)
        normalized_entry["criterion"] = criterion
        normalized_entry["evidence"] = evidence
        normalized_entry["aigc score"] = score
        entry_map[criterion] = normalized_entry
        if score:
            any_positive = True

    normalized_entries = []
    for criterion in CRITERIA:
        normalized_entries.append(
            entry_map.get(
                criterion,
                {
                    "criterion": criterion,
                    "evidence": DEFAULT_EVIDENCE,
                    "aigc score": 0,
                },
            )
        )

    normalized_payload = dict(payload)
    normalized_payload["per_criterion"] = normalized_entries
    normalized_payload["overall_likelihood"] = FAKE_LABEL if any_positive else REAL_LABEL
    return normalized_payload


def canonicalize_criterion(name: str) -> str:
    cleaned = clean_text(name)
    for criterion in CRITERIA:
        if cleaned == criterion:
            return criterion
    return cleaned


def truncate_words(text: str, max_words: int) -> str:
    cleaned = clean_text(text)
    if max_words <= 0 or not cleaned:
        return cleaned
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words])


def teacher_entry_score(entry: Dict[str, object]) -> int:
    if not isinstance(entry, dict):
        return 0
    if "final_score" in entry:
        return 1 if int(entry.get("final_score", 0) or 0) else 0
    if "proposed_score" in entry:
        return 1 if int(entry.get("proposed_score", 0) or 0) else 0
    if "score" in entry:
        return 1 if int(entry.get("score", 0) or 0) else 0
    if "aigc score" in entry:
        return 1 if int(entry.get("aigc score", 0) or 0) else 0
    return 0


def format_final_prediction_json(step2_draft: Dict[str, object]) -> Dict[str, object]:
    per_criterion = []
    any_positive = False
    for criterion in CRITERIA:
        draft_entry = find_entry(step2_draft.get("per_criterion_draft", []), criterion)
        score = teacher_entry_score(draft_entry)
        evidence = clean_text(draft_entry.get("evidence", "")) or DEFAULT_EVIDENCE
        if score:
            any_positive = True
        per_criterion.append(
            {
                "criterion": criterion,
                "evidence": evidence,
                "aigc score": score,
            }
        )
    overall = normalize_label(step2_draft.get("overall_likelihood", UNCERTAIN_LABEL))
    if overall == UNCERTAIN_LABEL:
        overall = FAKE_LABEL if any_positive else REAL_LABEL
    return {"per_criterion": per_criterion, "overall_likelihood": overall}


def infer_artifact_taxonomy(criterion: str, evidence: str, score: int) -> str:
    if not score:
        return "none"
    lowered = clean_text(evidence).casefold()
    for label, markers in ARTIFACT_TAXONOMY_RULES.get(criterion, []):
        if any(marker in lowered for marker in markers):
            return label
    return "artifact_generic"


def evidence_trace_from_step2(step2_draft: Dict[str, object]) -> Dict[str, object]:
    trace_entries = []
    for criterion in CRITERIA:
        draft_entry = find_entry(step2_draft.get("per_criterion_draft", []), criterion)
        score = teacher_entry_score(draft_entry)
        evidence = clean_text(draft_entry.get("evidence", "")) or DEFAULT_EVIDENCE
        support_type = clean_text(draft_entry.get("support_type", "")) or "unsupported"
        holmes_span = clean_text(draft_entry.get("holmes_span", ""))
        non_applicable = bool(draft_entry.get("non_applicable", evidence == DEFAULT_EVIDENCE))
        artifact_score_conflict = bool(
            draft_entry.get("artifact_score_conflict", score == 1 and evidence == DEFAULT_EVIDENCE)
        )
        trace_entries.append(
            {
                "criterion": criterion,
                "score": score,
                "evidence": evidence,
                "support_type": support_type,
                "holmes_span": holmes_span,
                "artifact_taxonomy": infer_artifact_taxonomy(criterion, evidence, score),
                "non_applicable": non_applicable,
                "artifact_score_conflict": artifact_score_conflict,
            }
        )
    return {
        "overall_likelihood": normalize_label(step2_draft.get("overall_likelihood", UNCERTAIN_LABEL)),
        "per_criterion": trace_entries,
    }


def compact_trace_payload(
    trace: Dict[str, object],
    evidence_words: int = 14,
    holmes_span_words: int = 12,
) -> Dict[str, object]:
    compact_entries = []
    for criterion in CRITERIA:
        item = find_entry(trace.get("per_criterion", []), criterion)
        evidence = clean_text(item.get("evidence", "")) or DEFAULT_EVIDENCE
        holmes_span = clean_text(item.get("holmes_span", ""))
        if evidence != DEFAULT_EVIDENCE:
            evidence = truncate_words(evidence, evidence_words)
        holmes_span = truncate_words(holmes_span, holmes_span_words)
        compact_entries.append(
            {
                "criterion": criterion,
                "score": 1 if int(item.get("score", 0) or 0) else 0,
                "evidence": evidence,
                "support_type": clean_text(item.get("support_type", "unsupported")) or "unsupported",
                "holmes_span": holmes_span,
                "artifact_taxonomy": clean_text(item.get("artifact_taxonomy", "none")) or "none",
                "non_applicable": bool(item.get("non_applicable", False)),
                "artifact_score_conflict": bool(item.get("artifact_score_conflict", False)),
            }
        )
    return {
        "overall_likelihood": normalize_label(trace.get("overall_likelihood", UNCERTAIN_LABEL)),
        "per_criterion": compact_entries,
    }


def taxonomy_target_from_trace(trace: Dict[str, object]) -> Dict[str, object]:
    return {
        "overall_likelihood": normalize_label(trace.get("overall_likelihood", UNCERTAIN_LABEL)),
        "per_criterion": [
            {
                "criterion": item["criterion"],
                "artifact_taxonomy": item["artifact_taxonomy"],
                "support_type": item["support_type"],
            }
            for item in trace.get("per_criterion", [])
        ],
    }


def consistency_target_from_trace(trace: Dict[str, object]) -> Dict[str, object]:
    entries = []
    any_positive = False
    for item in trace.get("per_criterion", []):
        score = 1 if int(item.get("score", 0) or 0) else 0
        evidence = clean_text(item.get("evidence", ""))
        support_type = clean_text(item.get("support_type", "unsupported"))
        if score:
            any_positive = True
        has_negative_marker = any(marker in evidence.casefold() for marker in NEGATIVE_EVIDENCE_MARKERS)
        consistent = not (
            score == 1 and (evidence == DEFAULT_EVIDENCE or has_negative_marker or support_type == "unsupported")
        )
        reason = "evidence supports score"
        if not consistent:
            reason = "positive score lacks grounded artifact evidence"
        entries.append(
            {
                "criterion": item["criterion"],
                "consistent": consistent,
                "reason": reason,
            }
        )
    expected_overall = FAKE_LABEL if any_positive else REAL_LABEL
    overall_label = normalize_label(trace.get("overall_likelihood", UNCERTAIN_LABEL))
    return {
        "overall_consistent": overall_label == expected_overall,
        "expected_overall_likelihood": expected_overall,
        "per_criterion": entries,
    }


def quality_flags_from_trace(trace: Dict[str, object]) -> List[str]:
    flags: List[str] = []
    overall = normalize_label(trace.get("overall_likelihood", UNCERTAIN_LABEL))
    positives = sum(1 for item in trace.get("per_criterion", []) if int(item.get("score", 0) or 0))
    if overall == REAL_LABEL and positives:
        flags.append("real_has_positive_artifact")
    if overall == FAKE_LABEL and positives == 0:
        flags.append("fake_has_no_positive_artifact")
    for item in trace.get("per_criterion", []):
        if item.get("artifact_score_conflict"):
            flags.append(f"{item['criterion']}:artifact_score_conflict")
        if int(item.get("score", 0) or 0) and clean_text(item.get("evidence", "")) == DEFAULT_EVIDENCE:
            flags.append(f"{item['criterion']}:positive_without_evidence")
    return flags


def find_entry(entries: object, criterion: str) -> Dict[str, object]:
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and canonicalize_criterion(entry.get("criterion", "")) == criterion:
                return dict(entry)
    return {"criterion": criterion, "proposed_score": 0, "evidence": DEFAULT_EVIDENCE}


def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:]


def safe_json_loads(text: str) -> Tuple[Dict[str, object], str]:
    candidate = extract_first_json_object(text)
    try:
        return json.loads(candidate), ""
    except json.JSONDecodeError as exc:
        return {}, str(exc)


# Legacy alias kept so older code paths can continue to import the prior name.
format_competition_json = format_final_prediction_json
