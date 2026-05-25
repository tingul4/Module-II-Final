#!/usr/bin/env python3
"""Convert Holmes SFT supervision into LPCVC3-style JSONL targets."""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zipfile import ZipFile

import requests
import torch
from tqdm.auto import tqdm


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

HIGH_RISK_CRITERIA = {
    "Text & Symbols",
    "Human & Biological Structure Integrity",
    "Perspective & Spatial Relationships",
    "Physical & Common Sense Logic",
}

LOW_RISK_CRITERIA = [criterion for criterion in CRITERIA if criterion not in HIGH_RISK_CRITERIA]

HOLMES_PRIMARY_CRITERIA = {
    "Lighting & Shadows Consistency",
    "Edges & Boundaries",
    "Texture & Resolution",
    "Perspective & Spatial Relationships",
}

LOW_COVERAGE_HOLMES_STRONG_CRITERIA = {
    "Text & Symbols",
    "Human & Biological Structure Integrity",
}

LPCVC_ADDED_CRITERIA = {
    "Material & Object Details",
}

HYBRID_CRITERIA = {
    "Physical & Common Sense Logic",
}

VALID_SUPPORT_TYPES = {
    "explicit_holmes",
    "implied_holmes",
    "image_only",
    "unsupported",
}

VALID_JUDGE_VERDICTS = {
    "accept",
    "downgrade_to_0",
    "needs_specialist_check",
}

VALID_SCORE_CONSISTENCY = {
    "consistent",
    "inconsistent",
}

VALID_RECOMMENDED_SCORE = {
    "keep",
    "set_to_0",
    "set_to_1",
    "defer_to_specialist",
}

VALID_SPECIALIST_CONFIDENCE = {
    "high",
    "medium",
    "low",
}

DEFAULT_EVIDENCE = "Not assessable due to lack of relevant content"
SOURCE_NAME = "holmes_sft"
REAL_LABEL = "Real"
FAKE_LABEL = "AI-Generated"
GENERATOR_SYSTEM_PROMPT = (
    "You convert Holmes explanations into structured LPCVC draft supervision. "
    "This is a Holmes-first rewrite task, not a free re-judging task. Return JSON only."
)
JUDGE_SYSTEM_PROMPT = (
    "You are the quality-control judge for Holmes-to-LPCVC conversion. "
    "Review proposed criterion evidence conservatively and return JSON only."
)
SPECIALIST_SYSTEM_PROMPT = (
    "You are a narrow specialist for a single LPCVC criterion. "
    "Answer only for the requested criterion and return JSON only."
)
DEBUG_RAW_OUTPUT_DIRNAME = "debug_raw_outputs"

NEGATIVE_EVIDENCE_MARKERS = (
    "not assessable",
    "does not apply",
    "not applicable",
    "n/a",
    "no discernible",
    "no visible",
    "no text",
    "no symbols",
    "no faces",
    "no face",
    "no human",
    "inanimate",
    "lack of relevant content",
)

ARTIFACT_EVIDENCE_MARKERS = (
    "inconsistent",
    "too sharp",
    "too dark",
    "overly crisp",
    "discontinuities",
    "artifact",
    "unnaturally",
    "synthetic",
    "slightly off",
    "distorted",
    "gibberish",
    "warped",
    "blurred",
    "pixelation",
    "repetitive pattern",
    "does not fully align",
)

MATERIAL_OBJECT_POSITIVE_MARKERS = (
    "material",
    "materials",
    "surface",
    "finish",
    "fabric",
    "texture",
    "coating",
    "metallic",
    "plastic",
    "wood",
    "fur",
    "skin",
    "color",
    "colours",
    "colors",
    "hue",
    "saturation",
    "vibrant",
    "uniform",
    "pattern",
    "patterns",
    "repetitive",
    "reflection",
    "reflections",
    "reflective",
    "glossy",
    "gloss",
    "matte",
    "grain",
    "details",
    "detail",
    "smooth",
    "rough",
    "roughness",
    "scales",
    "scale pattern",
    "skin texture",
    "rubber",
    "organic",
    "painted",
    "plastic-like",
    "realistic",
    "realism",
    "sheen",
)

MATERIAL_OBJECT_EXCLUSION_MARKERS = (
    "edge",
    "edges",
    "outline",
    "outlined",
    "shape",
    "shapes",
    "proportion",
    "proportions",
    "perspective",
    "distortion",
    "distorted",
    "warped",
    "elongated",
    "stretched",
    "anatomical",
    "body structure",
    "spatial",
    "shadow",
    "shadows",
    "lighting",
    "light source",
    "text",
    "symbol",
)

CANONICAL_NAME_MAP: Dict[str, str] = {name.casefold(): name for name in CRITERIA}
CRITERION_ALIASES: Dict[str, str] = {
    "lighting": "Lighting & Shadows Consistency",
    "shadows": "Lighting & Shadows Consistency",
    "lighting & shadows consistency": "Lighting & Shadows Consistency",
    "shadows and lighting": "Lighting & Shadows Consistency",
    "line segments": "Lighting & Shadows Consistency",
    "edges": "Edges & Boundaries",
    "boundaries": "Edges & Boundaries",
    "edges & boundaries": "Edges & Boundaries",
    "texture": "Texture & Resolution",
    "texture & resolution": "Texture & Resolution",
    "clarity": "Texture & Resolution",
    "overall hue": "Texture & Resolution",
    "resolution": "Texture & Resolution",
    "perspective": "Perspective & Spatial Relationships",
    "perspective & spatial relationships": "Perspective & Spatial Relationships",
    "distortion": "Perspective & Spatial Relationships",
    "physical laws": "Physical & Common Sense Logic",
    "physical & common sense logic": "Physical & Common Sense Logic",
    "common sense": "Physical & Common Sense Logic",
    "text": "Text & Symbols",
    "text & symbols": "Text & Symbols",
    "symbols": "Text & Symbols",
    "ocr": "Text & Symbols",
    "faces": "Human & Biological Structure Integrity",
    "body structure": "Human & Biological Structure Integrity",
    "human & biological structure integrity": "Human & Biological Structure Integrity",
    "human structure": "Human & Biological Structure Integrity",
    "biological structure": "Human & Biological Structure Integrity",
    "material": "Material & Object Details",
    "material & object details": "Material & Object Details",
    "object details": "Material & Object Details",
}


@dataclass
class HolmesRecord:
    image_member: str
    image_output_rel: str
    label: str
    original_query: str
    original_response: str
    source: str = SOURCE_NAME


@dataclass
class GeneratorDraftRecord:
    image: str
    source: str
    original_query: str
    original_response: str
    step1_target: str
    step2_draft: Dict[str, object]


@dataclass
class TeacherBackends:
    generator: TeacherBackend
    judge: TeacherBackend
    specialist: TeacherBackend


class TeacherBackend:
    batch_size = 16

    def generate_json_prompt(self, prompt: str, image_path: Path, system_prompt: str) -> Dict[str, object]:
        raise NotImplementedError

    def generate_json_prompts_batch(
        self,
        prompts: Sequence[str],
        image_paths: Sequence[Path],
        system_prompt: str,
    ) -> List[Dict[str, object]]:
        return [
            self.generate_json_prompt(prompt, image_path, system_prompt)
            for prompt, image_path in zip(prompts, image_paths)
        ]


class HeuristicTeacherBackend(TeacherBackend):
    SECTION_PATTERN = re.compile(r"\n\s*\d+\.\s+\*\*(.*?)\*\*:\s*(.+?)(?=\n\s*\d+\.\s+\*\*|\Z)", re.S)
    DESCRIPTION_PATTERN = re.compile(
        r"Image Description:\s*(.+?)(?=\n\s*Based on the provided|\Z)", re.S
    )

    def extract_sections(self, response: str) -> Dict[str, List[str]]:
        sections: Dict[str, List[str]] = defaultdict(list)
        for raw_name, raw_body in self.SECTION_PATTERN.findall(response):
            canonical = canonicalize_criterion(raw_name)
            if canonical is None:
                continue
            cleaned = clean_text(raw_body)
            if cleaned:
                sections[canonical].append(cleaned)
        return sections

    def extract_description(self, response: str) -> str:
        match = self.DESCRIPTION_PATTERN.search(response)
        return clean_text(match.group(1)) if match else ""

    def generate_json_prompt(self, prompt: str, image_path: Path, system_prompt: str) -> Dict[str, object]:
        del prompt, image_path, system_prompt
        raise NotImplementedError("Heuristic backend uses rule-based helpers directly.")


class OpenAICompatibleTeacherBackend(TeacherBackend):
    def __init__(
        self,
        api_base: str,
        model: str,
        api_key: Optional[str],
        timeout: int,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_json_prompt(self, prompt: str, image_path: Path, system_prompt: str) -> Dict[str, object]:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
        }

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return parse_teacher_output(content)


class TransformersGemma4TeacherBackend(TeacherBackend):
    def __init__(
        self,
        model_name: str,
        device: str,
        torch_dtype: str,
        max_new_tokens: int,
        temperature: float,
        batch_size: int = 1,
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = max(1, batch_size)
        dtype = resolve_torch_dtype(torch_dtype)

        processor_kwargs = {"trust_remote_code": True}
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }

        if device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = {"": device}

        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            if hasattr(self.model.generation_config, "top_p"):
                self.model.generation_config.top_p = None
            if hasattr(self.model.generation_config, "top_k"):
                self.model.generation_config.top_k = None

    def _generate_text_batch(
        self,
        prompts: Sequence[str],
        image_paths: Sequence[Path],
        system_prompt: str,
    ) -> List[str]:
        from PIL import Image

        conversations = []
        opened_images = []
        try:
            for prompt, image_path in zip(prompts, image_paths):
                image = Image.open(image_path).convert("RGB")
                opened_images.append(image)
                conversations.append(
                    [
                        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ]
                )

            inputs = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
            }
            if self.temperature > 0:
                generate_kwargs["temperature"] = self.temperature

            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **generate_kwargs)

            prompt_width = inputs["input_ids"].shape[-1]
            generated_texts: List[str] = []
            for idx in range(len(prompts)):
                generated_ids = outputs[idx][prompt_width:]
                generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
                generated_texts.append(generated_text)
            return generated_texts
        finally:
            for image in opened_images:
                image.close()

    def generate_json_prompt(self, prompt: str, image_path: Path, system_prompt: str) -> Dict[str, object]:
        generated_text = self._generate_text_batch([prompt], [image_path], system_prompt)[0]
        return parse_teacher_output(generated_text)

    def generate_json_prompt_with_raw(
        self,
        prompt: str,
        image_path: Path,
        system_prompt: str,
    ) -> Tuple[Dict[str, object], str]:
        generated_text = self._generate_text_batch([prompt], [image_path], system_prompt)[0]
        return parse_teacher_output(generated_text), generated_text

    def generate_json_prompts_batch(
        self,
        prompts: Sequence[str],
        image_paths: Sequence[Path],
        system_prompt: str,
    ) -> List[Dict[str, object]]:
        generated_texts = self._generate_text_batch(prompts, image_paths, system_prompt)

        parsed_outputs: List[Dict[str, object]] = []
        retried_indices: List[int] = []
        for idx, generated_text in enumerate(generated_texts):
            try:
                parsed_outputs.append(parse_teacher_output(generated_text))
                continue
            except json.JSONDecodeError as exc:
                retried_indices.append(idx)

            retry_text = self._generate_text_batch(
                [prompts[idx]],
                [image_paths[idx]],
                system_prompt,
            )[0]
            parsed_outputs.append(parse_teacher_output(retry_text))

        if retried_indices:
            joined = ",".join(str(idx) for idx in retried_indices)
            print(
                f"[info] recovered malformed JSON via single-item retry for batch indices: {joined}",
                file=sys.stderr,
            )

        return parsed_outputs


def build_generator_prompt_v2(
    record: HolmesRecord,
    anchor_map: Dict[str, Dict[str, str]],
    description: str,
    relax_image_only_candidates: bool = False,
) -> str:
    anchor_lines = []
    for criterion in CRITERIA:
        anchor = anchor_map[criterion]
        anchor_lines.append(
            f'- {criterion}: support_type={anchor["support_type"]}; holmes_span="{anchor["holmes_span"]}"'
        )
    anchors_text = "\n".join(anchor_lines)
    criteria_description = "\n".join(f"- {name}" for name in CRITERIA)
    primary_text = ", ".join(sorted(HOLMES_PRIMARY_CRITERIA))
    low_coverage_text = ", ".join(sorted(LOW_COVERAGE_HOLMES_STRONG_CRITERIA))
    lpcvc_added_text = ", ".join(sorted(LPCVC_ADDED_CRITERIA))
    hybrid_text = ", ".join(sorted(HYBRID_CRITERIA))
    relaxed_rule = ""
    if relax_image_only_candidates and record.label == FAKE_LABEL:
        relaxed_rule = (
            f"\n10. Keep Holmes-first behavior for primary Holmes-aligned criteria: {primary_text}."
            f"\n11. For low-coverage but Holmes-strong criteria: {low_coverage_text}, Holmes remains preferred, but keep strong `image_only` candidates when Holmes is silent and the anomaly is visually concrete."
            f"\n12. For LPCVC-added criteria: {lpcvc_added_text}, prefer preserving concrete `image_only` candidates over collapsing them to `unsupported`."
            f"\n13. For hybrid criteria: {hybrid_text}, preserve image-based candidates when the physical or common-sense failure is specific and image-grounded."
        )
    return textwrap.dedent(
        f"""
        Convert a Holmes explanation into LPCVC Track 3 draft supervision.

        Rules:
        1. Keep the label fixed at {record.label}.
        2. First analyze all 8 LPCVC criteria internally, then output the final JSON only.
        3. Holmes evidence is primary. `image_only` is candidate-only and not a final positive by itself.
        4. For each criterion, decide whether the strongest support is `explicit_holmes`, `implied_holmes`, `image_only`, or `unsupported`.
        5. `proposed_score = 1` means the criterion contains an AI artifact. `0` means no such artifact or not applicable.
        6. Do not pair clear fake artifact evidence with `proposed_score = 0`.
        7. Evidence must be short, specific, complete, and criterion-aligned.
        8. Return all 8 criteria in the exact order shown below.
        9. For Real images, do not add image_only positives.
        10. `step1_target` must be `Key points:` plus 2 to 3 short image-specific segments.
        11. When Holmes is silent, relax `image_only` most for LPCVC-added criteria, moderately for Human/Bio and Physical/Common Sense, and least for primary Holmes-aligned criteria.
        12. Use `Material & Object Details` for material and surface realism first: skin, scales, fur, fabric, finish, reflectance, smoothness, roughness, gloss, wear, grain, or repetitive surface detail. Color or saturation alone is weaker evidence unless it clearly supports a material/surface abnormality.
        13. Do not use `Material & Object Details` for geometry, edge quality, perspective, lighting/shadow inconsistency, text, or anatomy/limb structure. If another criterion is the better fit, keep the evidence there instead of duplicating it in Material.
        14. Return JSON only.
        {relaxed_rule}

        Criteria order:
        {criteria_description}

        Holmes anchors:
        {anchors_text}

        Holmes response:
        {record.original_response}

        Required JSON schema:
        {{
          "step1_target": "Key points: ...",
          "per_criterion_draft": [
            {{
              "criterion": "Lighting & Shadows Consistency",
              "proposed_score": 0,
              "evidence": "...",
              "support_type": "explicit_holmes",
              "holmes_span": "..."
            }}
          ]
        }}
        """
    ).strip()


def build_judge_prompt(record: HolmesRecord, draft_entries: Sequence[Dict[str, object]]) -> str:
    slim_drafts = []
    for entry in draft_entries:
        slim_drafts.append(
            {
                "criterion": entry["criterion"],
                "proposed_score": entry["proposed_score"],
                "evidence": finalize_evidence_sentence(str(entry.get("evidence", "")), 24),
                "support_type": entry["support_type"],
                "holmes_span": clip_words_plain(str(entry.get("holmes_span", "")), 20),
            }
        )
    draft_json = json.dumps(slim_drafts, indent=2, ensure_ascii=True)
    return textwrap.dedent(
        f"""
        Review Holmes-to-LPCVC draft entries for consistency.

        Rules:
        1. Keep the label fixed at {record.label}.
        2. Prefer Holmes-backed evidence over image-only speculation.
        3. Accept `explicit_holmes` only when evidence matches the criterion.
        4. Accept `implied_holmes` only when Holmes strongly supports it.
        5. Accept `image_only` only when the issue is obvious, specific, non-redundant, and not contradicted by Holmes.
        6. Use `needs_specialist_check` only for uncertain high-risk criteria: Text, Human/Bio, Perspective, Physical/Common Sense.
        7. Return brief reasons only. Keep each reason under 14 words.
        8. Return JSON only.

        Draft entries:
        {draft_json}

        Required JSON schema:
        {{
          "per_criterion_review": [
            {{
              "criterion": "Lighting & Shadows Consistency",
              "verdict": "accept",
              "reason": "...",
              "score_consistency": "consistent",
              "recommended_score": "keep"
            }}
          ]
        }}
        """
    ).strip()


def build_specialist_prompt(record: HolmesRecord, criterion: str, draft_entry: Dict[str, object]) -> str:
    specialist_focus = {
        "Text & Symbols": "Only judge whether visible text or symbols exist and whether they are malformed, unreadable, or inconsistent.",
        "Human & Biological Structure Integrity": "Only judge whether face, body, limb, or anatomy structure is clearly abnormal.",
        "Perspective & Spatial Relationships": "Only judge whether geometry, scale, or perspective relationships are inconsistent.",
        "Physical & Common Sense Logic": "Only judge whether the scene clearly violates physical or common-sense object relationships.",
    }.get(criterion, "Judge only the requested criterion.")

    return textwrap.dedent(
        f"""
        Review one LPCVC criterion only.

        Criterion: {criterion}
        Label: {record.label}
        Focus: {specialist_focus}

        Draft entry:
        {json.dumps({
            "criterion": draft_entry.get("criterion"),
            "proposed_score": draft_entry.get("proposed_score"),
            "evidence": finalize_evidence_sentence(str(draft_entry.get("evidence", "")), 24),
            "support_type": draft_entry.get("support_type"),
            "holmes_span": clip_words_plain(str(draft_entry.get("holmes_span", "")), 20),
        }, indent=2, ensure_ascii=True)}

        Return JSON only with this schema:
        {{
          "criterion": "{criterion}",
          "verdict": "accept | downgrade_to_0",
          "reason": "...",
          "evidence": "...",
          "confidence": "high | medium | low"
        }}
        """
    ).strip()


JSON_SMART_CHAR_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u3000": " ",
    }
)


def normalize_teacher_json_text(text: str) -> str:
    normalized = text.replace("\ufeff", "").translate(JSON_SMART_CHAR_TRANSLATION)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n").strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json)?\s*", "", normalized)
        normalized = re.sub(r"\s*```$", "", normalized)
    return normalized.strip()


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


def close_unbalanced_json(text: str) -> str:
    stack: List[str] = []
    in_string = False
    escape = False

    for ch in text:
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
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and ch == stack[-1]:
                stack.pop()

    if in_string:
        text += '"'
    if stack:
        text += "".join(reversed(stack))
    return text


def repair_teacher_json_candidates(text: str) -> List[str]:
    candidates: List[str] = []

    normalized = normalize_teacher_json_text(text)
    extracted = extract_first_json_object(normalized)

    def add(candidate: str) -> None:
        cleaned = clean_text(candidate) if "\n" not in candidate else candidate.strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    add(normalized)
    if extracted != normalized:
        add(extracted)

    base_candidates = list(candidates)
    for candidate in base_candidates:
        no_trailing_commas = re.sub(r",(?=\s*[}\]])", "", candidate)
        add(no_trailing_commas)

        inserted_missing_field_commas = re.sub(
            r'((?:true|false|null)|[0-9\]}"])(\s*)(?="[^"]+"\s*:)',
            r"\1,\2",
            no_trailing_commas,
        )
        add(inserted_missing_field_commas)
        add(close_unbalanced_json(inserted_missing_field_commas))

    return candidates


def parse_teacher_output(content: str) -> Dict[str, object]:
    last_error: Optional[Exception] = None
    for candidate in repair_teacher_json_candidates(content):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("Unable to locate JSON object", content, 0)


def save_raw_teacher_output(
    output_root: Path,
    phase: str,
    image_rel: str,
    content: str,
    error: Exception,
    attempt: str,
) -> None:
    debug_dir = output_root / DEBUG_RAW_OUTPUT_DIRNAME / phase
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_image = re.sub(r"[^A-Za-z0-9._-]+", "_", image_rel)
    path = debug_dir / f"{safe_image}.{attempt}.txt"
    payload = [
        f"image: {image_rel}",
        f"phase: {phase}",
        f"attempt: {attempt}",
        f"error: {type(error).__name__}: {error}",
        "",
        content,
    ]
    path.write_text("\n".join(payload), encoding="utf-8")


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    mapping = {
        "auto": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[normalized]


def canonicalize_criterion(name: str) -> Optional[str]:
    lowered = clean_text(name).casefold()
    if lowered in CANONICAL_NAME_MAP:
        return CANONICAL_NAME_MAP[lowered]
    return CRITERION_ALIASES.get(lowered)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def strip_leading_label(text: str) -> str:
    cleaned = clean_text(text)
    cleaned = re.sub(r"^\*\*(.*?)\*\*:\s*", "", cleaned)
    cleaned = re.sub(r"^[A-Z][A-Za-z& /\-]+:\s*", "", cleaned)
    cleaned = cleaned.replace("**", "")
    return clean_text(cleaned)


BAD_FRAGMENT_START_PREFIXES = (
    "such as",
    "especially",
    "particularly",
    "including",
    "while",
    "and",
    "but",
    "or",
    "which",
    "that",
)
BAD_FRAGMENT_ENDINGS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def is_viable_fragment(text: str, min_words: int = 6) -> bool:
    words = clean_text(text).split()
    if len(words) < min_words:
        return False
    lowered = clean_text(text).casefold()
    if any(lowered.startswith(prefix) for prefix in BAD_FRAGMENT_START_PREFIXES):
        return False
    if words[-1].rstrip(".,;:!?").casefold() in BAD_FRAGMENT_ENDINGS:
        return False
    return True


def compact_complete_text(text: str, max_words: int, default: str = "") -> str:
    cleaned = strip_leading_label(text)
    if not cleaned:
        return default
    sentences = split_sentences(cleaned)
    candidate = clean_text(sentences[0] if sentences else cleaned)
    if len(candidate.split()) > max_words:
        clause_parts = [clean_text(part) for part in re.split(r"[;:]", candidate) if clean_text(part)]
        shorter_clause = next(
            (part for part in clause_parts if len(part.split()) <= max_words and is_viable_fragment(part)),
            "",
        )
        if shorter_clause:
            candidate = shorter_clause
    if len(candidate.split()) > max_words:
        comma_parts = [clean_text(part) for part in re.split(r",", candidate) if clean_text(part)]
        shorter_comma = next(
            (part for part in comma_parts if len(part.split()) <= max_words and is_viable_fragment(part)),
            "",
        )
        if shorter_comma:
            candidate = shorter_comma
    if len(candidate.split()) > max_words:
        candidate = clip_words_plain(candidate, max_words)
    candidate = candidate.strip(" \"'")
    candidate = re.sub(r"\.\.+", ".", candidate)
    if candidate and candidate[-1] not in ".!?":
        candidate += "."
    return candidate or default


def clip_words(text: str, max_words: int) -> str:
    return compact_complete_text(text, max_words, clean_text(text))


def clip_words_plain(text: str, max_words: int) -> str:
    words = clean_text(text).split()
    return " ".join(words[:max_words]).rstrip(" ,;:.")


def finalize_evidence_sentence(text: str, max_words: int = 28) -> str:
    cleaned = strip_leading_label(text)
    if not cleaned or cleaned.casefold() == DEFAULT_EVIDENCE.casefold():
        return DEFAULT_EVIDENCE

    return compact_complete_text(cleaned.replace("...", "."), max_words, DEFAULT_EVIDENCE)


def finalize_step1_segment(text: str, max_words: int = 60) -> str:
    cleaned = strip_leading_label(text).replace("...", ".")
    if not cleaned:
        return ""
    sentences = split_sentences(cleaned)
    candidate = clean_text(sentences[0] if sentences else cleaned)
    return candidate.rstrip(".")


def is_artifact_evidence(text: str) -> bool:
    normalized = clean_text(text).casefold()
    if not normalized or normalized in {DEFAULT_EVIDENCE.casefold(), "none."}:
        return False
    if any(marker in normalized for marker in NEGATIVE_EVIDENCE_MARKERS):
        return False
    if any(marker in normalized for marker in ARTIFACT_EVIDENCE_MARKERS):
        return True
    return len(normalized.split()) >= 6


def stage1_image_only_policy(criterion: str) -> Tuple[str, int]:
    if criterion in LPCVC_ADDED_CRITERIA:
        return "broad", 4
    if criterion in HYBRID_CRITERIA:
        return "moderate", 5
    if criterion in LOW_COVERAGE_HOLMES_STRONG_CRITERIA:
        if criterion == "Human & Biological Structure Integrity":
            return "moderate", 5
        return "limited", 7
    if criterion in HOLMES_PRIMARY_CRITERIA:
        return "strict", 9
    if criterion in LOW_RISK_CRITERIA:
        return "broad", 4
    return "limited", 7


def is_valid_material_object_evidence(text: str) -> bool:
    normalized = clean_text(text).casefold()
    if not normalized or normalized == DEFAULT_EVIDENCE.casefold():
        return False
    if any(marker in normalized for marker in MATERIAL_OBJECT_EXCLUSION_MARKERS):
        return False
    return any(marker in normalized for marker in MATERIAL_OBJECT_POSITIVE_MARKERS)


def artifact_implied_positive(record: HolmesRecord, support_type: str, evidence: str, holmes_span: str) -> bool:
    if record.label != FAKE_LABEL:
        return False
    evidence_lower = clean_text(f"{evidence} {holmes_span}").casefold()
    if not evidence_lower:
        return False
    if any(marker in evidence_lower for marker in NEGATIVE_EVIDENCE_MARKERS):
        return False
    if support_type in {"explicit_holmes", "implied_holmes"}:
        return True
    if support_type == "image_only" and any(marker in evidence_lower for marker in ARTIFACT_EVIDENCE_MARKERS):
        return True
    return False


def export_consistency_check(
    record: HolmesRecord,
    internal_entries: Sequence[Dict[str, object]],
    stats: Counter,
) -> None:
    for item in internal_entries:
        criterion = str(item["criterion"])
        final_score = 1 if int(item.get("final_score", 0)) else 0
        judge_verdict = str(item.get("judge_verdict", ""))
        support_type = str(item.get("support_type", "unsupported"))
        evidence = str(item.get("evidence", DEFAULT_EVIDENCE))

        if finalize_evidence_sentence(evidence) == DEFAULT_EVIDENCE and final_score != 0:
            item["final_score"] = 0
            stats["export_consistency_fix_count"] += 1

        if judge_verdict == "downgrade_to_0" and int(item.get("final_score", 0)) != 0:
            item["final_score"] = 0
            stats["export_consistency_fix_count"] += 1

        if (
            record.label == FAKE_LABEL
            and judge_verdict in {"accept", "needs_specialist_check"}
            and support_type in {"explicit_holmes", "implied_holmes"}
            and is_artifact_evidence(evidence)
            and int(item.get("final_score", 0)) == 0
        ):
            item["final_score"] = 1
            stats["export_consistency_fix_count"] += 1

        if record.label == REAL_LABEL and int(item.get("final_score", 0)) != 0:
            item["final_score"] = 0
            stats["export_consistency_fix_count"] += 1


def derive_label_from_member(member: str) -> str:
    parts = Path(member).parts
    if "0_real" in parts:
        return REAL_LABEL
    if "1_fake" in parts:
        return FAKE_LABEL
    raise ValueError(f"Cannot derive label from member path: {member}")


def normalize_image_member(image_ref: str) -> str:
    normalized = image_ref.strip().replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("dataset/"):
        normalized = normalized[len("dataset/") :]
    return normalized


def load_source_rows(path: Path) -> List[HolmesRecord]:
    records: List[HolmesRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_ref = row["images"][0]
            image_member = normalize_image_member(image_ref)
            label = derive_label_from_member(image_member)
            records.append(
                HolmesRecord(
                    image_member=image_member,
                    image_output_rel=f"images/{image_member}",
                    label=label,
                    original_query=row["query"].strip(),
                    original_response=row["response"].strip(),
                )
            )
    return records


def stratified_sample(records: Sequence[HolmesRecord], sample_size: int, seed: int) -> List[HolmesRecord]:
    if sample_size <= 0 or sample_size >= len(records):
        return list(records)
    rng = random.Random(seed)
    buckets: Dict[str, List[HolmesRecord]] = defaultdict(list)
    for record in records:
        buckets[record.label].append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    labels = sorted(buckets)
    per_label = sample_size // len(labels)
    remainder = sample_size % len(labels)

    sampled: List[HolmesRecord] = []
    for idx, label in enumerate(labels):
        take = per_label + (1 if idx < remainder else 0)
        sampled.extend(buckets[label][:take])
    rng.shuffle(sampled)
    return sampled


def interleave_records_by_label(records: Sequence[HolmesRecord], seed: int) -> List[HolmesRecord]:
    if not records:
        return []

    rng = random.Random(seed)
    buckets: Dict[str, List[HolmesRecord]] = defaultdict(list)
    for record in records:
        buckets[record.label].append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    labels = sorted(buckets)
    interleaved: List[HolmesRecord] = []
    while True:
        progressed = False
        for label in labels:
            bucket = buckets[label]
            if bucket:
                interleaved.append(bucket.pop())
                progressed = True
        if not progressed:
            break
    return interleaved


def materialize_image(archive: ZipFile, member: str, output_root: Path, overwrite: bool) -> Path:
    output_path = output_root / "images" / member
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as src, output_path.open("wb") as dst:
        dst.write(src.read())
    return output_path


def append_jsonl_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_stats(path: Path, stats: Counter, requested_rows: int, written_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"requested_rows": requested_rows, "written_rows": written_rows, **dict(stats)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def update_tqdm_progress(
    progress_bar: tqdm,
    label: str,
    image_rel: str,
    batch_fill: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> None:
    postfix = {
        "label": label,
        "image": Path(image_rel).name[:40],
    }
    if batch_fill is not None and batch_size is not None:
        postfix["batch"] = f"{batch_fill}/{batch_size}"
    progress_bar.set_postfix(postfix)


def normalize_step1(step1_value: object, record: HolmesRecord) -> str:
    if isinstance(step1_value, list):
        parts = [finalize_step1_segment(str(item)) for item in step1_value if clean_text(str(item))]
        text = "; ".join(part for part in parts[:3] if part)
    else:
        text = clean_text(str(step1_value or ""))
    if not text:
        if record.label == FAKE_LABEL:
            text = "Key points: Holmes explanation indicates localized artifacts that support an AI-generated label."
        else:
            text = "Key points: Holmes explanation supports a Real label without clear AIGC artifacts."
    if not text.lower().startswith("key points:"):
        text = "Key points: " + text
    prefix, body = text.split(":", 1)
    segments = [finalize_step1_segment(seg) for seg in re.split(r"[;\n]+", body) if clean_text(seg)]
    joined = "; ".join(seg for seg in segments[:3] if seg)
    return f"{prefix}: {joined}".strip()


def anchor_holmes_response(response: str, heuristic_teacher: HeuristicTeacherBackend) -> Tuple[Dict[str, Dict[str, str]], str]:
    sections = heuristic_teacher.extract_sections(response)
    description = heuristic_teacher.extract_description(response)
    anchor_map: Dict[str, Dict[str, str]] = {}
    lowered_response = response.casefold()

    for criterion in CRITERIA:
        evidences = sections.get(criterion, [])
        if evidences:
            anchor_map[criterion] = {
                "support_type": "explicit_holmes",
                "holmes_span": finalize_evidence_sentence(evidences[0], 28),
            }
            continue

        implied = ""
        implied_sentence = ""
        for alias, canonical in CRITERION_ALIASES.items():
            if canonical != criterion:
                continue
            pattern = r"(?<!\w)" + re.escape(alias.casefold()) + r"(?!\w)"
            if re.search(pattern, lowered_response):
                implied = alias
                for sentence in split_sentences(response):
                    if re.search(pattern, sentence.casefold()):
                        implied_sentence = sentence
                        break
                break
        if implied:
            anchor_map[criterion] = {
                "support_type": "implied_holmes",
                "holmes_span": finalize_evidence_sentence(implied_sentence or implied, 28),
            }
        else:
            anchor_map[criterion] = {
                "support_type": "unsupported",
                "holmes_span": "",
            }
    return anchor_map, description


def build_heuristic_generator_output(
    record: HolmesRecord,
    anchor_map: Dict[str, Dict[str, str]],
    description: str,
) -> Dict[str, object]:
    step1_segments: List[str] = []
    if description:
        step1_segments.append(finalize_step1_segment(description, 18))

    draft_entries = []
    for criterion in CRITERIA:
        anchor = anchor_map[criterion]
        support_type = anchor["support_type"]
        holmes_span = anchor["holmes_span"]
        proposed_score = 0
        evidence = DEFAULT_EVIDENCE

        if holmes_span:
            evidence = finalize_evidence_sentence(holmes_span, 28)
            if record.label == FAKE_LABEL and support_type in {"explicit_holmes", "implied_holmes"}:
                proposed_score = 1
            if len(step1_segments) < 3:
                step1_segments.append(finalize_step1_segment(evidence, 18))

        draft_entries.append(
            {
                "criterion": criterion,
                "proposed_score": proposed_score,
                "evidence": evidence,
                "support_type": support_type,
                "holmes_span": holmes_span,
            }
        )

    if not step1_segments:
        step1_segments.append("Holmes explanation mapped into LPCVC criteria with fixed label supervision.")

    return {
        "step1_target": "Key points: " + "; ".join(seg for seg in step1_segments[:3] if seg),
        "per_criterion_draft": draft_entries,
    }


def normalize_generator_output(
    generator_output: object,
    record: HolmesRecord,
    anchor_map: Dict[str, Dict[str, str]],
    stats: Counter,
    relax_image_only_candidates: bool = False,
) -> Tuple[str, List[Dict[str, object]]]:
    if not isinstance(generator_output, dict):
        stats["json_repair_count"] += 1
        generator_output = {}

    step1 = normalize_step1(generator_output.get("step1_target"), record)
    raw_list = generator_output.get("per_criterion_draft", [])
    extracted: Dict[str, Dict[str, object]] = {}

    if isinstance(raw_list, list):
        for item in raw_list:
            if not isinstance(item, dict):
                stats["json_repair_count"] += 1
                continue
            criterion = canonicalize_criterion(str(item.get("criterion", "")))
            if criterion is None:
                continue

            try:
                proposed_score = 1 if int(item.get("proposed_score", 0)) else 0
            except (TypeError, ValueError):
                proposed_score = 0
                stats["json_repair_count"] += 1

            support_type = clean_text(str(item.get("support_type", ""))).lower()
            if support_type not in VALID_SUPPORT_TYPES:
                inferred_anchor = anchor_map[criterion]
                if proposed_score and not inferred_anchor["holmes_span"] and record.label == FAKE_LABEL:
                    support_type = "image_only"
                else:
                    support_type = inferred_anchor["support_type"]
                stats["json_repair_count"] += 1

            holmes_span = finalize_evidence_sentence(str(item.get("holmes_span", "") or ""), 32)
            if support_type in {"explicit_holmes", "implied_holmes"} and not holmes_span:
                holmes_span = anchor_map[criterion]["holmes_span"]

            evidence = finalize_evidence_sentence(str(item.get("evidence", "") or ""), 28)
            if not evidence:
                evidence = finalize_evidence_sentence(holmes_span, 28) if holmes_span else DEFAULT_EVIDENCE
            if support_type == "implied_holmes" and len(clean_text(evidence).split()) < 4:
                evidence = finalize_evidence_sentence(holmes_span, 28) if holmes_span else DEFAULT_EVIDENCE
            if support_type == "unsupported":
                holmes_span = ""
                evidence = DEFAULT_EVIDENCE
                proposed_score = 0

            if support_type == "image_only" and criterion == "Material & Object Details":
                if not is_valid_material_object_evidence(evidence):
                    support_type = "unsupported"
                    holmes_span = ""
                    evidence = DEFAULT_EVIDENCE
                    proposed_score = 0
                    stats["stage1_material_boundary_downgrade_count"] += 1

            extracted[criterion] = {
                "criterion": criterion,
                "proposed_score": proposed_score,
                "evidence": evidence,
                "support_type": support_type,
                "holmes_span": holmes_span,
                "artifact_score_conflict": False,
                "non_applicable": False,
            }

    draft_entries: List[Dict[str, object]] = []
    for criterion in CRITERIA:
        if criterion in extracted:
            draft = extracted[criterion]
        else:
            anchor = anchor_map[criterion]
            default_evidence = finalize_evidence_sentence(anchor["holmes_span"], 28) if anchor["holmes_span"] else DEFAULT_EVIDENCE
            if anchor["support_type"] == "implied_holmes" and len(clean_text(default_evidence).split()) < 4:
                default_evidence = DEFAULT_EVIDENCE
            draft = {
                "criterion": criterion,
                "proposed_score": 0,
                "evidence": default_evidence,
                "support_type": anchor["support_type"],
                "holmes_span": anchor["holmes_span"],
                "artifact_score_conflict": False,
                "non_applicable": default_evidence == DEFAULT_EVIDENCE,
            }
            stats["criterion_fill_in_count"] += 1

        if record.label == REAL_LABEL and draft["support_type"] == "image_only":
            draft["proposed_score"] = 0
            draft["support_type"] = "unsupported"
            draft["holmes_span"] = ""
            draft["evidence"] = DEFAULT_EVIDENCE
            draft["non_applicable"] = True
            stats["real_image_only_block_count"] += 1

        evidence_has_negative_marker = any(
            marker in str(draft["evidence"]).casefold() for marker in NEGATIVE_EVIDENCE_MARKERS
        )
        if evidence_has_negative_marker:
            draft["proposed_score"] = 0
            draft["non_applicable"] = True

        evidence_supports_positive = artifact_implied_positive(
            record,
            str(draft["support_type"]),
            str(draft["evidence"]),
            str(draft["holmes_span"]),
        )
        unsupported_image_candidate = (
            record.label == FAKE_LABEL
            and draft["support_type"] == "unsupported"
            and not clean_text(str(draft["holmes_span"]))
            and is_artifact_evidence(str(draft["evidence"]))
        )
        evidence_words = len(clean_text(str(draft["evidence"])).split())
        if (
            relax_image_only_candidates
            and record.label == FAKE_LABEL
            and draft["support_type"] == "unsupported"
            and not clean_text(str(draft["holmes_span"]))
            and unsupported_image_candidate
        ):
            policy, min_words = stage1_image_only_policy(criterion)
            material_ok = (
                criterion != "Material & Object Details"
                or is_valid_material_object_evidence(str(draft["evidence"]))
            )
            if evidence_words >= min_words and material_ok:
                draft["support_type"] = "image_only"
                draft["non_applicable"] = False
                stats["stage1_image_only_promotion_count"] += 1
                stats[f"stage1_image_only_promotion_{policy}_count"] += 1
            elif criterion == "Material & Object Details" and evidence_words >= min_words:
                stats["stage1_material_boundary_block_count"] += 1

        if draft["support_type"] == "unsupported":
            draft["holmes_span"] = ""
            draft["evidence"] = DEFAULT_EVIDENCE
            draft["proposed_score"] = 0
            draft["non_applicable"] = True

        conflict_positive_signal = evidence_supports_positive or (
            draft["support_type"] == "image_only" and is_artifact_evidence(str(draft["evidence"]))
        )
        draft["artifact_score_conflict"] = bool(
            (record.label == FAKE_LABEL and conflict_positive_signal and int(draft["proposed_score"]) == 0)
            or (int(draft["proposed_score"]) == 1 and (draft["non_applicable"] or draft["support_type"] == "unsupported"))
        )

        draft_entries.append(draft)

    return step1, draft_entries


def heuristic_judge_review(record: HolmesRecord, draft_entries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    reviews = []
    seen_positive_keys = set()

    for item in draft_entries:
        criterion = str(item["criterion"])
        proposed_score = 1 if int(item.get("proposed_score", 0)) else 0
        support_type = str(item.get("support_type", "unsupported"))
        holmes_span = clean_text(str(item.get("holmes_span", "")))
        evidence = clean_text(str(item.get("evidence", "")))
        key = clean_text((holmes_span or evidence).casefold())
        artifact_conflict = bool(item.get("artifact_score_conflict", False))
        non_applicable = bool(item.get("non_applicable", False))

        verdict = "downgrade_to_0"
        reason = "Proposed criterion is not sufficiently supported."
        score_consistency = "consistent"
        recommended_score = "keep"

        if proposed_score == 0:
            verdict = "accept"
            reason = "Negative or neutral entry kept as non-positive evidence."
        elif record.label == REAL_LABEL:
            verdict = "downgrade_to_0"
            reason = "Real samples do not allow positive AIGC criteria."
            score_consistency = "inconsistent"
            recommended_score = "set_to_0"
        elif support_type == "explicit_holmes":
            verdict = "accept"
            reason = "Directly supported by Holmes wording."
        elif support_type == "implied_holmes":
            verdict = "accept" if holmes_span else "downgrade_to_0"
            reason = (
                "Holmes wording strongly implies this criterion."
                if verdict == "accept"
                else "Implied support is too weak."
            )
        elif support_type == "image_only":
            if criterion in HIGH_RISK_CRITERIA:
                verdict = "needs_specialist_check"
                reason = "Image-only positive on a high-risk criterion requires specialist review."
            elif evidence and len(evidence.split()) >= 4:
                verdict = "accept"
                reason = "Image-only issue is specific enough for a low-risk criterion."
            else:
                verdict = "downgrade_to_0"
                reason = "Image-only evidence is too generic."
        elif any(marker in evidence.casefold() for marker in ARTIFACT_EVIDENCE_MARKERS):
            verdict = "accept"
            reason = "Evidence includes concrete artifact language."

        if artifact_conflict:
            score_consistency = "inconsistent"
            if record.label == FAKE_LABEL and support_type in {"explicit_holmes", "implied_holmes"} and verdict == "accept":
                recommended_score = "set_to_1"
            else:
                recommended_score = "set_to_0"
        elif non_applicable and proposed_score != 0:
            score_consistency = "inconsistent"
            recommended_score = "set_to_0"
        elif verdict == "needs_specialist_check":
            recommended_score = "defer_to_specialist"

        if proposed_score == 1 and verdict == "accept" and key:
            if key in seen_positive_keys:
                verdict = "downgrade_to_0"
                reason = "Redundant with another accepted positive criterion."
                score_consistency = "inconsistent"
                recommended_score = "set_to_0"
            else:
                seen_positive_keys.add(key)

        reviews.append(
            {
                "criterion": criterion,
                "verdict": verdict,
                "reason": reason,
                "score_consistency": score_consistency,
                "recommended_score": recommended_score,
            }
        )

    return {"per_criterion_review": reviews}


def normalize_judge_output(
    judge_output: object,
    draft_entries: Sequence[Dict[str, object]],
    stats: Counter,
) -> Dict[str, Dict[str, str]]:
    if not isinstance(judge_output, dict):
        stats["json_repair_count"] += 1
        judge_output = {}

    extracted: Dict[str, Dict[str, str]] = {}
    raw_list = judge_output.get("per_criterion_review", [])
    if isinstance(raw_list, list):
        for item in raw_list:
            if not isinstance(item, dict):
                stats["json_repair_count"] += 1
                continue
            criterion = canonicalize_criterion(str(item.get("criterion", "")))
            if criterion is None:
                continue
            verdict = clean_text(str(item.get("verdict", ""))).lower()
            if verdict not in VALID_JUDGE_VERDICTS:
                verdict = "downgrade_to_0"
                stats["json_repair_count"] += 1
            reason = clip_words(str(item.get("reason", "") or "No judge reason provided."), 25)
            score_consistency = clean_text(str(item.get("score_consistency", "consistent"))).lower()
            if score_consistency not in VALID_SCORE_CONSISTENCY:
                score_consistency = "consistent"
                stats["json_repair_count"] += 1
            recommended_score = clean_text(str(item.get("recommended_score", "keep"))).lower()
            if recommended_score not in VALID_RECOMMENDED_SCORE:
                recommended_score = "keep"
                stats["json_repair_count"] += 1
            extracted[criterion] = {
                "verdict": verdict,
                "reason": reason,
                "score_consistency": score_consistency,
                "recommended_score": recommended_score,
            }

    review_map: Dict[str, Dict[str, str]] = {}
    for draft in draft_entries:
        criterion = str(draft["criterion"])
        if criterion in extracted:
            review_map[criterion] = extracted[criterion]
        else:
            fallback_verdict = "accept" if int(draft.get("proposed_score", 0)) == 0 else "downgrade_to_0"
            if fallback_verdict == "downgrade_to_0":
                stats["judge_missing_review_count"] += 1
            review_map[criterion] = {
                "verdict": fallback_verdict,
                "reason": "Fallback review applied during normalization.",
                "score_consistency": "consistent",
                "recommended_score": "keep" if fallback_verdict == "accept" else "set_to_0",
            }
    return review_map


def heuristic_specialist_review(criterion: str, draft_entry: Dict[str, object]) -> Dict[str, object]:
    evidence = clip_words(str(draft_entry.get("evidence", "") or DEFAULT_EVIDENCE), 25)
    supported = bool(clean_text(str(draft_entry.get("holmes_span", ""))))
    verdict = "accept" if supported else "downgrade_to_0"
    reason = (
        "Specialist accepted because Holmes provides criterion-specific support."
        if supported
        else "Specialist downgraded because support is image-only or too weak."
    )
    return {
        "criterion": criterion,
        "verdict": verdict,
        "reason": reason,
        "evidence": evidence,
        "confidence": "high" if supported else "medium",
    }


def finalize_step2(
    record: HolmesRecord,
    draft_entries: Sequence[Dict[str, object]],
    review_map: Dict[str, Dict[str, str]],
    specialist_map: Dict[str, Dict[str, object]],
    stats: Counter,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    internal_entries: List[Dict[str, object]] = []
    any_positive = False
    image_only_positive_count = 0
    seen_positive_keys = set()

    for draft in draft_entries:
        criterion = str(draft["criterion"])
        proposed_score = 1 if int(draft.get("proposed_score", 0)) else 0
        support_type = str(draft.get("support_type", "unsupported"))
        holmes_span = clip_words(str(draft.get("holmes_span", "") or ""), 30)
        evidence = clip_words(str(draft.get("evidence", "") or DEFAULT_EVIDENCE), 25)
        review = review_map.get(criterion, {"verdict": "downgrade_to_0", "reason": "Missing review."})
        verdict = review["verdict"]
        judge_reason = review["reason"]
        score_consistency = review.get("score_consistency", "consistent")
        recommended_score = review.get("recommended_score", "keep")
        specialist = specialist_map.get(criterion)
        specialist_used = specialist is not None
        specialist_verdict = None
        specialist_confidence = None

        if specialist is not None:
            specialist_verdict = clean_text(str(specialist.get("verdict", ""))).lower() or None
            candidate_conf = clean_text(str(specialist.get("confidence", ""))).lower()
            specialist_confidence = candidate_conf if candidate_conf in VALID_SPECIALIST_CONFIDENCE else None
            judge_reason = clip_words(str(specialist.get("reason", "") or judge_reason), 25)
            if specialist.get("evidence"):
                evidence = clip_words(str(specialist["evidence"]), 25)

        final_score = 0
        if record.label == REAL_LABEL:
            if holmes_span or support_type in {"explicit_holmes", "implied_holmes"}:
                stats["real_supported_zero_count"] += 1
            else:
                evidence = DEFAULT_EVIDENCE
        else:
            evidence_supports_positive = artifact_implied_positive(record, support_type, evidence, holmes_span)
            if recommended_score == "set_to_0":
                final_score = 0
            elif recommended_score == "set_to_1" and verdict == "accept" and evidence_supports_positive:
                final_score = 1
            elif verdict == "accept" and (proposed_score == 1 or evidence_supports_positive):
                final_score = 1
            if verdict == "needs_specialist_check":
                final_score = 1 if specialist_verdict == "accept" and (proposed_score == 1 or evidence_supports_positive) else 0
                if specialist_confidence == "low":
                    final_score = 0
            if support_type == "image_only" and final_score == 1:
                image_only_positive_count += 1
                if image_only_positive_count > 1:
                    final_score = 0
                    judge_reason = "Image-only positives are capped at one per fake sample."
                    stats["image_only_cap_downgrade_count"] += 1
            if support_type == "unsupported":
                final_score = 0
            if any(marker in evidence.casefold() for marker in NEGATIVE_EVIDENCE_MARKERS):
                final_score = 0

            positive_key = clean_text((holmes_span or evidence).casefold())
            if final_score == 1 and positive_key:
                if positive_key in seen_positive_keys:
                    final_score = 0
                    judge_reason = "Redundant with another accepted positive criterion."
                    stats["redundant_positive_downgrade_count"] += 1
                else:
                    seen_positive_keys.add(positive_key)

            if final_score == 1:
                any_positive = True
            elif not holmes_span and support_type in {"unsupported", "image_only"}:
                evidence = DEFAULT_EVIDENCE

        final_evidence = finalize_evidence_sentence(evidence)
        if verdict == "downgrade_to_0" and proposed_score == 1:
            stats["judge_downgrade_count"] += 1
        if verdict == "needs_specialist_check":
            stats["specialist_check_count"] += 1
        if support_type == "image_only" and proposed_score == 1:
            stats["image_only_positive_count"] += 1

        internal_entries.append(
            {
                "criterion": criterion,
                "proposed_score": proposed_score,
                "evidence": final_evidence,
                "support_type": support_type,
                "holmes_span": holmes_span,
                "artifact_score_conflict": bool(draft.get("artifact_score_conflict", False)),
                "non_applicable": bool(draft.get("non_applicable", False)),
                "judge_verdict": verdict,
                "judge_reason": judge_reason,
                "score_consistency": score_consistency,
                "recommended_score": recommended_score,
                "specialist_used": specialist_used,
                "specialist_verdict": specialist_verdict,
                "specialist_confidence": specialist_confidence,
                "final_score": final_score,
            }
        )
    if record.label == FAKE_LABEL and not any_positive:
        for item in internal_entries:
            if item["support_type"] in {"explicit_holmes", "implied_holmes"} and item["holmes_span"]:
                item["judge_verdict"] = "accept"
                item["judge_reason"] = "Fallback positive restored from Holmes-supported evidence."
                item["final_score"] = 1
                any_positive = True
                stats["fake_positive_fallback_count"] += 1
                break

    export_consistency_check(record, internal_entries, stats)

    official_entries = [
        {
            "criterion": str(item["criterion"]),
            "score": 1 if int(item.get("final_score", 0)) else 0,
            "evidence": finalize_evidence_sentence(str(item.get("evidence", DEFAULT_EVIDENCE))),
        }
        for item in internal_entries
    ]

    internal_step2 = {"overall_likelihood": record.label, "per_criterion_draft": internal_entries}
    official_step2 = {"overall_likelihood": record.label, "per_criterion": official_entries}
    stats["official_export_rows"] += 1
    return internal_step2, official_step2


def convert_records(
    records: Sequence[HolmesRecord],
    archive: ZipFile,
    output_root: Path,
    output_jsonl: Path,
    stats_path: Path,
    backends: TeacherBackends,
    overwrite_images: bool,
    judge_enabled: bool,
    specialist_enabled: bool,
) -> Tuple[int, Counter]:
    stats: Counter = Counter()
    heuristic_teacher = HeuristicTeacherBackend()
    archive_members = set(archive.namelist())
    batch_size = max(1, getattr(backends.generator, "batch_size", 1))
    requested_rows = len(records)

    progress_bar = tqdm(total=requested_rows, desc="full", unit="img", file=sys.stderr, dynamic_ncols=True)

    def process_record(idx: int, record: HolmesRecord, image_path: Path) -> None:
        update_tqdm_progress(progress_bar, record.label, record.image_output_rel)
        anchor_map, description = anchor_holmes_response(record.original_response, heuristic_teacher)
        generator_prompt = build_generator_prompt_v2(record, anchor_map, description)

        if isinstance(backends.generator, HeuristicTeacherBackend):
            generator_output = build_heuristic_generator_output(record, anchor_map, description)
        else:
            try:
                generator_output = backends.generator.generate_json_prompt(
                    generator_prompt, image_path, GENERATOR_SYSTEM_PROMPT
                )
            except Exception as exc:  # noqa: BLE001
                stats["teacher_error_count"] += 1
                stats["json_repair_count"] += 1
                generator_output = build_heuristic_generator_output(record, anchor_map, description)
                print(f"[warn] generator fallback on row {idx}: {exc}", file=sys.stderr)

        step1, draft_entries = normalize_generator_output(generator_output, record, anchor_map, stats)

        if judge_enabled:
            if isinstance(backends.judge, HeuristicTeacherBackend):
                judge_output = heuristic_judge_review(record, draft_entries)
            else:
                judge_prompt = build_judge_prompt(record, draft_entries)
                try:
                    judge_output = backends.judge.generate_json_prompt(
                        judge_prompt, image_path, JUDGE_SYSTEM_PROMPT
                    )
                except Exception as exc:  # noqa: BLE001
                    stats["judge_error_count"] += 1
                    stats["json_repair_count"] += 1
                    judge_output = heuristic_judge_review(record, draft_entries)
                    print(f"[warn] judge fallback on row {idx}: {exc}", file=sys.stderr)
        else:
            judge_output = heuristic_judge_review(record, draft_entries)

        review_map = normalize_judge_output(judge_output, draft_entries, stats)
        specialist_map: Dict[str, Dict[str, object]] = {}
        if specialist_enabled:
            for draft in draft_entries:
                criterion = str(draft["criterion"])
                review = review_map.get(criterion, {})
                if review.get("verdict") != "needs_specialist_check":
                    continue
                if isinstance(backends.specialist, HeuristicTeacherBackend):
                    specialist_map[criterion] = heuristic_specialist_review(criterion, draft)
                else:
                    specialist_prompt = build_specialist_prompt(record, criterion, draft)
                    try:
                        specialist_map[criterion] = backends.specialist.generate_json_prompt(
                            specialist_prompt, image_path, SPECIALIST_SYSTEM_PROMPT
                        )
                    except Exception as exc:  # noqa: BLE001
                        stats["specialist_error_count"] += 1
                        stats["json_repair_count"] += 1
                        specialist_map[criterion] = heuristic_specialist_review(criterion, draft)
                        print(f"[warn] specialist fallback on row {idx}: {exc}", file=sys.stderr)

        internal_step2, official_step2 = finalize_step2(record, draft_entries, review_map, specialist_map, stats)

        row = {
            "image": record.image_output_rel,
            "source": record.source,
            "original_query": record.original_query,
            "original_response": record.original_response,
            "step1_target": step1,
            "step2_target": official_step2,
            "step2_internal": internal_step2,
        }
        append_jsonl_row(output_jsonl, row)
        stats["processed_rows"] += 1
        save_stats(stats_path, stats, requested_rows=requested_rows, written_rows=stats["processed_rows"])
        progress_bar.update(1)

    pending_batch: List[Tuple[int, HolmesRecord, Path]] = []

    def flush_batch(batch_items: List[Tuple[int, HolmesRecord, Path]]) -> None:
        if not batch_items:
            return
        for idx, record, image_path in batch_items:
            process_record(idx, record, image_path)

    for idx, record in enumerate(records, start=1):
        if record.image_member not in archive_members:
            stats["missing_image_rows"] += 1
            continue
        image_path = materialize_image(archive, record.image_member, output_root, overwrite_images)
        pending_batch.append((idx, record, image_path))
        update_tqdm_progress(progress_bar, record.label, record.image_output_rel, len(pending_batch), batch_size)
        if len(pending_batch) >= batch_size:
            flush_batch(pending_batch)
            pending_batch = []

    flush_batch(pending_batch)
    progress_bar.close()
    return stats["processed_rows"], stats


def convert_records_generator_only(
    records: Sequence[HolmesRecord],
    archive: ZipFile,
    output_root: Path,
    output_jsonl: Path,
    stats_path: Path,
    backends: TeacherBackends,
    overwrite_images: bool,
) -> Tuple[int, Counter]:
    stats: Counter = Counter()
    heuristic_teacher = HeuristicTeacherBackend()
    archive_members = set(archive.namelist())
    batch_size = max(1, getattr(backends.generator, "batch_size", 1))
    requested_rows = len(records)

    progress_bar = tqdm(
        total=requested_rows,
        desc="generator_only",
        unit="img",
        file=sys.stderr,
        dynamic_ncols=True,
    )

    def process_record_result(
        idx: int,
        record: HolmesRecord,
        anchor_map: Dict[str, Dict[str, str]],
        generator_output: Dict[str, object],
    ) -> None:
        step1, draft_entries = normalize_generator_output(
            generator_output,
            record,
            anchor_map,
            stats,
            relax_image_only_candidates=True,
        )
        row = {
            "image": record.image_output_rel,
            "source": record.source,
            "original_query": record.original_query,
            "original_response": record.original_response,
            "step1_target": step1,
            "step2_draft": {
                "overall_likelihood": record.label,
                "per_criterion_draft": draft_entries,
            },
        }
        append_jsonl_row(output_jsonl, row)
        stats["processed_rows"] += 1
        stats["generator_only_rows"] += 1
        save_stats(
            stats_path,
            stats,
            requested_rows=requested_rows,
            written_rows=stats["processed_rows"],
        )
        progress_bar.update(1)

    def flush_batch(batch_items: List[Tuple[int, HolmesRecord, Path]]) -> None:
        if not batch_items:
            return

        prompts = []
        image_paths = []
        anchor_maps = []
        records_in_batch = []

        for idx, record, image_path in batch_items:
            update_tqdm_progress(
                progress_bar,
                record.label,
                record.image_output_rel,
                len(records_in_batch) + 1,
                batch_size,
            )
            anchor_map, description = anchor_holmes_response(
                record.original_response,
                heuristic_teacher,
            )
            generator_prompt = build_generator_prompt_v2(
                record,
                anchor_map,
                description,
                relax_image_only_candidates=True,
            )

            prompts.append(generator_prompt)
            image_paths.append(image_path)
            anchor_maps.append(anchor_map)
            records_in_batch.append((idx, record))

        if isinstance(backends.generator, HeuristicTeacherBackend):
            generator_outputs = []
            for (_, record), anchor_map in zip(records_in_batch, anchor_maps):
                description = ""  # heuristic fallback 用不到實際 batch image
                generator_outputs.append(
                    build_heuristic_generator_output(record, anchor_map, description)
                )
        else:
            try:
                generator_outputs = backends.generator.generate_json_prompts_batch(
                    prompts,
                    image_paths,
                    GENERATOR_SYSTEM_PROMPT,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] batch generator retry as single items: {exc}", file=sys.stderr)
                generator_outputs = []
                stats["batch_retry_count"] += 1
                for prompt, image_path, (_, record), anchor_map in zip(
                    prompts,
                    image_paths,
                    records_in_batch,
                    anchor_maps,
                ):
                    raw_text: Optional[str] = None
                    try:
                        if hasattr(backends.generator, "generate_json_prompt_with_raw"):
                            result, raw_text = backends.generator.generate_json_prompt_with_raw(
                                prompt,
                                image_path,
                                GENERATOR_SYSTEM_PROMPT,
                            )
                        else:
                            result = backends.generator.generate_json_prompt(
                                prompt,
                                image_path,
                                GENERATOR_SYSTEM_PROMPT,
                            )
                        generator_outputs.append(result)
                    except Exception as single_exc:  # noqa: BLE001
                        stats["teacher_error_count"] += 1
                        stats["json_repair_count"] += 1
                        if raw_text:
                            save_raw_teacher_output(
                                output_root,
                                "generator",
                                record.image_output_rel,
                                raw_text,
                                single_exc,
                                "single_retry",
                            )
                        print(
                            f"[warn] generator single-item fallback on image {record.image_output_rel}: {single_exc}",
                            file=sys.stderr,
                        )
                        generator_outputs.append(
                            build_heuristic_generator_output(record, anchor_map, "")
                        )

        if len(generator_outputs) != len(records_in_batch):
            print(
                f"[warn] batch output size mismatch: got {len(generator_outputs)} "
                f"for batch of {len(records_in_batch)}",
                file=sys.stderr,
            )
            fixed_outputs = []
            for i, ((_, record), anchor_map) in enumerate(zip(records_in_batch, anchor_maps)):
                if i < len(generator_outputs):
                    fixed_outputs.append(generator_outputs[i])
                else:
                    stats["teacher_error_count"] += 1
                    stats["json_repair_count"] += 1
                    fixed_outputs.append(
                        build_heuristic_generator_output(record, anchor_map, "")
                    )
            generator_outputs = fixed_outputs

        for (idx, record), anchor_map, generator_output in zip(
            records_in_batch,
            anchor_maps,
            generator_outputs,
        ):
            process_record_result(idx, record, anchor_map, generator_output)

    pending_batch: List[Tuple[int, HolmesRecord, Path]] = []

    for idx, record in enumerate(records, start=1):
        if record.image_member not in archive_members:
            stats["missing_image_rows"] += 1
            continue

        image_path = materialize_image(
            archive,
            record.image_member,
            output_root,
            overwrite_images,
        )
        pending_batch.append((idx, record, image_path))
        update_tqdm_progress(
            progress_bar,
            record.label,
            record.image_output_rel,
            len(pending_batch),
            batch_size,
        )

        if len(pending_batch) >= batch_size:
            flush_batch(pending_batch)
            pending_batch = []

    flush_batch(pending_batch)
    progress_bar.close()
    return stats["processed_rows"], stats


def convert_draft_rows_review_only(
    draft_rows: Sequence[Dict[str, object]],
    output_root: Path,
    output_jsonl: Path,
    stats_path: Path,
    backends: TeacherBackends,
    judge_enabled: bool,
    specialist_enabled: bool,
    image_root: Optional[Path] = None,
) -> Tuple[int, Counter]:
    stats: Counter = Counter()
    requested_rows = len(draft_rows)
    progress_bar = tqdm(total=requested_rows, desc="review_only", unit="img", file=sys.stderr, dynamic_ncols=True)

    def process_row(idx: int, row: Dict[str, object]) -> None:
        image_rel = str(row["image"])
        base_image_root = image_root or output_root
        image_path = base_image_root / image_rel
        label = str(row.get("step2_draft", {}).get("overall_likelihood", ""))
        if label not in {REAL_LABEL, FAKE_LABEL}:
            label = REAL_LABEL if "/0_real/" in image_rel or image_rel.startswith("images/0_real/") else FAKE_LABEL
        update_tqdm_progress(progress_bar, label, image_rel)
        record = HolmesRecord(
            image_member=image_rel.removeprefix("images/"),
            image_output_rel=image_rel,
            label=label,
            original_query=str(row["original_query"]),
            original_response=str(row["original_response"]),
            source=str(row.get("source", SOURCE_NAME)),
        )

        draft_entries_raw = row.get("step2_draft", {}).get("per_criterion_draft", [])
        draft_entries = normalize_judge_input_drafts(draft_entries_raw)

        if judge_enabled:
            if isinstance(backends.judge, HeuristicTeacherBackend):
                judge_output = heuristic_judge_review(record, draft_entries)
            else:
                judge_prompt = build_judge_prompt(record, draft_entries)
                try:
                    judge_output = backends.judge.generate_json_prompt(
                        judge_prompt, image_path, JUDGE_SYSTEM_PROMPT
                    )
                except Exception as exc:  # noqa: BLE001
                    stats["judge_error_count"] += 1
                    stats["json_repair_count"] += 1
                    judge_output = heuristic_judge_review(record, draft_entries)
                    print(f"[warn] judge fallback on review row {idx}: {exc}", file=sys.stderr)
        else:
            judge_output = heuristic_judge_review(record, draft_entries)

        review_map = normalize_judge_output(judge_output, draft_entries, stats)
        specialist_map: Dict[str, Dict[str, object]] = {}
        if specialist_enabled:
            for draft in draft_entries:
                criterion = str(draft["criterion"])
                review = review_map.get(criterion, {})
                if review.get("verdict") != "needs_specialist_check":
                    continue
                if isinstance(backends.specialist, HeuristicTeacherBackend):
                    specialist_map[criterion] = heuristic_specialist_review(criterion, draft)
                else:
                    specialist_prompt = build_specialist_prompt(record, criterion, draft)
                    try:
                        specialist_map[criterion] = backends.specialist.generate_json_prompt(
                            specialist_prompt, image_path, SPECIALIST_SYSTEM_PROMPT
                        )
                    except Exception as exc:  # noqa: BLE001
                        stats["specialist_error_count"] += 1
                        stats["json_repair_count"] += 1
                        specialist_map[criterion] = heuristic_specialist_review(criterion, draft)
                        print(f"[warn] specialist fallback on review row {idx}: {exc}", file=sys.stderr)

        internal_step2, official_step2 = finalize_step2(record, draft_entries, review_map, specialist_map, stats)
        out_row = {
            "image": image_rel,
            "source": record.source,
            "original_query": record.original_query,
            "original_response": record.original_response,
            "step1_target": str(row.get("step1_target", "")),
            "step2_target": official_step2,
            "step2_internal": internal_step2,
        }
        append_jsonl_row(output_jsonl, out_row)
        stats["processed_rows"] += 1
        stats["review_only_rows"] += 1
        save_stats(stats_path, stats, requested_rows=requested_rows, written_rows=stats["processed_rows"])
        progress_bar.update(1)

    for idx, row in enumerate(draft_rows, start=1):
        process_row(idx, row)
    progress_bar.close()
    return stats["processed_rows"], stats


def normalize_judge_input_drafts(draft_entries_raw: object) -> List[Dict[str, object]]:
    drafts: List[Dict[str, object]] = []
    if not isinstance(draft_entries_raw, list):
        return drafts
    for item in draft_entries_raw:
        if not isinstance(item, dict):
            continue
        criterion = canonicalize_criterion(str(item.get("criterion", "")))
        if criterion is None:
            continue
        support_type = clean_text(str(item.get("support_type", "unsupported"))).lower()
        if support_type not in VALID_SUPPORT_TYPES:
            support_type = "unsupported"
        proposed_score = 1 if int(item.get("proposed_score", 0) or 0) else 0
        evidence = clean_text(str(item.get("evidence", DEFAULT_EVIDENCE))) or DEFAULT_EVIDENCE
        holmes_span = clean_text(str(item.get("holmes_span", "")))
        drafts.append(
            {
                "criterion": criterion,
                "proposed_score": proposed_score,
                "evidence": evidence,
                "support_type": support_type,
                "holmes_span": holmes_span,
                "artifact_score_conflict": bool(item.get("artifact_score_conflict", False)),
                "non_applicable": bool(item.get("non_applicable", False)),
            }
        )
    seen = {str(d["criterion"]) for d in drafts}
    for criterion in CRITERIA:
        if criterion not in seen:
            drafts.append(
                {
                    "criterion": criterion,
                    "proposed_score": 0,
                    "evidence": DEFAULT_EVIDENCE,
                    "support_type": "unsupported",
                    "holmes_span": "",
                    "artifact_score_conflict": False,
                    "non_applicable": True,
                }
            )
    ordered = {str(d["criterion"]): d for d in drafts}
    return [ordered[c] for c in CRITERIA]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Holmes SFT data to LPCVC3-style JSONL.")
    parser.add_argument("--holmes-root", type=Path, default=Path("/raid/ron/LPCVC/dataset/holmes"))
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--dataset-archive", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("/raid/ron/LPCVC/holmes_lpcvc4/output"))
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--draft-jsonl", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pipeline-stage",
        choices=["full", "generator_only", "review_only"],
        default="full",
    )
    parser.add_argument(
        "--teacher-backend",
        choices=["heuristic", "openai_compatible", "transformers_gemma4"],
        default="heuristic",
    )
    parser.add_argument("--api-base", type=str, default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--judge-api-base", type=str, default=os.environ.get("OPENAI_JUDGE_API_BASE", ""))
    parser.add_argument("--specialist-api-base", type=str, default=os.environ.get("OPENAI_SPECIALIST_API_BASE", ""))
    parser.add_argument("--model", type=str, default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--judge-model", type=str, default=os.environ.get("OPENAI_JUDGE_MODEL", ""))
    parser.add_argument("--specialist-model", type=str, default=os.environ.get("OPENAI_SPECIALIST_MODEL", ""))
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--judge-max-tokens", type=int, default=500)
    parser.add_argument("--specialist-max-tokens", type=int, default=300)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--judge-max-new-tokens", type=int, default=500)
    parser.add_argument("--specialist-max-new-tokens", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--balance-label-order",
        action="store_true",
        help="Interleave Real and AI-Generated records before conversion so early progress is more balanced.",
    )
    parser.add_argument("--disable-judge", action="store_true")
    parser.add_argument("--enable-specialist", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    return parser.parse_args()


def build_backends(args: argparse.Namespace) -> TeacherBackends:
    if args.teacher_backend == "heuristic":
        backend = HeuristicTeacherBackend()
        return TeacherBackends(generator=backend, judge=backend, specialist=backend)

    cache: Dict[Tuple[object, ...], TeacherBackend] = {}

    def get_backend(role: str) -> TeacherBackend:
        if args.teacher_backend == "transformers_gemma4":
            default_model = args.model or "google/gemma-4-e2b-it"
            model_name = {
                "generator": default_model,
                "judge": args.judge_model or default_model,
                "specialist": args.specialist_model or args.judge_model or default_model,
            }[role]
            max_new_tokens = {
                "generator": args.max_new_tokens,
                "judge": args.judge_max_new_tokens,
                "specialist": args.specialist_max_new_tokens,
            }[role]
            batch_size = args.batch_size if role == "generator" else 1
            key = (
                "transformers_gemma4",
                model_name,
                args.device,
                args.torch_dtype,
                max_new_tokens,
                args.temperature,
                batch_size,
            )
            if key not in cache:
                cache[key] = TransformersGemma4TeacherBackend(
                    model_name=model_name,
                    device=args.device,
                    torch_dtype=args.torch_dtype,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                    batch_size=batch_size,
                )
            return cache[key]

        api_key = os.environ.get(args.api_key_env)
        default_model = args.model
        default_api_base = args.api_base
        model_name = {
            "generator": default_model,
            "judge": args.judge_model or default_model,
            "specialist": args.specialist_model or args.judge_model or default_model,
        }[role]
        api_base = {
            "generator": default_api_base,
            "judge": args.judge_api_base or default_api_base,
            "specialist": args.specialist_api_base or args.judge_api_base or default_api_base,
        }[role]
        if not model_name:
            raise ValueError("--model is required when using openai_compatible teacher backend")
        max_tokens = {
            "generator": args.max_tokens,
            "judge": args.judge_max_tokens,
            "specialist": args.specialist_max_tokens,
        }[role]
        key = ("openai_compatible", model_name, max_tokens, api_base, args.timeout, args.temperature)
        if key not in cache:
            cache[key] = OpenAICompatibleTeacherBackend(
                api_base=api_base,
                model=model_name,
                api_key=api_key,
                timeout=args.timeout,
                temperature=args.temperature,
                max_tokens=max_tokens,
            )
        return cache[key]

    placeholder = HeuristicTeacherBackend()
    if args.pipeline_stage == "generator_only":
        return TeacherBackends(
            generator=get_backend("generator"),
            judge=placeholder,
            specialist=placeholder,
        )
    if args.pipeline_stage == "review_only":
        return TeacherBackends(
            generator=placeholder,
            judge=get_backend("judge"),
            specialist=get_backend("specialist") if args.enable_specialist else placeholder,
        )
    return TeacherBackends(
        generator=get_backend("generator"),
        judge=get_backend("judge"),
        specialist=get_backend("specialist"),
    )


def main() -> int:
    args = parse_args()
    holmes_root = args.holmes_root
    input_jsonl = args.input_jsonl or holmes_root / "SFTDATA.jsonl"
    dataset_archive = args.dataset_archive or holmes_root / "dataset_huggingface.zip"
    output_root = args.output_root
    output_jsonl = args.output_jsonl or output_root / "holmes_lpcvc_sft.jsonl"
    stats_path = args.stats_path or output_root / "stats.json"

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("", encoding="utf-8")

    backends = build_backends(args)
    if args.pipeline_stage == "review_only":
        draft_jsonl = args.draft_jsonl or input_jsonl
        draft_rows = read_jsonl_rows(draft_jsonl)
        if args.max_samples and args.max_samples > 0:
            draft_rows = draft_rows[: args.max_samples]
        written_rows, stats = convert_draft_rows_review_only(
            draft_rows=draft_rows,
            output_root=output_root,
            output_jsonl=output_jsonl,
            stats_path=stats_path,
            backends=backends,
            judge_enabled=not args.disable_judge,
            specialist_enabled=args.enable_specialist,
            image_root=draft_jsonl.parent,
        )
        requested_count = len(draft_rows)
    else:
        records = load_source_rows(input_jsonl)
        if args.balance_label_order:
            records = interleave_records_by_label(records, args.seed)
        if args.max_samples and args.max_samples > 0:
            records = stratified_sample(records, args.max_samples, args.seed)
        with ZipFile(dataset_archive) as archive:
            if args.pipeline_stage == "generator_only":
                written_rows, stats = convert_records_generator_only(
                    records=records,
                    archive=archive,
                    output_root=output_root,
                    output_jsonl=output_jsonl,
                    stats_path=stats_path,
                    backends=backends,
                    overwrite_images=args.overwrite_images,
                )
            else:
                written_rows, stats = convert_records(
                    records=records,
                    archive=archive,
                    output_root=output_root,
                    output_jsonl=output_jsonl,
                    stats_path=stats_path,
                    backends=backends,
                    overwrite_images=args.overwrite_images,
                    judge_enabled=not args.disable_judge,
                    specialist_enabled=args.enable_specialist,
                )
        requested_count = len(records)

    save_stats(stats_path, stats, requested_rows=requested_count, written_rows=written_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
