import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image

from model_utils import load_image_text_model, load_processor, move_inputs_to_generation_device, prepare_generation_inputs
from task_utils import (
    DEFAULT_EVIDENCE,
    compact_json_dumps,
    compact_trace_payload,
    extract_first_json_object,
    normalize_final_prediction_payload,
    safe_json_loads,
)
from visual_expert import predict_visual_expert


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage image-authenticity inference CLI.")
    parser.add_argument("--base_model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt_dir", type=str, default=str(REPO_ROOT / "prompts"))
    parser.add_argument("--expert_path", type=str, default=None)
    parser.add_argument("--fusion_alpha", type=float, default=0.8)
    parser.add_argument("--max_new_tokens_trace", type=int, default=1536)
    parser.add_argument("--max_new_tokens_json", type=int, default=1024)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def load_model(base_model_name: str, adapter_path: str, local_files_only: bool = False):
    processor = load_processor(base_model_name, local_files_only=local_files_only)
    base_model = load_image_text_model(
        base_model_name,
        local_files_only=local_files_only,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return processor, model


def load_prompts(prompt_dir: str):
    prompt_dir = Path(prompt_dir)
    with (prompt_dir / "evidence_trace.txt").open("r", encoding="utf-8") as handle:
        evidence_trace_prompt = handle.read().strip()
    with (prompt_dir / "stage2.txt").open("r", encoding="utf-8") as handle:
        final_json_prompt = handle.read().strip()
    return evidence_trace_prompt, final_json_prompt


def generate_text(model, processor, image_path: str, prompt_text: str, max_new_tokens: int):
    inputs = prepare_generation_inputs(processor, image_path, prompt_text)
    inputs = move_inputs_to_generation_device(model, inputs)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return output_text[0]


def generate_trace_payload(model, processor, image_path: str, prompt_text: str, max_new_tokens: int):
    trace_text = generate_text(model, processor, image_path, prompt_text, max_new_tokens=max_new_tokens)
    trace_json, trace_parse_error = safe_json_loads(trace_text)
    retry_used = False
    retry_budget = min(2048, max(max_new_tokens + 512, int(max_new_tokens * 1.5)))
    if not trace_json and retry_budget > max_new_tokens:
        retry_used = True
        retry_text = generate_text(model, processor, image_path, prompt_text, max_new_tokens=retry_budget)
        retry_json, retry_error = safe_json_loads(retry_text)
        if retry_json or len(retry_text) >= len(trace_text):
            trace_text = retry_text
            trace_json = retry_json
            trace_parse_error = retry_error
    if trace_json:
        trace_for_final = compact_json_dumps(compact_trace_payload(trace_json))
    else:
        trace_for_final = extract_first_json_object(trace_text)
    return trace_text, trace_json, trace_parse_error, trace_for_final, retry_used


def apply_expert_fusion(final_json: dict, expert_path: str, image_path: str, fusion_alpha: float) -> dict:
    if not expert_path or not final_json:
        return final_json
    expert_path = Path(expert_path)
    if expert_path.is_dir():
        expert_path = expert_path / "expert.pt"
    image = Image.open(image_path).convert("RGB")
    _, criterion_probs = predict_visual_expert(expert_path, image)
    per_criterion = final_json.get("per_criterion", [])
    any_positive = False
    for idx, entry in enumerate(per_criterion):
        student_score = 1 if int(entry.get("aigc score", 0) or 0) else 0
        expert_prob = float(criterion_probs[idx])
        fused_score = 1 if fusion_alpha * student_score + (1.0 - fusion_alpha) * expert_prob >= 0.5 else 0
        if student_score == 0 and fused_score == 1 and entry.get("evidence", DEFAULT_EVIDENCE) == DEFAULT_EVIDENCE:
            fused_score = 0
        entry["aigc score"] = fused_score
        any_positive = any_positive or bool(fused_score)
    final_json["overall_likelihood"] = "AI-Generated" if any_positive else "Real"
    return final_json


def main():
    args = parse_args()
    processor, model = load_model(args.base_model, args.adapter_path, local_files_only=args.local_files_only)
    evidence_trace_prompt, final_json_prompt = load_prompts(args.prompt_dir)

    evidence_trace_text, trace_json, trace_parse_error, trace_for_final, trace_retry_used = generate_trace_payload(
        model,
        processor,
        args.image_path,
        evidence_trace_prompt,
        max_new_tokens=args.max_new_tokens_trace,
    )
    final_prompt = (
        f"{final_json_prompt}\n\n"
        "Here is the structured evidence trace for this image:\n"
        f"{trace_for_final}\n\n"
        "Use the trace to synthesize the final structured decision JSON."
    )
    final_json_text = generate_text(
        model,
        processor,
        args.image_path,
        final_prompt,
        max_new_tokens=args.max_new_tokens_json,
    )

    final_json, parse_error = safe_json_loads(final_json_text)
    if final_json:
        final_json = apply_expert_fusion(final_json, args.expert_path, args.image_path, args.fusion_alpha)
        final_json = normalize_final_prediction_payload(final_json)

    result = {
        "evidence_trace_text": evidence_trace_text,
        "evidence_trace_json": trace_json,
        "evidence_trace_parse_error": trace_parse_error,
        "evidence_trace_retry_used": trace_retry_used,
        "final_json_text": final_json_text,
        "final_json": final_json,
        "parse_error": parse_error,
    }
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
