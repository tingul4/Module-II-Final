import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel

from detectors.holmes_clip_lora.runtime import (
    DEFAULT_THRESHOLD,
    default_checkpoint_path,
    load_detector,
    prediction_payload,
    score_single_image,
)
from utils.model_utils import (
    load_image_text_model,
    load_processor,
    move_inputs_to_generation_device,
    prepare_generation_inputs,
)
from utils.task_utils import (
    compact_json_dumps,
    compact_trace_payload,
    extract_first_json_value,
    normalize_final_prediction_payload,
    normalize_trace_prediction_payload,
    safe_json_loads,
    safe_json_loads_any,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage image-authenticity inference CLI.")
    parser.add_argument("--base_model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt_dir", type=str, default=str(REPO_ROOT / "prompts"))
    parser.add_argument("--prediction_source", choices=("student", "detector_student"), default="detector_student")
    parser.add_argument("--detector_checkpoint_path", type=str, default=default_checkpoint_path())
    parser.add_argument("--detector_clip_weights", type=str, default=None)
    parser.add_argument("--detector_threshold", type=float, default=DEFAULT_THRESHOLD)
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
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return output_text[0]


def generate_trace_payload(
    model,
    processor,
    image_path: str,
    prompt_text: str,
    max_new_tokens: int,
    retry_token_cap: int | None = None,
):
    trace_text = generate_text(model, processor, image_path, prompt_text, max_new_tokens=max_new_tokens)
    trace_payload, trace_parse_error = safe_json_loads_any(trace_text)
    trace_json = normalize_trace_prediction_payload(trace_payload)
    retry_used = False
    retry_budget = min(3072, max(max_new_tokens + 1024, int(max_new_tokens * 2.0)))
    if retry_token_cap is not None:
        retry_budget = min(retry_budget, int(retry_token_cap))
    if not trace_json and retry_budget > max_new_tokens:
        retry_used = True
        retry_text = generate_text(model, processor, image_path, prompt_text, max_new_tokens=retry_budget)
        retry_payload, retry_error = safe_json_loads_any(retry_text)
        retry_json = normalize_trace_prediction_payload(retry_payload)
        if retry_json or len(retry_text) >= len(trace_text):
            trace_text = retry_text
            trace_json = retry_json
            trace_parse_error = retry_error
    if trace_json:
        trace_for_final = compact_json_dumps(compact_trace_payload(trace_json))
    else:
        trace_for_final = extract_first_json_value(trace_text)
    return trace_text, trace_json, trace_parse_error, trace_for_final, retry_used


def build_final_prompt(final_json_prompt: str, trace_for_final: str) -> str:
    return (
        f"{final_json_prompt}\n\n"
        "Here is the structured evidence trace for this image:\n"
        f"{trace_for_final}\n\n"
        "Use the trace to synthesize the final structured decision JSON."
    )


def overlay_detector_label(final_json: dict, detector_meta: dict) -> dict:
    final_json = normalize_final_prediction_payload(final_json)
    result = dict(final_json)
    result["student_overall_likelihood"] = final_json.get("overall_likelihood")
    result["overall_likelihood"] = detector_meta["detector_label"]
    result["detector_score"] = detector_meta["detector_score"]
    result["detector_threshold"] = detector_meta["detector_threshold"]
    result["detector_label"] = detector_meta["detector_label"]
    return result


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
    final_json_text = generate_text(
        model,
        processor,
        args.image_path,
        build_final_prompt(final_json_prompt, trace_for_final),
        max_new_tokens=args.max_new_tokens_json,
    )

    final_json, parse_error = safe_json_loads(final_json_text)
    detector_meta = None
    if args.prediction_source == "detector_student":
        bundle = load_detector(
            args.detector_checkpoint_path,
            args.detector_clip_weights,
            threshold=args.detector_threshold,
        )
        score = score_single_image(bundle, args.image_path)
        detector_meta = prediction_payload(score, bundle.threshold)
    if final_json:
        final_json = normalize_final_prediction_payload(final_json)
        if args.prediction_source == "detector_student":
            final_json = overlay_detector_label(final_json, detector_meta)

    result = {
        "prediction_source": args.prediction_source,
        "evidence_trace_text": evidence_trace_text,
        "evidence_trace_json": trace_json,
        "evidence_trace_parse_error": trace_parse_error,
        "evidence_trace_retry_used": trace_retry_used,
        "final_json_text": final_json_text,
        "final_json": final_json,
        "parse_error": parse_error,
        "detector": detector_meta,
    }
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
