import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration

from lpcvc_utils import DEFAULT_EVIDENCE, extract_first_json_object, safe_json_loads
from visual_expert import predict_visual_expert

try:
    from qwen_vl_utils import process_vision_info
except ImportError:  # pragma: no cover
    process_vision_info = None


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage LPCVC inference CLI")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt_dir", type=str, default="/ssd4/LPCVC2026/Module-II-Final/prompts")
    parser.add_argument("--expert_path", type=str, default=None)
    parser.add_argument("--fusion_alpha", type=float, default=0.8)
    parser.add_argument("--max_new_tokens_trace", type=int, default=1024)
    parser.add_argument("--max_new_tokens_json", type=int, default=1024)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def load_model(base_model_name: str, adapter_path: str, local_files_only: bool = False):
    processor = AutoProcessor.from_pretrained(base_model_name, local_files_only=local_files_only)
    model_class = Qwen2_5_VLForConditionalGeneration if "Qwen2.5" in base_model_name else Qwen2VLForConditionalGeneration
    base_model = model_class.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
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


def prepare_inputs(processor, image_path: str, prompt_text: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if process_vision_info is not None:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text],
            images=[Image.open(image_path).convert("RGB")],
            padding=True,
            return_tensors="pt",
        )
    return inputs.to("cuda" if torch.cuda.is_available() else "cpu")


def generate_text(model, processor, image_path: str, prompt_text: str, max_new_tokens: int):
    inputs = prepare_inputs(processor, image_path, prompt_text)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return output_text[0]


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

    evidence_trace_text = generate_text(
        model,
        processor,
        args.image_path,
        evidence_trace_prompt,
        max_new_tokens=args.max_new_tokens_trace,
    )
    final_prompt = (
        f"{final_json_prompt}\n\n"
        "Here is the structured evidence trace for this image:\n"
        f"{extract_first_json_object(evidence_trace_text)}\n\n"
        "Use the trace to synthesize the final competition JSON."
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

    result = {
        "evidence_trace_text": evidence_trace_text,
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
