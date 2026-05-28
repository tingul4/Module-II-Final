from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info as _legacy_multimodal_process_vision_info
except ImportError:  # pragma: no cover
    _legacy_multimodal_process_vision_info = None


def load_processor(model_name_or_path: str, local_files_only: bool = False):
    return AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )


def load_image_text_model(
    model_name_or_path: str,
    *,
    quantization_config=None,
    device_map="auto",
    local_files_only: bool = False,
    torch_dtype=torch.bfloat16,
    use_safetensors: bool = True,
):
    common_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "torch_dtype": torch_dtype,
        "use_safetensors": use_safetensors,
    }
    if quantization_config is not None:
        common_kwargs["quantization_config"] = quantization_config
    try:
        return AutoModelForMultimodalLM.from_pretrained(
            model_name_or_path,
            attn_implementation="sdpa",
            **common_kwargs,
        )
    except Exception:
        pass
    try:
        return AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            attn_implementation="sdpa",
            **common_kwargs,
        )
    except Exception:
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation="sdpa",
            **common_kwargs,
        )


def should_use_legacy_vision_helper(processor) -> bool:
    if _legacy_multimodal_process_vision_info is None:
        return False
    processor_name = processor.__class__.__name__.casefold()
    return "qwen" in processor_name


def build_user_message(image_source, prompt_text: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_source},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def prepare_generation_inputs(processor, image_path: str, prompt_text: str):
    messages = build_user_message(image_path, prompt_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if should_use_legacy_vision_helper(processor):
        image_inputs, video_inputs = _legacy_multimodal_process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        with Image.open(image_path).convert("RGB") as image:
            inputs = processor(
                text=[text],
                images=[image.copy()],
                padding=True,
                return_tensors="pt",
            )
    return inputs


def move_inputs_to_generation_device(model, inputs):
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for _, device in model.hf_device_map.items():
            if isinstance(device, str) and device not in {"cpu", "disk"}:
                return inputs.to(device)
            if isinstance(device, int):
                return inputs.to(f"cuda:{device}")
    try:
        return inputs.to(model.device)
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return inputs.to(device)

def merge_lora_adapter(base_model_name: str, adapter_path: str, output_dir: str, local_files_only: bool = False) -> Tuple[str, str]:
    from peft import PeftModel

    processor = load_processor(base_model_name, local_files_only=local_files_only)
    base_model = load_image_text_model(
        base_model_name,
        device_map="cpu",
        local_files_only=local_files_only,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_root, safe_serialization=True)
    processor.save_pretrained(output_root)
    return str(output_root), str(output_root / "config.json")
