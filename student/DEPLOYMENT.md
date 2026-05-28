# Android Deployment Guide

Active deployment target:

- model family: `google/gemma-4-E2B-it`
- runtime: MediaPipe LLM Inference API
- artifact: MediaPipe `.task`

## Packaging Environment

Use a dedicated `uv` environment for merge validation and LiteRT packaging. Do not reuse the training environment.

```bash
uv venv .venv-google-ai-edge --python 3.11
source .venv-google-ai-edge/bin/activate
uv pip install -r student/deployment/requirements.txt
```

This environment is for:

- merged-model local inference validation
- LiteRT Torch export
- MediaPipe `.task` bundling

Active external tokenizer asset for `.task` bundling:

- `student/deployment/tokenizers/gemma4_e2b_omote_ai/tokenizer.model`

## Expected Artifacts

For Android app integration, prepare:

- merged Hugging Face model directory
- MediaPipe `.task` bundle
- tokenizer assets packaged into the `.task`
- prompt formatting notes for the app-side prompt builder

## Workflow

1. Fine-tune a Gemma 4 E2B LoRA adapter
2. Do not automatically deploy the last checkpoint. First compare saved checkpoints and choose one that still satisfies the JSON contract on an offline slice.
3. Merge the selected adapter:

```bash
python3 student/src/merge_student.py \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --output_dir student/merged_models/gemma4_e2b_latest
```

4. Validate merged-model local inference in the packaging environment:

```bash
source .venv-google-ai-edge/bin/activate
PYTHONNOUSERSITE=1 python - <<'PY'
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_dir = "student/merged_models/gemma4_e2b_latest"
image_path = "teacher/stage1_g31b_v5_full_balanced/images/1_fake/code_lcm-lora-sdv1-5_val2017_000000089078.jpg"

processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": Image.open(image_path).convert("RGB")},
        {"type": "text", "text": "Inspect this image and return a short JSON with keys overall_likelihood and evidence."},
    ],
}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
device = next(iter(model.hf_device_map.values()))
if isinstance(device, str) and device not in ("cpu", "disk"):
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0])
PY
```

5. Export LiteRT artifacts and bundle the `.task`:

```bash
source .venv-google-ai-edge/bin/activate
PYTHONNOUSERSITE=1 python student/src/export_mediapipe_task.py \
  --merged_model_dir student/merged_models/gemma4_e2b_latest \
  --output_dir student/mobile_artifacts/gemma4_e2b \
  --tokenizer_model_path student/deployment/tokenizers/gemma4_e2b_omote_ai/tokenizer.model \
  --prefill_seq_len 128 \
  --kv_cache_max_len 512 \
  --trust_remote_code \
  --keep_temporary_files
```

6. The export CLI writes:
   - `model(_quantized).tflite`
   - `vision_encoder(_quantized).tflite`
   - `vision_adapter(_quantized).tflite`
   - `<task_name>.task` when MediaPipe bundling succeeds
   - `model.litertlm` when the script falls back to LiteRT-LM packaging
   - `conversion_recipe.json`
   - `EXPORT_GUIDE.md`
7. Ship the `.task` in the Android app and load it with MediaPipe LLM Inference API

## Notes

- The current public Google guide documents the Hugging Face -> LiteRT -> `.task` flow with Gemma examples and a MediaPipe bundler step.
- The export helper in this repo can now run LiteRT export directly and bundle a `.task` when the LiteRT files are present.
- The current Gemma 4 E2B Hugging Face snapshot on this workstation only exposes `tokenizer.json`, not `tokenizer.model`. The current `.task` path works by explicitly supplying `student/deployment/tokenizers/gemma4_e2b_omote_ai/tokenizer.model`.
- The generated bundle for the current deployment candidate is `student/mobile_artifacts/gemma4_e2b_round1_checkpoint4000_export/gemma4-e2b-authenticity.task`.
- On this workstation, validation is bundle-level: create the `.task`, verify it contains the LiteRT model, tokenizer, and metadata, then hand it off to Android integration. A local Python multimodal MediaPipe runtime is not available here.
- `.litertlm` is not the primary repo target in this version.
- If LiteRT Torch prints a warning about C++ extensions because of the installed `torch` build, treat it as an environment issue, not a model-quality issue. Rebuild the packaging env before suspecting the adapter weights.
