# Android Deployment Guide

Active deployment target:

- model family: `google/gemma-4-E2B-it`
- runtime: MediaPipe LLM Inference API
- artifact: `model.litertlm` for runtime, with split LiteRT `.tflite` files kept only when export inspection or bundle rebuild is needed

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
- LiteRT-LM bundling

## Expected Artifacts

For Android app integration, prepare:

- merged Hugging Face model directory for conversion only
- `model.litertlm` as the file the Android runtime loads
- split quantized LiteRT `.tflite` artifacts only if you need to inspect or rebuild the bundle
- the repo default two-stage prompts from `prompts/evidence_trace.txt` and `prompts/stage2.txt`

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

4. Validate merged-model loading in the packaging environment:

```bash
source .venv-google-ai-edge/bin/activate
PYTHONNOUSERSITE=1 python - <<'PY'
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_dir = "student/merged_models/gemma4_e2b_latest"

processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)
print("Loaded merged model and processor.")
PY
```

Do not treat a raw Hugging Face `apply_chat_template()` snippet as a stable Android contract. On this workstation, Gemma 4 multimodal chat-template behavior is version-sensitive across `transformers` builds. Reproduce the repo prompt text and message structure in the app, not the exact Python helper API.

5. Export LiteRT artifacts and bundle `model.litertlm`:

```bash
source .venv-google-ai-edge/bin/activate
PYTHONNOUSERSITE=1 python student/src/export_litert_model.py \
  --merged_model_dir student/merged_models/gemma4_e2b_latest \
  --output_dir student/mobile_artifacts/gemma4_e2b \
  --prefill_seq_len 128 \
  --kv_cache_max_len 512 \
  --trust_remote_code
```

6. The export CLI writes:
   - `model(_quantized).tflite`
   - `vision_encoder(_quantized).tflite`
   - `vision_adapter(_quantized).tflite`
   - `embedder(_quantized).tflite` and `per_layer_embedder(_quantized).tflite` when emitted by `litert-torch`
   - `model.litertlm`
   - `conversion_recipe.json`
   - `EXPORT_GUIDE.md`
7. Ship `model.litertlm` in the Android app and point `modelPath` at that file. Keep the split quantized `.tflite` files only if you explicitly want rebuild/debug artifacts alongside the bundle.

## Android Inference Contract

To match the repo's active inference path on one image, the Android app should reproduce the same two-stage flow as `student/src/inference.py`:

1. Create `LlmInference` from the exported `model.litertlm`.
2. Create an `LlmInferenceSession` that enables vision input and send one `MPImage` plus the exact text from `prompts/evidence_trace.txt`.
3. Parse the stage-1 JSON trace and compact it down to `criterion`, `score`, and `evidence` per criterion.
4. Reuse the same image and send a second prompt built from `prompts/stage2.txt`, followed by:

```text
Here is the structured evidence trace for this image:
<compact trace json>

Use the trace to synthesize the final structured decision JSON.
```

5. Parse the final JSON and treat it as the canonical 8-criteria report.

If the app wants a human-readable report paragraph or table, render it from the final JSON in the app layer. Do not switch the model prompt to free-form prose if you need parity with training and evaluation.

## Experimental Minimal Artifact Export

On this workstation, the closest result to the official LiteRT community E2B artifact size came from a 4-bit weight-only export:

```bash
source .venv-google-ai-edge/bin/activate
PYTHONNOUSERSITE=1 python student/src/export_litert_model.py \
  --merged_model_dir student/merged_models/gemma4_e2b_round1_checkpoint4000 \
  --output_dir student/mobile_artifacts/gemma4_e2b_round1_checkpoint4000_minimal_wi4 \
  --quantize weight_only_wi4_afp32 \
  --vision_quantize weight_only_wi4_afp32 \
  --prefill_seq_len 128 \
  --kv_cache_max_len 512 \
  --trust_remote_code \
  --keep_temporary_files
```

Measured outputs from that export:

- `model.litertlm`: `2,638,581,088` bytes, about `2.64 GB` decimal
- `model_quantized.tflite`: `1,162,531,728` bytes
- `embedder_quantized.tflite`: `204,475,472` bytes
- `per_layer_embedder_quantized.tflite`: `1,177,554,272` bytes
- `vision_encoder_quantized.tflite`: `87,783,184` bytes
- `vision_adapter_quantized.tflite`: `619,328` bytes

This is close to the current official LiteRT community Gemma 4 E2B size band. The earlier `dynamic_wi8_afp32` export produced a much larger `model.litertlm` at about `5.24 GB`.

## Notes

- The current public LiteRT path for Gemma export is Hugging Face -> split LiteRT `.tflite` assets -> `model.litertlm`.
- The export helper in this repo resolves tokenizer assets directly from the merged Hugging Face model directory and does not require an external SentencePiece asset.
- The regenerated deployment candidate artifact should live at `student/mobile_artifacts/gemma4_e2b_round1_checkpoint4000/model.litertlm`.
- On this workstation, validation is export-level: generate the split LiteRT files and `model.litertlm`, then hand the artifact set to Android integration. A local Python multimodal MediaPipe runtime is not available here.
- Use `--keep_temporary_files` only when you need to inspect raw and quantized `.tflite` outputs. Omit it for the smallest shippable export workspace.
- If LiteRT Torch prints a warning about C++ extensions because of the installed `torch` build, treat it as an environment issue, not a model-quality issue. Rebuild the packaging env before suspecting the adapter weights.
