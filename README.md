# Holmes-Derived AI Image Authenticity

This repository builds a compact vision-language model pipeline for AI image authenticity inspection. The input is one image. The output is a structured JSON decision over 8 fixed authenticity criteria, plus an overall `Real` or `AI-Generated` label.

The active stack is:

- Student model: `google/gemma-4-E2B-it`
- Fine-tuning method: 4-bit QLoRA multi-task SFT
- Classification policy: detector-first binary decision with Gemma-generated structured explanation
- Deployment target: Google AI Edge / LiteRT `.litertlm`

## Task Contract

The model inspects a single image and emits JSON with:

- `per_criterion`
- `overall_likelihood`

The 8 criteria are fixed and must stay in this order:

1. `Lighting & Shadows Consistency`
2. `Edges & Boundaries`
3. `Texture & Resolution`
4. `Perspective & Spatial Relationships`
5. `Physical & Common Sense Logic`
6. `Text & Symbols`
7. `Human & Biological Structure Integrity`
8. `Material & Object Details`

Score semantics:

- `aigc score = 1`: explicit grounded artifact evidence is visible for the criterion.
- `aigc score = 0`: no grounded artifact is visible, or the criterion is not applicable.
- `overall_likelihood = AI-Generated`: one or more criteria have score `1`.
- `overall_likelihood = Real`: all criteria are `0`.

In the detector-first runtime path, the detector can override the final binary `overall_likelihood`; Gemma still produces the 8-criterion explanation payload.

## Repository Layout

```text
.
├── prompts/                         # Final JSON, trace, taxonomy, and consistency prompts
├── teacher/                         # Holmes conversion and deterministic derived-data builder
│   ├── convert_holmes_sft.py
│   ├── build_derived_dataset.py
│   ├── stage1_g31b_v5_full_balanced/
│   └── derived_deterministic_v1/
├── student/                         # Training, inference, evaluation, detector, and deployment code
│   ├── src/
│   │   ├── train.py
│   │   ├── inference.py
│   │   ├── evaluate.py
│   │   ├── detectors/
│   │   ├── deployment/
│   │   └── utils/
│   ├── outputs/
│   ├── merged_models/
│   └── mobile_artifacts/
├── PROJECT_OVERVIEW.md              # More detailed technical overview
├── requirements.txt
└── AGENTS.md
```

Important docs:

- `teacher/README.md`: teacher-side Holmes conversion and derived dataset generation
- `student/README.md`: student training, evaluation, detector, and inference details
- `student/DEPLOYMENT.md`: merge, LiteRT export, and Android integration notes

## Data Flow

The active training pipeline is:

```text
Holmes source data
  -> teacher/convert_holmes_sft.py
  -> teacher/stage1_g31b_v5_full_balanced/holmes_lpcvc_sft.jsonl
  -> teacher/build_derived_dataset.py
  -> teacher/derived_deterministic_v1/derived.jsonl
  -> student/src/train.py
  -> student/outputs/<run>/checkpoint-<step>
  -> student/src/inference.py or student/src/evaluate.py
```

The derived dataset adds multiple supervision targets per image:

- `final_json_target`: final 8-criterion task JSON
- `evidence_trace_target`: criterion-level evidence decomposition
- `taxonomy_target`: artifact taxonomy and support provenance
- `consistency_target`: score/evidence consistency target
- `quality_flags`: dataset QA signals

The default split is a deterministic stratified `90/10` train/eval split stored at:

```text
teacher/derived_deterministic_v1/derived_split.json
```

## Dataset Source

The upstream Holmes-style dataset used in this project can be downloaded from Hugging Face:

- [AIGI-Holmes-Dataset](https://huggingface.co/datasets/zzy0123/AIGI-Holmes-Dataset)

From the repository root, the expected relative source root for teacher conversion is:

```text
../dataset/holmes
```

The top-level training flow is:

1. Download the source dataset.
2. Convert Holmes supervision into teacher-side canonical JSONL.
3. Build `teacher/derived_deterministic_v1/derived.jsonl`.
4. Use that derived JSONL for both Gemma student fine-tuning and CLIP detector fine-tuning.

## What `derived.jsonl` Means

`teacher/derived_deterministic_v1/derived.jsonl` is the active row-wise supervision file for this repository. Each line is one image sample plus all derived targets needed by the student stack.

Each row contains:

- `row_id`: stable sample ID used by train/eval split and offline evaluation.
- `image`: image path relative to `image_root`.
- `image_root`: root directory that resolves the actual image file.
- `source`: source dataset tag, currently Holmes-derived.
- `original_query`: original Holmes prompt text.
- `original_response`: original Holmes explanation text.
- `step1_target`: short image summary used by the derived supervision pipeline.
- `final_json_target`: final deployment-style 8-criterion JSON target.
- `evidence_trace_target`: richer criterion-level trace with score, evidence, support type, Holmes span, and artifact taxonomy.
- `taxonomy_target`: compact artifact-type and support-provenance supervision.
- `consistency_target`: score/evidence consistency supervision.
- `quality_flags`: teacher-side QA flags for suspicious or noisy rows.

This means `derived.jsonl` is not just a file list. It is the canonical multi-task supervision surface for:

- `student/src/train.py`
- `student/src/evaluate.py`
- `student/src/detectors/holmes_clip_lora/train.py`

The other files under `teacher/derived_deterministic_v1/` are:

- `manifest.json`: dataset-level summary, field list, label counts, and example payloads.
- `derived_split.json`: deterministic train/eval split manifest keyed by `row_id`.

## Environment

Create a dedicated training environment for teacher conversion, Gemma fine-tuning, detector training, inference, and evaluation:

```bash
uv venv .venv-train --python 3.10
uv pip install --python .venv-train/bin/python 'torch==2.9.0' -r requirements.txt
.venv-train/bin/python -m nltk.downloader wordnet omw-1.4
```

For LiteRT packaging, use a separate Python 3.11 environment:

```bash
uv venv .venv-google-ai-edge --python 3.11
uv pip install --python .venv-google-ai-edge/bin/python 'torch==2.9.0' -r requirements.txt
```

Do not mix the training environment and the LiteRT packaging environment. On this workstation:

- use `.venv-train/bin/python` for teacher, student, detector, inference, and evaluation commands
- use `PYTHONNOUSERSITE=1 .venv-google-ai-edge/bin/python` for merge and LiteRT export
- if Gemma weights are not already cached, the first teacher/student command will download them from Hugging Face; set `HF_TOKEN` if the machine needs authenticated access

## Build Teacher And Derived Data

If the generated teacher and derived datasets already exist, guests can skip this section and train from:

```text
teacher/derived_deterministic_v1/derived.jsonl
```

Expected Holmes source root after download, relative to the repository root:

```text
../dataset/holmes
```

Generate Holmes-derived teacher data:

```bash
.venv-train/bin/python teacher/convert_holmes_sft.py \
  --holmes-root ../dataset/holmes \
  --teacher-backend transformers_gemma4 \
  --model google/gemma-4-31B-it \
  --judge-model google/gemma-4-31B-it \
  --specialist-model google/gemma-4-31B-it \
  --pipeline-stage full \
  --balance-label-order \
  --batch-size 1 \
  --max-samples 32070 \
  --output-root teacher/stage1_g31b_v5_full_balanced \
  --overwrite-images
```

Build deterministic student supervision:

```bash
.venv-train/bin/python teacher/build_derived_dataset.py \
  --input-jsonl teacher/stage1_g31b_v5_full_balanced/holmes_lpcvc_sft.jsonl \
  --output-root teacher/derived_deterministic_v1
```

The filename `holmes_lpcvc_sft.jsonl` is legacy. Treat it as the current Holmes-derived teacher JSONL.

## Fine-Tune Gemma Student

Before a full run, run a bounded smoke test. This verifies that the model loads, the dataset builds, optimization starts, and checkpoints/logs are written.

```bash
.venv-train/bin/python student/src/train.py \
  --model_name_or_path google/gemma-4-E2B-it \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts \
  --output_dir student/outputs \
  --run_name gemma4_e2b_smoke \
  --batch_size 1 \
  --epochs 1 \
  --max_steps 5 \
  --save_steps 5 \
  --disable_epoch_eval
```

Full training example with epoch-level detector-student evaluation:

```bash
.venv-train/bin/python student/src/train.py \
  --model_name_or_path google/gemma-4-E2B-it \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts \
  --output_dir student/outputs \
  --run_name gemma4_e2b_round2 \
  --batch_size 2 \
  --epochs 3 \
  --lr 1e-4 \
  --task_mix '{"final_json":0.4,"evidence_trace":0.35,"taxonomy_classification":0.15,"consistency_check":0.1}' \
  --detector_checkpoint_path student/outputs/detectors/holmes_clip_lora_vitl14_336/checkpoints/model_epoch_0.94_0.99.pth \
  --detector_clip_weights ../bk/AIGI-Holmes/pretrained/clip/ViT-L-14-336px.pt \
  --detector_threshold 0.34
```

Training writes:

- `student/outputs/<run>/training.log`
- `student/outputs/<run>/checkpoint-<step>/`
- `student/outputs/<run>/training_eval/epoch_<N>.json`
- `student/outputs/<run>/training_eval/epoch_<N>.md`
- `student/outputs/<run>/epoch_checkpoints/epoch_<N>/`

Monitor `training.log` during long runs. Stop and inspect if loss becomes `NaN`, diverges, or shows obvious instability.

## Fine-Tune CLIP Detector

The detector is a separate binary classifier from the Gemma student. Its job is to predict the final `Real` vs `AI-Generated` label, while Gemma produces the structured 8-criterion explanation.

Detector fine-tuning uses:

- backbone: `CLIP ViT-L/14@336px`
- train mode: LoRA plus classifier head fine-tuning
- active config: `student/outputs/detectors/holmes_clip_lora_vitl14_336/config_train.yaml`
- active data source: `teacher/derived_deterministic_v1/derived.jsonl`
- active split file: `teacher/derived_deterministic_v1/derived_split.json`

The detector training script reads the derived JSONL, resolves each image path, and converts `overall_likelihood` into the binary training label. It trains only the LoRA parameters and the detector head, not the full CLIP backbone.

Example detector smoke run:

```bash
.venv-train/bin/python student/src/detectors/holmes_clip_lora/train.py \
  --config student/outputs/detectors/holmes_clip_lora_vitl14_336/config_train.yaml \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --split_manifest_path teacher/derived_deterministic_v1/derived_split.json \
  --clip_weights ../bk/AIGI-Holmes/pretrained/clip/ViT-L-14-336px.pt \
  --run_name clip_lora_derived_smoke \
  --gpu_ids 0 \
  --batch_size 1 \
  --val_batch_size 1 \
  --num_workers 2 \
  --epochs 1 \
  --max_steps 2
```

Full detector fine-tuning example:

```bash
.venv-train/bin/python student/src/detectors/holmes_clip_lora/train.py \
  --config student/outputs/detectors/holmes_clip_lora_vitl14_336/config_train.yaml \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --split_manifest_path teacher/derived_deterministic_v1/derived_split.json \
  --clip_weights ../bk/AIGI-Holmes/pretrained/clip/ViT-L-14-336px.pt \
  --run_name clip_lora_derived_v1 \
  --gpu_ids 0 \
  --batch_size 8 \
  --val_batch_size 8 \
  --epochs 4 \
  --lr 5e-5 \
  --threshold 0.5
```

Detector training writes checkpoint artifacts under:

```text
student/outputs/detectors/holmes_clip_lora_vitl14_336/checkpoints/<run_name>/
```

Typical outputs include:

- `resolved_config.yaml`
- `training.log`
- `metrics_history.jsonl`
- `model_best_f1_*.pth`
- `model_epoch_last.pth`

After training, evaluate and calibrate the detector threshold on an offline slice before using it in `detector_student` inference.

## Evaluate Checkpoints

Do not deploy the last checkpoint by default. Pick a checkpoint explicitly using offline evaluation in addition to the epoch-level training evaluation artifacts.

Student evaluation:

```bash
.venv-train/bin/python student/src/evaluate.py \
  --prediction_source student \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/gemma4_e2b_round1_20260527/checkpoint-4000 \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts \
  --split eval \
  --eval_slice_count 4 \
  --eval_slice_seed 42 \
  --output_path student/outputs/student_eval.json
```

Detector-only evaluation:

```bash
.venv-train/bin/python student/src/evaluate.py \
  --prediction_source detector \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --split eval \
  --eval_slice_count 50 \
  --eval_slice_seed 42 \
  --detector_checkpoint_path student/outputs/detectors/holmes_clip_lora_vitl14_336/checkpoints/model_epoch_0.94_0.99.pth \
  --detector_clip_weights ../bk/AIGI-Holmes/pretrained/clip/ViT-L-14-336px.pt \
  --detector_threshold 0.34 \
  --output_path student/outputs/detector_eval.json
```

Detector plus student evaluation:

```bash
.venv-train/bin/python student/src/evaluate.py \
  --prediction_source detector_student \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/gemma4_e2b_round1_20260527/checkpoint-4000 \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts \
  --split eval \
  --eval_slice_count 4 \
  --eval_slice_seed 42 \
  --detector_checkpoint_path student/outputs/detectors/holmes_clip_lora_vitl14_336/checkpoints/model_epoch_0.94_0.99.pth \
  --detector_clip_weights ../bk/AIGI-Holmes/pretrained/clip/ViT-L-14-336px.pt \
  --detector_threshold 0.34 \
  --output_path student/outputs/detector_student_eval.json
```

When `--output_path` is set, evaluation writes a JSON report and a Markdown companion report.

## Run Inference

Student-only inference:

```bash
.venv-train/bin/python student/src/inference.py \
  --prediction_source student \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/gemma4_e2b_round1_20260527/checkpoint-4000 \
  --image_path teacher/stage1_g31b_v5_full_balanced/images/1_fake/code_lcm-lora-sdv1-5_val2017_000000089078.jpg \
  --prompt_dir prompts
```

Detector-first inference with Gemma explanation:

```bash
.venv-train/bin/python student/src/inference.py \
  --prediction_source detector_student \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/gemma4_e2b_round1_20260527/checkpoint-4000 \
  --image_path teacher/stage1_g31b_v5_full_balanced/images/1_fake/code_lcm-lora-sdv1-5_val2017_000000089078.jpg \
  --prompt_dir prompts \
  --detector_checkpoint_path student/outputs/detectors/holmes_clip_lora_vitl14_336/checkpoints/model_epoch_0.94_0.99.pth \
  --detector_clip_weights ../bk/AIGI-Holmes/pretrained/clip/ViT-L-14-336px.pt \
  --detector_threshold 0.34 \
  --output_path student/outputs/inference_result.json
```

The detector-first output includes:

- `detector_score`
- `detector_label`
- `student_overall_likelihood`
- final `overall_likelihood`, overridden by the detector label
- raw and parsed evidence-trace / final-JSON generations

## Merge And Export For LiteRT

Merge a selected LoRA checkpoint into a Hugging Face model directory:

```bash
PYTHONNOUSERSITE=1 .venv-google-ai-edge/bin/python student/src/deployment/merge_student.py \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/gemma4_e2b_round1_20260527/checkpoint-4000 \
  --output_dir student/merged_models/gemma4_e2b_round1_checkpoint4000_readme
```

Prepare a LiteRT export workspace with a dry run:

```bash
PYTHONNOUSERSITE=1 .venv-google-ai-edge/bin/python student/src/deployment/export_litert_model.py \
  --merged_model_dir student/merged_models/gemma4_e2b_round1_checkpoint4000 \
  --output_dir student/mobile_artifacts/gemma4_e2b \
  --dry_run
```

The dry run writes:

- `conversion_recipe.json`
- `EXPORT_GUIDE.md`

For a full export, rerun without `--dry_run`:

```bash
PYTHONNOUSERSITE=1 .venv-google-ai-edge/bin/python student/src/deployment/export_litert_model.py \
  --merged_model_dir student/merged_models/gemma4_e2b_round1_checkpoint4000 \
  --output_dir student/mobile_artifacts/gemma4_e2b \
  --quantize dynamic_wi8_afp32 \
  --vision_quantize dynamic_wi8_afp32 \
  --prefill_seq_len 128 \
  --kv_cache_max_len 512 \
  --trust_remote_code
```

The full export workspace should contain:

- `model.litertlm`
- split LiteRT `.tflite` assets
- tokenizer assets
- `conversion_recipe.json`
- `EXPORT_GUIDE.md`

See `student/DEPLOYMENT.md` for Android runtime notes.

## Quick Verification

Syntax-check the main Python entry points:

```bash
.venv-train/bin/python -m py_compile \
  teacher/convert_holmes_sft.py \
  teacher/build_derived_dataset.py \
  student/src/utils/task_utils.py \
  student/src/utils/model_utils.py \
  student/src/train.py \
  student/src/inference.py \
  student/src/evaluate.py \
  student/src/deployment/merge_student.py \
  student/src/deployment/export_litert_model.py \
  student/src/detectors/holmes_clip_lora/train.py
```

For a new machine, the recommended minimum validation order is:

1. Install dependencies.
2. Run the Gemma LoRA smoke test with `--max_steps 5`.
3. Run inference on one image with a known checkpoint.
4. Run a small eval slice before selecting a deployment checkpoint.
5. Merge the selected checkpoint.
6. Run a LiteRT dry run in the isolated packaging environment.
7. Export LiteRT artifacts once the dry run looks correct.
