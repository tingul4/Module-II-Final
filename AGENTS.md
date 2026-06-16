# Holmes-Derived AI Image Authenticity Project Guide

This repository trains a compact student vision-language model that inspects one image, reasons across 8 fixed authenticity criteria, and emits a structured JSON decision. The active stack is **Gemma 4 E2B + Google AI Edge / LiteRT**. Older `Qwen`, `LPCVC`, and Qualcomm-oriented artifacts are legacy only.

## Active Architecture

- Student backbone default: `google/gemma-4-E2B-it`
- Training path: Holmes-derived deterministic multi-task QLoRA SFT
- Deployment path: `HF fine-tune -> merge -> .litertlm`
- Secondary artifact: merged Hugging Face model directory for local validation and conversion
- Reports: every new report path must emit **Markdown**; JSON sidecars are optional machine-readable companions

Do not add new active-path naming that reintroduces `lpcvc`, `competition`, `qwen`, `qualcomm`, `onnx`, `qnn`, or `aimet`. Legacy filenames may remain when they point to already-generated data.

## Task Contract

Input: one image.

Output: JSON with:

- `per_criterion`
- `overall_likelihood`

The 8 criteria are fixed and remain in this exact order:

1. `Lighting & Shadows Consistency`
2. `Edges & Boundaries`
3. `Texture & Resolution`
4. `Perspective & Spatial Relationships`
5. `Physical & Common Sense Logic`
6. `Text & Symbols`
7. `Human & Biological Structure Integrity`
8. `Material & Object Details`

Score semantics:

- `aigc score = 1`: explicit grounded artifact evidence exists for the criterion
- `aigc score = 0`: no grounded artifact is visible, or the criterion is not applicable
- `overall_likelihood = AI-Generated` when one or more criteria have score `1`
- `overall_likelihood = Real` when all criteria are `0`

## Repository Layout

- `teacher/`: Holmes supervision conversion and derived-data builder
- `student/`: Gemma student training, detector integration, inference, evaluation, and deployment helpers
- `prompts/`: final JSON, evidence trace, taxonomy, and consistency prompts

Important active files:

- `student/src/train.py`
- `student/src/inference.py`
- `student/src/evaluate.py`
- `student/src/deployment/merge_student.py`
- `student/src/deployment/export_litert_model.py`
- `student/src/utils/task_utils.py`

## Data Sources

Expected Holmes source root:

```text
../dataset/holmes
```

Local teacher dataset:

```text
teacher/stage1_g31b_v5_full_balanced/
  holmes_lpcvc_sft.jsonl
  stats.json
  images/0_real/
  images/1_fake/
```

The filename `holmes_lpcvc_sft.jsonl` is a legacy artifact name. Treat it as historical only.

Deterministic derived dataset:

```text
teacher/derived_deterministic_v1/
  derived.jsonl
  manifest.json
```

Derived rows add:

- `final_json_target`
- `evidence_trace_target`
- `taxonomy_target`
- `consistency_target`
- `quality_flags`

## Teacher Pipeline

Main files:

- `teacher/convert_holmes_sft.py`
- `teacher/build_derived_dataset.py`

Purpose:

- Read Holmes `SFTDATA.jsonl`
- Resolve images from `dataset_huggingface.zip`
- Preserve Holmes-fixed labels derived from image path
- Rewrite Holmes explanations into the 8 canonical criteria
- Materialize draft JSONL and derived multi-task supervision

Supported teacher backends:

- `heuristic`
- `openai_compatible`
- `transformers_gemma4`

Typical draft-generation command:

```bash
python3 teacher/convert_holmes_sft.py \
  --holmes-root ../dataset/holmes \
  --teacher-backend transformers_gemma4 \
  --model google/gemma-4-e2b-it \
  --pipeline-stage generator_only \
  --balance-label-order \
  --batch-size 8 \
  --max-samples 32070 \
  --output-root teacher/stage1_g31b_v5_full_balanced \
  --overwrite-images
```

Build derived supervision without relabeling:

```bash
python3 teacher/build_derived_dataset.py \
  --input-jsonl teacher/stage1_g31b_v5_full_balanced/holmes_lpcvc_sft.jsonl \
  --output-root teacher/derived_deterministic_v1
```

## Student Training Pipeline

Active student implementation:

- Backbone default: `google/gemma-4-E2B-it`
- Training method: 4-bit QLoRA with PEFT LoRA adapters
- Optimizer: `paged_adamw_8bit`
- Precision: `bf16`
- Gradient accumulation: `8`
- Loss masking: tokens before the assistant response are `-100`
- Training log requirement: `training.log` must persist `epoch`, `epoch_step`, `global_step`, progress percentage, loss, learning rate, and ETA
- Before any full run, complete a LoRA smoke run on the active backbone and confirm the run can load the model, build the dataset, start optimization, and write checkpoints/logs.
- Bound every smoke run explicitly with `--max_steps` so it finishes quickly and never turns into an accidental long run.
- During active training, monitor loss and optimizer metrics from `training.log`; stop and inspect if loss diverges, becomes `NaN`, or shows obvious instability.
- Run fixed-step sample evaluation during training and persist its artifacts under the run directory so prediction-vs-GT drift is inspectable without attaching to the process.
- Every fixed-step sample evaluation must record at least final JSON parse status, predicted `overall_likelihood`, gold `overall_likelihood`, and raw prediction text for a deterministic sample slice.
- Fixed-step trace generation should retry once with a larger token budget before recording a trace parse failure, so token-budget truncation is not misread as model collapse.
- Do not judge checkpoint quality from a 2-row fixed-step sample alone; treat such a slice as a smoke signal only and use a larger offline evaluation slice before concluding that a checkpoint regressed.
- Before merge or deployment, select a checkpoint explicitly. Do not assume the last checkpoint is the best deployment candidate.
- Checkpoint selection must use at least one offline evaluation slice in addition to fixed-step sample evaluation artifacts.
- When reporting binary `overall_likelihood` metrics, normalize predicted final JSON back to the task contract before scoring: fill missing criteria with score `0`, then derive `overall_likelihood` from predicted criterion scores. Do not treat `Uncertain` as a stable third evaluation class for this task.
- If the goal is to maximize overall classification `accuracy` / `macro F1`, compare Gemma-only results against a detector-first baseline on the same offline slice before changing prompts or training mix. The current verified external baseline is `AIGI-Holmes` CLIP LoRA.
- Detector thresholds must be calibrated on an offline slice. The default detector threshold `0.5` may materially under-call fake images even when the detector score distribution is otherwise useful.
- Treat detector-assisted inference as detector-first classification plus Gemma explanation unless a broader evaluation shows that Gemma materially improves the binary label beyond a small calibration bump.

Primary dataset:

- `teacher/derived_deterministic_v1/derived.jsonl`

Task mix:

- `final_json`
- `evidence_trace`
- `taxonomy_classification`
- `consistency_check`

Recommended task mix:

```bash
--task_mix '{"final_json":0.4,"evidence_trace":0.35,"taxonomy_classification":0.15,"consistency_check":0.1}'
```

Repo-relative training example:

```bash
python3 student/src/train.py \
  --model_name_or_path google/gemma-4-E2B-it \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts \
  --output_dir student/outputs \
  --batch_size 2 \
  --epochs 3 \
  --lr 1e-4 \
  --sample_eval_steps 500 \
  --sample_eval_rows 4
```

Smoke-run example:

```bash
python3 student/src/train.py \
  --model_name_or_path google/gemma-4-E2B-it \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts \
  --output_dir student/outputs \
  --run_name gemma4_e2b_smoke \
  --batch_size 1 \
  --epochs 1 \
  --max_steps 5 \
  --save_steps 5 \
  --sample_eval_steps 5 \
  --sample_eval_rows 2
```

Inference:

```bash
python3 student/src/inference.py \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --image_path <image> \
  --prompt_dir prompts
```

Evaluation:

```bash
python3 student/src/evaluate.py \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts
```

Detector-first evaluation uses the main evaluator with `--prediction_source detector`
or `--prediction_source detector_student`. The vendored CLIP LoRA checkpoint lives
under `student/outputs/detectors/holmes_clip_lora_vitl14_336/`, while the base CLIP
`ViT-L-14-336px.pt` path is supplied explicitly at runtime.

Legacy baseline:

- `student/outputs/20260427_105054/checkpoint-7014`

This checkpoint is historical only. Do not treat it as the active student line.

Training evaluation artifacts:

- `training.log`: persistent optimizer/progress metrics
- `training_eval/step_<N>.json`: fixed-step sample evaluation JSON
- `training_eval/step_<N>.md`: fixed-step sample evaluation Markdown

## Deployment Pipeline

The official repo deployment path is:

1. Fine-tune Gemma 4 E2B with LoRA
2. Select a deployment checkpoint
3. Merge the adapter into a full Hugging Face model directory
4. Validate merged-model local inference
5. Convert to LiteRT artifacts and bundle a `.litertlm` artifact
6. Deploy the exported LiteRT asset set in the chosen Android runtime path
7. Run the chosen Android artifact in the Google AI Edge runtime path

Packaging environment requirements:

- Use `uv venv .venv-google-ai-edge --python 3.11`
- Install repo-root `requirements.txt`
- Run packaging commands with `PYTHONNOUSERSITE=1`
- Keep packaging and training environments separate

Deployment CLIs:

```bash
python3 student/src/deployment/merge_student.py \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --output_dir student/merged_models/gemma4_e2b_latest
```

```bash
python3 student/src/deployment/export_litert_model.py \
  --merged_model_dir student/merged_models/gemma4_e2b_latest \
  --output_dir student/mobile_artifacts/gemma4_e2b \
  --prefill_seq_len 128 \
  --kv_cache_max_len 512 \
  --trust_remote_code \
  --keep_temporary_files
```

End-to-end packaging flow:

```bash
uv venv .venv-google-ai-edge --python 3.11
source .venv-google-ai-edge/bin/activate
uv pip install -r requirements.txt
PYTHONNOUSERSITE=1 python student/src/deployment/merge_student.py \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --output_dir student/merged_models/gemma4_e2b_latest
PYTHONNOUSERSITE=1 python student/src/deployment/export_litert_model.py \
  --merged_model_dir student/merged_models/gemma4_e2b_latest \
  --output_dir student/mobile_artifacts/gemma4_e2b \
  --quantize dynamic_wi8_afp32 \
  --vision_quantize dynamic_wi8_afp32 \
  --prefill_seq_len 128 \
  --kv_cache_max_len 512 \
  --trust_remote_code \
  --keep_temporary_files
```

`export_litert_model.py` is the active export CLI. It runs LiteRT export directly, writes a conversion recipe and export guide, and ensures the output workspace contains `model.litertlm` plus the split LiteRT `.tflite` assets required by the runtime path.

For Android smoke testing, prefer the `dynamic_wi8_afp32` memory-friendly artifact path before trying smaller weight-only exports. The 4-bit weight-only artifact is useful for file-size comparison, but it can increase runtime memory pressure through explicit float dequantization.

## Phase Status

Status legend:

- `[x]` completed
- `[-]` in progress
- `[ ]` pending

- `[x]` Holmes-derived deterministic dataset builder exists
- `[x]` Multi-task student pipeline exists
- `[x]` Detector-first Gemma + CLIP LoRA architecture is wired into the active student stack
- `[x]` Markdown report generation is wired into evaluation/report workflows
- `[x]` Active backbone switched to Gemma 4 E2B in code and docs
- `[x]` Merge and LiteRT export helper CLIs added
- `[x]` Gradient-path fix validated for Gemma 4 E2B QLoRA (`use_reentrant=False` plus post-PEFT input-gradient hook)
- `[x]` Bounded Gemma 4 E2B LoRA smoke run completed at `student/outputs/gemma4_e2b_smoke_fix`
- `[x]` First Gemma 4 E2B full training run completed at `student/outputs/gemma4_e2b_round1_20260527`
- `[x]` Current deployment candidate checkpoint selected as `student/outputs/gemma4_e2b_round1_20260527/checkpoint-4000`
- `[x]` Merged-model local inference validation on a real Gemma checkpoint
- `[x]` LiteRT export validated for `student/outputs/gemma4_e2b_round1_20260527/checkpoint-4000`
- `[x]` LiteRT-LM export path validated for `student/outputs/gemma4_e2b_round1_20260527/checkpoint-4000`
- `[-]` Android runtime integration is still pending on this workstation
- `[ ]` Android app integration smoke test

## Verification Commands

Syntax check:

```bash
python3 -m py_compile \
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

Packaging environment build:

```bash
uv venv .venv-google-ai-edge --python 3.11
source .venv-google-ai-edge/bin/activate
uv pip install -r requirements.txt
```

## Known Gaps

- LiteRT export depends on a working local LiteRT Torch toolchain and version-compatible Gemma export builder.
- LiteRT Torch public docs are still centered on Gemma 3 examples; Gemma 4 E2B export should be validated against the installed package version before treating it as production-ready.
- On this workstation, Android export requires a clean isolated `uv` environment. Mixed user-site packages can break LiteRT export even when the model weights are healthy.
- The current Hugging Face Gemma 4 E2B snapshot available locally exposes `tokenizer.json`; the active export path packages tokenizer assets directly from the merged model directory.
- This workstation can validate `.litertlm` creation and inspect exported LiteRT files, but does not provide a local multimodal MediaPipe runtime equivalent to the Android app path.
- Android smoke tests should pin `com.google.ai.edge.litertlm:litertlm-android` to a known version, start with CPU text and vision backends, set `maxNumTokens` to the exported metadata budget, and verify the Gemma 4 multimodal prompt template before testing GPU.
- Full evaluation remains expensive because the pipeline performs two generations per sample plus optional probes.
- On this workstation, the failure mode is environment-specific, not model-specific: the system `python3` path uses `torch 1.13.1+cu117`, which does not support Blackwell `sm_120` and can return all-zero CUDA results even for basic ops such as `matmul`, `conv`, and `sigmoid`. Do not trust detector GPU results from that environment.
- A version-compatible PyTorch build is required for Blackwell GPU validation. The local `.venv-google-ai-edge` environment uses `torch 2.9.0+cu128`, advertises `sm_120`, and reproduces the `AIGI-Holmes` CLIP LoRA detector scores on GPU within numerical tolerance of the CPU reference.

## Editing Guidance

- Keep the 8 criteria, JSON keys, and score semantics stable.
- Prefer repo-relative defaults in new scripts and docs.
- Preserve legacy dataset filenames when renaming would break existing artifacts.
- When changing prompts, do not change criterion order or output keys.
- When changing training loops or callbacks, preserve persisted progress visibility in `training.log`.
- Every new report-producing path must write Markdown output; add JSON only as a companion artifact when useful.
