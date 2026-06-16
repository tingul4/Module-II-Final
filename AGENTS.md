# AGENTS.md

Guidelines for AI coding agents working in this repository.

## Project Summary

This repository builds a compact image-authenticity vision-language system. The model receives one image, checks it against 8 fixed authenticity criteria, and returns a structured JSON decision.

Active stack:

- Student backbone: `google/gemma-4-E2B-it`
- Training: Holmes-derived deterministic multi-task QLoRA SFT
- Deployment: Hugging Face fine-tune, adapter merge, LiteRT / `.litertlm` export
- Reports: new report-producing paths must write Markdown; JSON sidecars are optional

Older `Qwen`, `LPCVC`, Qualcomm, ONNX, QNN, and AIMET paths are legacy. Do not introduce new active-path names or docs that revive those stacks. Legacy filenames may remain when existing artifacts depend on them.

## Task Contract

Input: one image.

Output: JSON with:

- `per_criterion`
- `overall_likelihood`

The 8 criteria are fixed and must stay in this exact order:

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

When scoring model outputs, normalize predictions back to this contract: fill missing criteria with score `0`, then derive `overall_likelihood` from criterion scores. Do not treat `Uncertain` as a stable third class.

## Repository Map

- `teacher/`: Holmes supervision conversion and deterministic derived-data builder
- `student/`: Gemma training, detector integration, inference, evaluation, and deployment helpers
- `prompts/`: final JSON, evidence trace, taxonomy, and consistency prompts

Important active files:

- `teacher/convert_holmes_sft.py`
- `teacher/build_derived_dataset.py`
- `student/src/train.py`
- `student/src/inference.py`
- `student/src/evaluate.py`
- `student/src/utils/task_utils.py`
- `student/src/deployment/merge_student.py`
- `student/src/deployment/export_litert_model.py`

Primary data:

- Holmes source root: `../dataset/holmes`
- Derived dataset: `teacher/derived_deterministic_v1/derived.jsonl`
- Legacy teacher JSONL: `teacher/stage1_g31b_v5_full_balanced/holmes_lpcvc_sft.jsonl`

## Agent Rules

- Think before coding. Restate the task, identify assumptions, and inspect relevant files before non-trivial edits.
- Keep changes surgical. Do not refactor, rename, reorganize, or reformat unrelated code.
- Prefer existing modules, naming, CLI style, and repo-relative defaults.
- Do not add dependencies unless the task requires them.
- Preserve legacy artifact names when changing them would break existing data or outputs.
- Keep logs, comments, and error messages concise.
- Every new report path must emit Markdown.
- When changing prompts, preserve the 8 criteria, their order, JSON keys, and score semantics.
- When changing training loops or callbacks, preserve `training.log` progress visibility: `epoch`, `epoch_step`, `global_step`, progress percentage, loss, learning rate, and ETA.
- Before merge or deployment, select a checkpoint explicitly. Do not assume the latest checkpoint is best.
- Use at least one offline evaluation slice for checkpoint selection. Treat tiny fixed-step sample evaluations as smoke signals only.
- For classification improvements, compare Gemma-only behavior against the detector-first baseline on the same offline slice before changing prompts or training mix.
- Calibrate detector thresholds on an offline slice; the default `0.5` threshold can under-call fake images.
- Treat detector-assisted inference as detector-first classification plus Gemma explanation unless evaluation shows Gemma improves the binary label.

## Training And Deployment Notes

Active student defaults:

- Backbone: `google/gemma-4-E2B-it`
- Method: 4-bit QLoRA with PEFT LoRA adapters
- Optimizer: `paged_adamw_8bit`
- Precision: `bf16`
- Gradient accumulation: `8`
- Loss masking: tokens before assistant response use `-100`

Before any full training run, complete a bounded LoRA smoke run on the active backbone with `--max_steps`. Confirm the run loads the model, builds the dataset, starts optimization, and writes checkpoints/logs.

During training, monitor `training.log`. Stop and inspect if loss diverges, becomes `NaN`, or shows clear instability. Fixed-step sample evaluations must persist artifacts under the run directory and record final JSON parse status, predicted and gold `overall_likelihood`, and raw prediction text for a deterministic sample slice. Trace generation should retry once with a larger token budget before recording a parse failure.

Deployment path:

1. Fine-tune Gemma 4 E2B with LoRA.
2. Select a deployment checkpoint.
3. Merge the adapter into a Hugging Face model directory.
4. Validate merged-model local inference.
5. Export LiteRT assets and a `.litertlm` artifact.
6. Integrate and smoke test the Android runtime path.

Packaging requirements:

- Use `uv venv .venv-google-ai-edge --python 3.11`
- Install repo-root `requirements.txt`
- Run packaging commands with `PYTHONNOUSERSITE=1`
- Keep packaging and training environments separate
- Prefer `dynamic_wi8_afp32` for Android smoke testing before trying smaller weight-only exports

Current deployment candidate:

- `student/outputs/gemma4_e2b_round1_20260527/checkpoint-4000`

Historical baseline only:

- `student/outputs/20260427_105054/checkpoint-7014`

## Verification

Run focused checks for the files you changed. For broad syntax validation:

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

For packaging environment setup:

```bash
uv venv .venv-google-ai-edge --python 3.11
source .venv-google-ai-edge/bin/activate
uv pip install -r requirements.txt
```

## Known Risks

- LiteRT export depends on a working local LiteRT Torch toolchain and version-compatible Gemma export builder.
- LiteRT public docs may lag Gemma 4 E2B support; validate against the installed package version before treating exports as production-ready.
- This workstation can validate `.litertlm` creation and inspect LiteRT files, but it does not provide a local multimodal MediaPipe runtime equivalent to the Android path.
- Full evaluation is expensive because the pipeline may perform two generations per sample plus optional probes.
- Do not trust detector GPU results from the system `python3` environment on this workstation. It uses `torch 1.13.1+cu117`, which does not support Blackwell `sm_120` correctly and can return all-zero CUDA results.
- Use a version-compatible PyTorch build for Blackwell GPU validation. The local `.venv-google-ai-edge` environment uses `torch 2.9.0+cu128` and reproduces AIGI-Holmes CLIP LoRA detector scores within CPU reference tolerance.
