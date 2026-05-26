# LPCVC 2026 Track 3 Project Guide

This repository contains the working pipeline for LPCVC 2026 Track 3: AI-Generated Image Detection. The goal is to train a compact student Vision-Language Model (VLM) that predicts whether an input image is `Real` or `AI-Generated` and emits criterion-level evidence in the competition JSON format.

## Task Summary

- Competition: LPCVC 2026 Track 3, AI-Generated Image Detection.
- Input: one image.
- Output: structured JSON with 8 criteria and an `overall_likelihood`.
- Runtime gate: model must pass the speed threshold, documented as `> 15 TPS`, before accuracy is evaluated.
- Target platform: Qualcomm Snapdragon 8 Elite Gen5 Mobile.
- Expected deployment direction: ONNX model plus validation script, with Qualcomm AI Hub / AIMET quantization work still to be added.

The 8 fixed criteria are:

1. `Lighting & Shadows Consistency`
2. `Edges & Boundaries`
3. `Texture & Resolution`
4. `Perspective & Spatial Relationships`
5. `Physical & Common Sense Logic`
6. `Text & Symbols`
7. `Human & Biological Structure Integrity`
8. `Material & Object Details`

## Repository Layout

- `prompts/stage1.txt`: three Stage 1 image-analysis prompts.
- `prompts/stage2.txt`: Stage 2 synthesis prompt for strict JSON output.
- `prompts/evidence_trace.txt`, `prompts/taxonomy.txt`, `prompts/consistency.txt`: multi-task prompts for the derived supervision pipeline.
- `teacher/`: Holmes-to-LPCVC teacher-data conversion pipeline.
- `student/`: Qwen VLM SFT training and inference pipeline.
- `run_eda.py`, `eda_results.md`: lightweight dataset inspection utility and previous EDA output.

## Data Sources

The original Holmes dataset is expected at:

```text
/ssd4/LPCVC2026/dataset/holmes
```

The source dataset normally contains:

- `SFTDATA.jsonl`
- `dataset_huggingface.zip`

The local generated teacher dataset in this repository is:

```text
teacher/stage1_g31b_v5_full_balanced/
  holmes_lpcvc_sft.jsonl
  stats.json
  images/0_real/
  images/1_fake/
```

Current local generated dataset status:

- rows: `32070`
- labels: `16035 Real`, `16035 AI-Generated`
- all `image` fields resolve relative to `teacher/stage1_g31b_v5_full_balanced/`
- current rows use the generator-only schema: `step2_draft`, not full reviewed `step2_target`

The deterministic derived dataset generated from the existing teacher labels is:

```text
teacher/derived_deterministic_v1/
  derived.jsonl
  manifest.json
```

Its rows keep the original `image` and `original_response`, and add:

- `final_json_target`
- `evidence_trace_target`
- `taxonomy_target`
- `consistency_target`
- `quality_flags`

There is also an external/generated dataset path used by defaults in the student code:

```text
/ssd4/LPCVC2026/holmes_lpcvc3_multi_teacher/stage1_g31b_v5_full_balanced
```

Use whichever dataset path is present and intended for the experiment. For a fully self-contained repo-relative run, pass the local teacher dataset path explicitly.

## Target JSON Semantics

Competition-style Stage 2 output should look like:

```json
{
  "per_criterion": [
    {
      "criterion": "Lighting & Shadows Consistency",
      "evidence": "Short criterion-specific evidence.",
      "aigc score": 0
    }
  ],
  "overall_likelihood": "Real"
}
```

Score semantics:

- `aigc score = 1`: this criterion has explicit AI-generation artifact evidence.
- `aigc score = 0`: no artifact is visible for this criterion, or the criterion is not applicable.
- `overall_likelihood = AI-Generated` whenever at least one criterion has score `1`.
- `overall_likelihood = Real` when all criteria are score `0` and evidence supports a real image.

Teacher conversion may use internal fields such as `proposed_score`, `support_type`, `holmes_span`, `judge_verdict`, and `final_score`. Student training converts the generator-only `step2_draft.per_criterion_draft[*].proposed_score` field into the competition-facing `aigc score` field.

## Teacher Pipeline

Main file:

```text
teacher/convert_holmes_sft.py
```

Derived-data builder:

```text
teacher/build_derived_dataset.py
```

Purpose:

- Read Holmes `SFTDATA.jsonl`.
- Resolve image files from `dataset_huggingface.zip`.
- Derive the fixed overall label from the Holmes image path: `0_real` -> `Real`, `1_fake` -> `AI-Generated`.
- Convert Holmes free-form explanations into LPCVC 8-criterion supervision.
- Materialize images and write JSONL training rows.

Supported pipeline stages:

- `generator_only`: produce `step1_target` and `step2_draft`.
- `full`: generator, normalization, judge, optional specialist, final `step2_target`.
- `review_only`: review an existing draft JSONL.

Supported teacher backends:

- `heuristic`
- `openai_compatible`
- `transformers_gemma4`

Typical generator-only command using the local repo output folder:

```bash
python3 teacher/convert_holmes_sft.py \
  --holmes-root /ssd4/LPCVC2026/dataset/holmes \
  --teacher-backend transformers_gemma4 \
  --model google/gemma-4-e2b-it \
  --pipeline-stage generator_only \
  --balance-label-order \
  --batch-size 8 \
  --max-samples 32070 \
  --output-root teacher/stage1_g31b_v5_full_balanced \
  --overwrite-images
```

Quick smoke test:

```bash
python3 -m py_compile teacher/convert_holmes_sft.py
python3 teacher/convert_holmes_sft.py \
  --holmes-root /ssd4/LPCVC2026/dataset/holmes \
  --teacher-backend heuristic \
  --pipeline-stage generator_only \
  --max-samples 10 \
  --output-root /tmp/lpcvc_teacher_smoke \
  --overwrite-images
```

Build the deterministic derived dataset without regenerating labels:

```bash
python3 teacher/build_derived_dataset.py \
  --input-jsonl teacher/stage1_g31b_v5_full_balanced/holmes_lpcvc_sft.jsonl \
  --output-root teacher/derived_deterministic_v1
```

## Student Training Pipeline

Main files:

- `student/src/dataset.py`
- `student/src/train.py`
- `student/src/inference.py`
- `student/src/evaluate.py`
- `student/src/train_visual_expert.py`
- `student/src/visual_expert.py`
- `student/finetune.sh`

Current student implementation:

- Phase 1 backbone default: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Baseline-only legacy checkpoint: `student/outputs/20260427_105054/checkpoint-7014`.
- Old LoRA adapters are for inference/evaluation baselines only; new Phase 1 runs start from the base model and create a fresh LoRA adapter.
- Previous completed runs used `Qwen/Qwen2.5-VL-3B-Instruct`.
- Training method: 4-bit QLoRA with PEFT LoRA adapters.
- LoRA target modules: attention projections plus MLP projections.
- Optimizer: `paged_adamw_8bit`.
- Precision: `bf16`.
- Gradient accumulation: `8`.
- Loss masking: labels before the assistant response are set to `-100`.
- Image preprocessing: images are loaded as RGB and thumbnail-limited to `1024 x 1024`.
- Logging: `training.log`, TensorBoard logs, checkpoints, and `experiments_summary.json` under each run directory.

Dataset behavior:

- Primary input is the derived dataset: `teacher/derived_deterministic_v1/derived.jsonl`.
- `image` is resolved via the row-level `image_root`, so the derived dataset does not need to copy images.
- Each epoch re-samples one of four tasks per row using the configured task mix:
  - `final_json`
  - `evidence_trace`
  - `taxonomy_classification`
  - `consistency_check`
- When `--visual_expert_path` points to `dataset_logits.jsonl`, training can add an auxiliary distillation loss through a lightweight pooled hidden-state head.

Repo-relative training example:

```bash
cd /ssd4/LPCVC2026/Module-II-Final
python3 student/src/train.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts \
  --output_dir student/outputs \
  --batch_size 2 \
  --epochs 3 \
  --lr 1e-4 \
  --train_mode multitask_sft
```

Recommended Phase 1 task mix:

```bash
--task_mix '{"final_json":0.4,"evidence_trace":0.35,"taxonomy_classification":0.15,"consistency_check":0.1}'
```

Baseline evaluation:

```bash
python3 student/src/evaluate.py \
  --base_model Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter_path student/outputs/20260427_105054/checkpoint-7014 \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts
```

Using the wrapper script:

```bash
cd /ssd4/LPCVC2026/Module-II-Final/student
./finetune.sh \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --derived_data_path /ssd4/LPCVC2026/Module-II-Final/teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir /ssd4/LPCVC2026/Module-II-Final/prompts \
  --output_dir /ssd4/LPCVC2026/Module-II-Final/student/outputs
```

Resume semantics:

- `--resume_from_checkpoint` only continues the same run after interruption.
- Use `--run_name <existing_run> --resume_from_checkpoint True` to resume the latest checkpoint in that run directory.
- To resume a specific checkpoint, pass its explicit `checkpoint-*` path.
- Do not use `--resume_from_checkpoint` as a way to initialize a new run from an old adapter.

Visual expert training:

```bash
python3 student/src/train_visual_expert.py \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --output_dir student/experts/default
```

Phase 1 distillation round:

```bash
python3 student/src/train.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts \
  --output_dir student/outputs \
  --train_mode multitask_sft \
  --task_mix '{"final_json":0.4,"evidence_trace":0.35,"taxonomy_classification":0.15,"consistency_check":0.1}' \
  --visual_expert_path student/experts/default/dataset_logits.jsonl \
  --distill_weight 0.1 \
  --batch_size 2 \
  --epochs 3 \
  --lr 1e-4
```

Inference CLI:

```bash
python3 student/src/inference.py \
  --base_model Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --image_path <image> \
  --prompt_dir prompts \
  --expert_path student/experts/default/expert.pt
```

Evaluation CLI:

```bash
python3 student/src/evaluate.py \
  --base_model Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts
```

## Phase 1 TODO

Status legend:

- `[x]` completed
- `[-]` in progress
- `[ ]` pending

Execution checklist:

- `[x]` Deterministic derived dataset builder implemented and dataset materialized at `teacher/derived_deterministic_v1/derived.jsonl`
- `[x]` Multi-task student pipeline implemented for `final_json`, `evidence_trace`, `taxonomy_classification`, `consistency_check`
- `[x]` Visual expert training and distillation hooks implemented
- `[x]` Baseline backbone fixed to `Qwen/Qwen2.5-VL-3B-Instruct`
- `[x]` Best legacy baseline checkpoint fixed to `student/outputs/20260427_105054/checkpoint-7014`
- `[x]` `resume_from_checkpoint` semantics corrected to mean same-run continuation only
- `[x]` Training, inference, and evaluation CLI smoke-tested
- `[x]` Run baseline evaluation smoke slice for `checkpoint-7014` and confirm the evaluation path is wired end to end
- `[x]` Train full visual expert and write `expert.pt`, `dataset_logits.jsonl`, `metadata.json`
- `[-]` Run Phase 1 round 1 multi-task SFT from base model with no distillation (`student/outputs/phase1_round1_20260526`, GPU 1)
- `[ ]` Evaluate Phase 1 round 1 checkpoint against the legacy baseline
- `[ ]` Run Phase 1 round 2 multi-task SFT with visual expert distillation (`distill_weight=0.1`)
- `[ ]` Evaluate Phase 1 round 2 checkpoint and compare against baseline + round 1
- `[ ]` Select the best Phase 1 checkpoint for downstream inference and further optimization

Current working assumptions:

- Old LoRA adapters are never used as initialization for new multi-task runs.
- Baseline evaluation and later comparisons use the same derived dataset and prompt set.
- The current long-running step is round 1 multi-task SFT from the base model without distillation.
- `student/src/evaluate.py` is functionally correct after malformed-trace hardening, but its current throughput is too slow for large comparison slices; use sample-based reports first.

## Prompt Flow

Stage 1 uses three prompts to reduce overload:

- edge, boundary, texture, resolution, material/object details
- physical/common-sense logic, text/symbols, human/biological integrity
- lighting/shadow consistency, perspective/spatial accuracy

Stage 2 receives all Stage 1 analytical excerpts and must output only the final JSON object.

When modifying prompts, keep these constraints:

- preserve the 8 canonical criterion names exactly
- preserve the JSON keys expected by the evaluator
- avoid extra commentary around JSON
- keep per-step outputs under the competition token limit

## Verification Commands

Syntax check:

```bash
python3 -m py_compile \
  teacher/convert_holmes_sft.py \
  student/src/dataset.py \
  student/src/train.py \
  student/src/inference.py
```

Inspect the local teacher dataset:

```bash
python3 - <<'PY'
import json, collections
from pathlib import Path

root = Path("teacher/stage1_g31b_v5_full_balanced")
path = root / "holmes_lpcvc_sft.jsonl"
labels = collections.Counter()
missing = 0

for line in path.open():
    row = json.loads(line)
    labels[row["step2_draft"]["overall_likelihood"]] += 1
    if not (root / row["image"]).exists():
        missing += 1

print(labels)
print("missing images:", missing)
PY
```

## Known Gaps

- ONNX export, AIMET quantization, AI Hub compile, and final mobile validation are not implemented in this repo yet.
- Full-dataset evaluation is currently expensive because the pipeline performs two generations per sample plus the optional probes.
- `student/finetune.sh` carries environment-specific absolute defaults. Override paths when running from this repository.
- Current local teacher dataset is generator-only. Run `full` or `review_only` if reviewed `step2_target` supervision is required.
- The current Phase 1 default backbone is `Qwen/Qwen2.5-VL-3B-Instruct`; older historical runs in `student/outputs/` include other backbones and should not be treated as current defaults.

## Editing Guidance

- Keep criterion names, output keys, and score semantics stable.
- Prefer repo-relative paths in new documentation and scripts, but note external dataset roots when they are required.
- Do not treat teacher `proposed_score` as final reviewed truth unless the dataset was produced by the full review stage.
- Avoid broad refactors in training code unless the change directly improves reproducibility, CLI usability, or competition compatibility.
