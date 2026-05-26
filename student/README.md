# Student VLM Training and Inference

This folder contains the student-side implementation for LPCVC 2026 Track 3. The current pipeline uses a deterministic derived dataset built from the existing teacher labels, then trains a multi-task QLoRA student to emit evidence traces and final competition JSON.

## Components

- `src/dataset.py`: JSONL dataset loader, prompt loader, multi-task sample construction, image loading, and assistant-only loss masking.
- `src/train.py`: QLoRA/LoRA SFT training entry point using Hugging Face `Trainer`.
- `src/inference.py`: two-stage inference CLI for evidence trace plus final JSON.
- `src/evaluate.py`: offline evaluation and probe CLI.
- `src/train_visual_expert.py`: lightweight handcrafted-feature visual expert trainer.
- `src/visual_expert.py`: feature extraction, expert MLP, and expert prediction helpers.
- `finetune.sh`: convenience wrapper for training with environment-specific defaults.
- `outputs/`: previous LoRA adapter runs and training logs.

## Expected Dataset

Primary training input is the derived dataset:

```text
teacher/derived_deterministic_v1/
  derived.jsonl
  manifest.json
```

Each JSONL row should include:

```json
{
  "row_id": 0,
  "image": "images/1_fake/example.jpg",
  "image_root": "teacher/stage1_g31b_v5_full_balanced",
  "original_response": "...",
  "final_json_target": {
    "overall_likelihood": "AI-Generated",
    "per_criterion": []
  },
  "evidence_trace_target": {
    "overall_likelihood": "AI-Generated",
    "per_criterion": [
      {
        "criterion": "Lighting & Shadows Consistency",
        "score": 1,
        "evidence": "...",
        "support_type": "explicit_holmes",
        "holmes_span": "...",
        "artifact_taxonomy": "shadow_mismatch",
        "non_applicable": false,
        "artifact_score_conflict": false
      }
    ]
  },
  "taxonomy_target": {},
  "consistency_target": {},
  "quality_flags": []
}
```

The derived rows keep `image` relative and store `image_root`, so images are resolved without copying them into the derived output directory.

## Training Behavior

For each epoch, the dataset deterministically re-samples one task per row from the configured task mix:

- `final_json`: synthesize the final competition JSON from a structured evidence trace.
- `evidence_trace`: output the full 8-criterion evidence trace.
- `taxonomy_classification`: predict `artifact_taxonomy` and `support_type` per criterion.
- `consistency_check`: judge whether criterion scores are consistent with grounded evidence.

Default task mix:

- `final_json`: `0.40`
- `evidence_trace`: `0.35`
- `taxonomy_classification`: `0.15`
- `consistency_check`: `0.10`

The processor chat template is used when available. Labels before the assistant response are masked to `-100`, so the loss is computed only on the answer. If `--visual_expert_path` points to `dataset_logits.jsonl`, `train.py` also adds an auxiliary distillation loss through a pooled hidden-state head.

## Model and Training Setup

Execution policy for the current Phase 1 pipeline:

- Use `Qwen/Qwen2.5-VL-3B-Instruct` as the only training backbone.
- Treat `student/outputs/20260427_105054/checkpoint-7014` as the baseline checkpoint for inference and evaluation only.
- Do not initialize new multi-task SFT runs from an old LoRA adapter.
- `--resume_from_checkpoint` is only for continuing the same run directory after interruption.

`src/train.py` currently supports Qwen VL model classes through model-name matching:

- names containing `Qwen2.5-VL` use `Qwen2_5_VLForConditionalGeneration`
- names containing `Qwen2-VL` use `Qwen2VLForConditionalGeneration`
- other names fall back to `AutoModelForCausalLM`

Training uses:

- 4-bit NF4 BitsAndBytes loading
- PEFT LoRA adapters
- LoRA rank `32`, alpha `64`, dropout `0.05`
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- `bf16`
- gradient checkpointing
- `paged_adamw_8bit`
- cosine LR scheduler with `warmup_ratio=0.05`
- TensorBoard logging

## Run Training

Baseline evaluation:

```bash
cd /ssd4/LPCVC2026/Module-II-Final
python3 student/src/evaluate.py \
  --base_model Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter_path student/outputs/20260427_105054/checkpoint-7014 \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts
```

Repo-relative example:

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

Recommended task mix for Phase 1:

```bash
--task_mix '{"final_json":0.4,"evidence_trace":0.35,"taxonomy_classification":0.15,"consistency_check":0.1}'
```

Use `--local_files_only` when the base model is already cached and network access should be avoided.

Resume from a checkpoint:

```bash
python3 student/src/train.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts \
  --output_dir student/outputs \
  --run_name <existing_run> \
  --resume_from_checkpoint True
```

`--resume_from_checkpoint True` looks for the latest `checkpoint-*` under `student/outputs/<existing_run>/`. To resume from a specific checkpoint, pass the explicit checkpoint path instead.

Phase 1 round 2 with visual expert distillation:

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

Wrapper-script example:

```bash
cd /ssd4/LPCVC2026/Module-II-Final/student
./finetune.sh \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --derived_data_path /ssd4/LPCVC2026/Module-II-Final/teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir /ssd4/LPCVC2026/Module-II-Final/prompts \
  --output_dir /ssd4/LPCVC2026/Module-II-Final/student/outputs \
  --batch_size 2 \
  --epochs 3
```

Note: `finetune.sh` has absolute defaults for this machine. Override `--data_path`, `--prompt_dir`, and `--output_dir` when running from a different checkout.

## Outputs

Each training run writes to:

```text
student/outputs/<timestamp>/
```

Expected files include:

- `training.log`
- `tensorboard_logs/`
- intermediate checkpoints
- final LoRA adapter files such as `adapter_model.safetensors`
- processor/tokenizer files
- `experiments_summary.json`

Previous completed run summaries in this repo include:

- `20260423_165003`: `Qwen/Qwen2.5-VL-3B-Instruct`, batch size `2`, epochs `10`, LR `1e-4`
- `20260424_210806`: `Qwen/Qwen2.5-VL-3B-Instruct`, batch size `8`, epochs `10`, LR `1e-4`
- `20260427_105054/checkpoint-7014`: current best legacy checkpoint, kept as baseline only

## Visual Expert

Train the lightweight handcrafted-feature expert:

```bash
python3 student/src/train_visual_expert.py \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --output_dir student/experts/default
```

The output directory contains:

- `expert.pt`
- `dataset_logits.jsonl`
- `metadata.json`

`dataset_logits.jsonl` can be passed to `train.py --visual_expert_path` for distillation. `expert.pt` can be passed to `inference.py --expert_path` for lightweight score fusion.

## Inference

`src/inference.py` is now a CLI:

```bash
python3 student/src/inference.py \
  --base_model Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --image_path <image> \
  --prompt_dir prompts \
  --expert_path student/experts/default/expert.pt
```

It runs:

1. Evidence-trace generation.
2. Final JSON synthesis from the trace.
3. Optional visual-expert score fusion.

## Evaluation

```bash
python3 student/src/evaluate.py \
  --base_model Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts
```

The evaluation script reports:

- JSON parse rate
- overall accuracy and macro F1
- per-criterion F1
- support-type accuracy
- taxonomy accuracy
- consistency score
- Real false-positive rate
- blank-image, shuffle, and oracle-trace probes

Recommended comparison set:

- legacy baseline: `student/outputs/20260427_105054/checkpoint-7014`
- new multi-task SFT run without distillation
- new multi-task SFT run with `--distill_weight 0.1`

## Sanity Checks

Compile check:

```bash
python3 -m py_compile \
  student/src/dataset.py \
  student/src/train.py \
  student/src/inference.py \
  student/src/evaluate.py \
  student/src/visual_expert.py \
  student/src/train_visual_expert.py
```

Inspect a single dataset item shape through the trainer logs by starting a short run with `--epochs 1 --max_steps` support is not currently exposed, so use a small dataset copy for quick end-to-end tests.

## Current Limitations

- ONNX export, AIMET quantization, and Snapdragon validation are not implemented here.
- Distillation uses a lightweight auxiliary head and expert logits; it is not DPO/ORPO or full collaborative decoding.
- The handcrafted visual expert is meant for low-cost artifact priors, not as a standalone SOTA detector.
