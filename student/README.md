# Student Model Pipeline

This folder contains the active student-side pipeline for Holmes-derived AI image authenticity reasoning. The active backbone is **`google/gemma-4-E2B-it`**. Older `Qwen` runs remain in `outputs/` as historical baselines only.

## Components

- `src/train.py`: multi-task QLoRA trainer
- `src/inference.py`: two-stage image-to-JSON inference CLI
- `src/evaluate.py`: offline evaluation plus HTML report output
- `src/train_visual_expert.py`: handcrafted-feature visual expert trainer
- `src/visual_expert.py`: visual expert feature extraction and inference helpers
- `src/merge_student.py`: LoRA merge helper
- `src/export_litert_model.py`: LiteRT split-model + `.litertlm` export CLI
- `src/task_utils.py`: active task/schema helpers
- `src/lpcvc_utils.py`: legacy compatibility shim
- `finetune.sh`: convenience wrapper

## Training Data

Primary input:

```text
teacher/derived_deterministic_v1/derived.jsonl
```

Each row provides:

- `final_json_target`
- `evidence_trace_target`
- `taxonomy_target`
- `consistency_target`
- `quality_flags`

Images remain relative to the original teacher dataset through `image` + `image_root`.

## Training Behavior

The dataset deterministically re-samples one task per row per epoch:

- `final_json`
- `evidence_trace`
- `taxonomy_classification`
- `consistency_check`

Recommended task mix:

```bash
--task_mix '{"final_json":0.4,"evidence_trace":0.35,"taxonomy_classification":0.15,"consistency_check":0.1}'
```

The student keeps the current multi-task QLoRA design:

- 4-bit NF4 loading
- LoRA adapters
- attention + MLP projection targets
- `bf16`
- gradient checkpointing
- `paged_adamw_8bit`
- optional visual expert distillation via `--visual_expert_path`
- fixed-step sample evaluation reports during training via `training_eval/step_<N>.{json,html}`

## Active Backbone Policy

- Active backbone: `google/gemma-4-E2B-it`
- Historical baseline: `student/outputs/20260427_105054/checkpoint-7014`
- Do not initialize new training runs from old adapters
- `--resume_from_checkpoint` means same-run continuation only

The model loader is now Gemma-first and uses a generic image-text path instead of Qwen-specific branching.

## Train

```bash
cd /ssd4/LPCVC2026/Module-II-Final
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

Smoke run:

```bash
cd /ssd4/LPCVC2026/Module-II-Final
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

Wrapper script:

```bash
cd /ssd4/LPCVC2026/Module-II-Final/student
./finetune.sh --batch_size 2 --epochs 3
```

## Visual Expert

```bash
python3 student/src/train_visual_expert.py \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --output_dir student/experts/default
```

Distillation round:

```bash
python3 student/src/train.py \
  --model_name_or_path google/gemma-4-E2B-it \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts \
  --output_dir student/outputs \
  --visual_expert_path student/experts/default/dataset_logits.jsonl \
  --distill_weight 0.1
```

## Inference

```bash
python3 student/src/inference.py \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --image_path <image> \
  --prompt_dir prompts
```

The pipeline is:

1. generate `evidence_trace`
2. synthesize `final_json`
3. optionally fuse visual expert scores

## Evaluation

```bash
python3 student/src/evaluate.py \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts
```

Evaluation writes HTML reports when `--output_path` is provided.

## Deployment

Primary deployment path:

1. merge LoRA into a full HF model
2. validate merged-model local inference
3. export LiteRT split `.tflite` artifacts plus `model.litertlm`
4. run on Android with MediaPipe LLM Inference API or another LiteRT-LM consumer

The Android app-side inference contract should match `student/src/inference.py`: one stage with `prompts/evidence_trace.txt`, then one stage with `prompts/stage2.txt` plus the compacted stage-1 trace JSON. If you need a human-readable 8-criteria report, render it from the final JSON in the app layer.

Merge:

```bash
python3 student/src/merge_student.py \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --output_dir student/merged_models/gemma4_e2b_latest
```

Prepare LiteRT export workspace:

```bash
python3 student/src/export_litert_model.py \
  --merged_model_dir student/merged_models/gemma4_e2b_latest \
  --output_dir student/mobile_artifacts/gemma4_e2b \
  --dry_run
```

This writes:

- `conversion_recipe.json`
- `EXPORT_GUIDE.md`

If split LiteRT `.tflite` files already exist, the same script can rebuild `model.litertlm`.

For a smaller experimental mobile bundle, this repo has already validated:

```bash
python3 student/src/export_litert_model.py \
  --merged_model_dir student/merged_models/gemma4_e2b_round1_checkpoint4000 \
  --output_dir student/mobile_artifacts/gemma4_e2b_round1_checkpoint4000_minimal_wi4 \
  --quantize weight_only_wi4_afp32 \
  --vision_quantize weight_only_wi4_afp32 \
  --prefill_seq_len 128 \
  --kv_cache_max_len 512 \
  --trust_remote_code \
  --keep_temporary_files
```

That export produced a `model.litertlm` of about `2.64 GB` decimal, close to the official LiteRT community Gemma 4 E2B size range.

## Notes

- `student/merged_models/`, `student/mobile_artifacts/`, `student/reports/`, and `student/experts/` are generated artifacts.
- Each active training run should start with a short smoke run before the long run.
- Use `--max_steps` on smoke runs so they stay bounded and predictable.
- Inspect `training.log` for loss, learning rate, grad norm, and ETA.
- Inspect `training_eval/step_<N>.html` to compare prediction output against fixed-slice ground truth during training.
- Legacy filenames and historical run folders may still contain `lpcvc` or `qwen`; treat them as old artifacts, not active architecture.
