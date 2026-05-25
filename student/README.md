# Student VLM Training and Inference

This folder contains the student-side implementation for LPCVC 2026 Track 3. The student model learns from Holmes-derived teacher supervision and is expected to produce the two-stage response used by the competition: visual analysis first, then strict JSON synthesis.

## Components

- `src/dataset.py`: JSONL dataset loader, prompt loader, multi-task sample construction, image loading, and assistant-only loss masking.
- `src/train.py`: QLoRA/LoRA SFT training entry point using Hugging Face `Trainer`.
- `src/inference.py`: fixed-path two-stage inference smoke test.
- `finetune.sh`: convenience wrapper for training with environment-specific defaults.
- `outputs/`: previous LoRA adapter runs and training logs.

## Expected Dataset

Training expects a directory or JSONL file in the teacher format:

```text
teacher/stage1_g31b_v5_full_balanced/
  holmes_lpcvc_sft.jsonl
  images/0_real/
  images/1_fake/
```

Each JSONL row should include:

```json
{
  "image": "images/1_fake/example.jpg",
  "source": "holmes_sft",
  "original_query": "...",
  "original_response": "...",
  "step1_target": "Key points: ...",
  "step2_draft": {
    "overall_likelihood": "AI-Generated",
    "per_criterion_draft": [
      {
        "criterion": "Lighting & Shadows Consistency",
        "proposed_score": 1,
        "evidence": "..."
      }
    ]
  }
}
```

`HolmesSFTDataset` resolves images relative to the JSONL directory. If an image is not found there, it falls back to `/ssd4/LPCVC2026/dataset/holmes`.

## Training Behavior

For each item, the dataset randomly samples one of two tasks:

- Stage 1 analysis: one prompt from `stage1.txt` is paired with `original_response` as the assistant target.
- Stage 2 JSON synthesis: `stage2.txt` plus `original_response` is paired with a converted JSON target.

The Stage 2 conversion maps:

- `step2_draft.per_criterion_draft[*].proposed_score` -> `per_criterion[*]["aigc score"]`
- `step2_draft.per_criterion_draft[*].evidence` -> `per_criterion[*].evidence`
- `step2_draft.overall_likelihood` -> `overall_likelihood`

The processor chat template is used when available. Labels before the assistant response are masked to `-100`, so the loss is computed only on the answer.

## Model and Training Setup

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

Repo-relative example:

```bash
cd /ssd4/LPCVC2026/Module-II-Final
python3 student/src/train.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --data_path teacher/stage1_g31b_v5_full_balanced \
  --prompt_dir prompts \
  --output_dir student/outputs \
  --batch_size 2 \
  --epochs 3 \
  --lr 1e-4
```

Use `--local_files_only` when the base model is already cached and network access should be avoided.

Resume from a checkpoint:

```bash
python3 student/src/train.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --data_path teacher/stage1_g31b_v5_full_balanced \
  --prompt_dir prompts \
  --output_dir student/outputs \
  --resume_from_checkpoint True
```

Wrapper-script example:

```bash
cd /ssd4/LPCVC2026/Module-II-Final/student
./finetune.sh \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --data_path /ssd4/LPCVC2026/Module-II-Final/teacher/stage1_g31b_v5_full_balanced \
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

## Inference

`src/inference.py` is currently a smoke-test script with hard-coded paths:

- `base_model_name`
- `adapter_path`
- `image_path`
- `val_stage1_path`
- `val_stage2_path`

It performs the intended competition-style flow:

1. Load the base Qwen VL model.
2. Load the PEFT LoRA adapter.
3. Run all Stage 1 prompts on the image.
4. Concatenate the Stage 1 answers into the Stage 2 prompt.
5. Generate the final JSON response.

Run after editing paths:

```bash
python3 student/src/inference.py
```

## Sanity Checks

Compile check:

```bash
python3 -m py_compile student/src/dataset.py student/src/train.py student/src/inference.py
```

Inspect a single dataset item shape through the trainer logs by starting a short run with `--epochs 1 --max_steps` support is not currently exposed, so use a small dataset copy for quick end-to-end tests.

## Current Limitations

- Inference is not yet a general CLI.
- ONNX export, AIMET quantization, and Snapdragon validation are not implemented here.
- `train.py` defaults to `Qwen/Qwen2-VL-2B-Instruct`, while the documented completed runs used `Qwen/Qwen2.5-VL-3B-Instruct`.
- Stage 2 training currently expects generator-only `step2_draft`; reviewed full-pipeline teacher rows with `step2_target` would need a small dataset-loader extension.
