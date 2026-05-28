#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Default Training Arguments
TRAIN_ARGS=(
  --model_name_or_path google/gemma-4-E2B-it
  --derived_data_path /ssd4/LPCVC2026/Module-II-Final/teacher/derived_deterministic_v1/derived.jsonl
  --prompt_dir /ssd4/LPCVC2026/Module-II-Final/prompts
  --output_dir /ssd4/LPCVC2026/Module-II-Final/student/outputs
  --train_mode multitask_sft
  --batch_size 4
  --epochs 10
)

# Run training. 
# You can override or add arguments by passing them to this script, 
# e.g., ./finetune.sh --batch_size 2 --lr 5e-5
conda run --no-capture-output -n base python3 /ssd4/LPCVC2026/Module-II-Final/student/src/train.py "${TRAIN_ARGS[@]}" "$@"
