#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=0

# Default Training Arguments
TRAIN_ARGS=(
  --model_name_or_path google/gemma-4-E2B-it
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl
  --prompt_dir prompts
  --output_dir student/outputs
  --train_mode multitask_sft
  --batch_size 4
  --epochs 10
)

# Run training. 
# You can override or add arguments by passing them to this script, 
# e.g., ./finetune.sh --batch_size 2 --lr 5e-5
conda run --no-capture-output -n base python3 student/src/train.py "${TRAIN_ARGS[@]}" "$@"
