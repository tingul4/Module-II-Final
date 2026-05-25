#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Default Training Arguments
TRAIN_ARGS=(
  --model_name_or_path Qwen/Qwen2-VL-2B-Instruct
  --data_path /ssd4/LPCVC2026/holmes_lpcvc3_multi_teacher/stage1_g31b_v5_full_balanced
  --batch_size 4
  --epochs 10
)

# Run training. 
# You can override or add arguments by passing them to this script, 
# e.g., ./finetune.sh --eval_steps 100 --lr 5e-5
conda run --no-capture-output -n base python3 /ssd4/LPCVC2026/student/src/train.py "${TRAIN_ARGS[@]}" "$@"
