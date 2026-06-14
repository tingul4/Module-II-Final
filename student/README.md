# Student Model Pipeline

This folder contains the active student-side stack for Holmes-derived AI image authenticity:

- `Gemma 4 E2B` multi-task SFT for evidence / taxonomy / consistency / final JSON
- `CLIP LoRA detector` for binary real-vs-AI classification

The active detector-first policy is:

1. detector decides `overall_likelihood`
2. Gemma produces the structured explanation payload

## Components

- `src/train.py`: multi-task QLoRA trainer
- `src/inference.py`: two-stage inference CLI, with optional `detector_student` mode
- `src/evaluate.py`: held-out evaluation for `student`, `teacher`, `detector`, and `detector_student`
- `src/utils/`: shared schema, split, and model helpers
- `src/detectors/holmes_clip_lora/`: vendored Holmes CLIP LoRA detector code
- `src/deployment/merge_student.py`: LoRA merge helper
- `src/deployment/export_litert_model.py`: LiteRT split-model + `.litertlm` export CLI
- `outputs/detectors/holmes_clip_lora_vitl14_336/`: detector config + checkpoint artifact

## Training Data

Primary input:

```text
teacher/derived_deterministic_v1/derived.jsonl
```

The train/eval split is persisted beside the derived dataset as `derived_split.json` by default.
The active default is a deterministic stratified `90/10` split on `final_json_target.overall_likelihood`.

## Student Training

The multi-task SFT dataset deterministically re-samples one task per row per epoch:

- `final_json`
- `evidence_trace`
- `taxonomy_classification`
- `consistency_check`

Recommended task mix:

```bash
--task_mix '{"final_json":0.4,"evidence_trace":0.35,"taxonomy_classification":0.15,"consistency_check":0.1}'
```

Train:

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

Relevant split flags:

- `--eval_ratio 0.1`
- `--split_seed 42`
- `--split_manifest_path <optional custom path>`
- `--regenerate_split`

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

## Detector

Detector artifact root:

```text
student/outputs/detectors/holmes_clip_lora_vitl14_336/
```

Included in-repo:

- `config_train.yaml`
- `checkpoints/model_epoch_0.94_0.99.pth`

Not vendored in-repo:

- OpenAI CLIP base weights `ViT-L-14-336px.pt`

You must provide the base CLIP weights path explicitly for detector evaluation or detector-student inference.

Detector retraining CLI:

```bash
python3 student/src/detectors/holmes_clip_lora/train.py \
  --config student/outputs/detectors/holmes_clip_lora_vitl14_336/config_train.yaml \
  --clip_weights /ssd4/LPCVC2026/bk/AIGI-Holmes/pretrained/clip/ViT-L-14-336px.pt
```

## Inference

Student-only:

```bash
python3 student/src/inference.py \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --image_path <image> \
  --prompt_dir prompts
```

Detector-first classification plus Gemma explanation:

```bash
python3 student/src/inference.py \
  --prediction_source detector_student \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --image_path <image> \
  --prompt_dir prompts \
  --detector_checkpoint_path student/outputs/detectors/holmes_clip_lora_vitl14_336/checkpoints/model_epoch_0.94_0.99.pth \
  --detector_clip_weights /ssd4/LPCVC2026/bk/AIGI-Holmes/pretrained/clip/ViT-L-14-336px.pt \
  --detector_threshold 0.34
```

`detector_student` output includes:

- `detector_score`
- `detector_label`
- `student_overall_likelihood`
- final `overall_likelihood` overridden by the detector label

## Evaluation

Install the shared project dependencies before running training, evaluation, or deployment tools:

```bash
python3 -m pip install -r requirements.txt
python3 -m nltk.downloader wordnet omw-1.4
```

Student held-out eval:

```bash
python3 student/src/evaluate.py \
  --prediction_source student \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts \
  --split eval \
  --output_path student/outputs/student_eval.json
```

Teacher baseline on the same held-out split:

```bash
python3 student/src/evaluate.py \
  --prediction_source teacher \
  --teacher_jsonl_path teacher/stage1_g31b_v5_full_balanced/holmes_lpcvc_sft.jsonl \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --split eval \
  --output_path student/outputs/teacher_eval.json
```

Detector-only held-out eval:

```bash
python3 student/src/evaluate.py \
  --prediction_source detector \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --split eval \
  --detector_checkpoint_path student/outputs/detectors/holmes_clip_lora_vitl14_336/checkpoints/model_epoch_0.94_0.99.pth \
  --detector_clip_weights /ssd4/LPCVC2026/bk/AIGI-Holmes/pretrained/clip/ViT-L-14-336px.pt \
  --detector_threshold 0.34 \
  --output_path student/outputs/detector_eval.json
```

Detector + student eval:

```bash
python3 student/src/evaluate.py \
  --prediction_source detector_student \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --prompt_dir prompts \
  --split eval \
  --detector_checkpoint_path student/outputs/detectors/holmes_clip_lora_vitl14_336/checkpoints/model_epoch_0.94_0.99.pth \
  --detector_clip_weights /ssd4/LPCVC2026/bk/AIGI-Holmes/pretrained/clip/ViT-L-14-336px.pt \
  --detector_threshold 0.34 \
  --output_path student/outputs/detector_student_eval.json
```

Evaluation writes Markdown reports when `--output_path` is provided, alongside the JSON report file.

Default explanatory metrics:

- `BLEU-1`
- `ROUGE-L`
- `METEOR`

Optional:

- `CIDEr` via `--enable_cider` when the extra dependency is installed

## Deployment

Merge:

```bash
python3 student/src/deployment/merge_student.py \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --output_dir student/merged_models/gemma4_e2b_latest
```

Prepare LiteRT export workspace:

```bash
python3 student/src/deployment/export_litert_model.py \
  --merged_model_dir student/merged_models/gemma4_e2b_latest \
  --output_dir student/mobile_artifacts/gemma4_e2b \
  --dry_run
```

This writes:

- `conversion_recipe.json`
- `EXPORT_GUIDE.md`

If split LiteRT `.tflite` files already exist, the same script can rebuild `model.litertlm`.
