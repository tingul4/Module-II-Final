# Teacher Data Conversion

This folder contains the Holmes-to-LPCVC conversion pipeline used to build supervised fine-tuning data for the student VLM.

The conversion goal is to turn Holmes `SFTDATA.jsonl` rows into image-text examples aligned with LPCVC Track 3:

- fixed `Real` / `AI-Generated` overall label
- 8 canonical LPCVC criteria
- short criterion-level evidence
- a Stage 1 analysis target
- a Stage 2 structured target or draft

## Files

- `convert_holmes_sft.py`: conversion script.
- `stage1_g31b_v5_full_balanced/holmes_lpcvc_sft.jsonl`: local generated teacher dataset.
- `stage1_g31b_v5_full_balanced/stats.json`: conversion stats for the local dataset.
- `stage1_g31b_v5_full_balanced/images/`: materialized images used by the local JSONL.

## Source Data

The script expects the original Holmes dataset layout:

```text
/ssd4/LPCVC2026/dataset/holmes/
  SFTDATA.jsonl
  dataset_huggingface.zip
```

Labels are derived from the Holmes image path:

- `0_real` -> `Real`
- `1_fake` -> `AI-Generated`

The teacher model does not decide the final overall label.

## Local Generated Dataset

The checked-in/generated local dataset is:

```text
teacher/stage1_g31b_v5_full_balanced/
```

Current status:

- rows: `32070`
- labels: `16035 Real`, `16035 AI-Generated`
- images: stored under `images/0_real/` and `images/1_fake/`
- schema: generator-only rows with `step2_draft`

Example row shape:

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
        "evidence": "...",
        "support_type": "explicit_holmes",
        "holmes_span": "..."
      }
    ]
  }
}
```

`student/src/dataset.py` converts this `step2_draft` format into the competition-facing `per_criterion` plus `aigc score` JSON during training.

## Criteria

The output always uses these 8 criteria in this exact order:

1. `Lighting & Shadows Consistency`
2. `Edges & Boundaries`
3. `Texture & Resolution`
4. `Perspective & Spatial Relationships`
5. `Physical & Common Sense Logic`
6. `Text & Symbols`
7. `Human & Biological Structure Integrity`
8. `Material & Object Details`

Score semantics:

- `1`: clear AI-generated artifact or anomaly for that criterion.
- `0`: no artifact for that criterion, or the criterion is not applicable.

`Not assessable due to lack of relevant content` must map to score `0`.

## Pipeline Stages

`convert_holmes_sft.py` supports three pipeline modes:

- `generator_only`: reads Holmes data, asks the generator backend for draft supervision, normalizes it, and writes `step2_draft`.
- `full`: generator plus judge review, optional specialist review, internal final decisions, and export-facing `step2_target`.
- `review_only`: starts from an existing draft JSONL and runs the review/export portion.

The local `stage1_g31b_v5_full_balanced` dataset was produced in generator-only form.

## Teacher Backends

Supported backends:

- `heuristic`: rule-based smoke-test backend.
- `openai_compatible`: OpenAI-compatible `/chat/completions` server.
- `transformers_gemma4`: local Transformers image-text model backend.

Useful backend arguments:

- `--model`
- `--judge-model`
- `--specialist-model`
- `--api-base`
- `--judge-api-base`
- `--specialist-api-base`
- `--api-key-env`
- `--device`
- `--torch-dtype`
- `--batch-size`

## Generate Draft Data

Repo-relative generator-only example:

```bash
cd /ssd4/LPCVC2026/Module-II-Final
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

This writes:

```text
<output-root>/holmes_lpcvc_sft.jsonl
<output-root>/stats.json
<output-root>/images/...
```

## Run a Smoke Test

```bash
cd /ssd4/LPCVC2026/Module-II-Final
python3 -m py_compile teacher/convert_holmes_sft.py
python3 teacher/convert_holmes_sft.py \
  --holmes-root /ssd4/LPCVC2026/dataset/holmes \
  --teacher-backend heuristic \
  --pipeline-stage generator_only \
  --max-samples 10 \
  --output-root /tmp/lpcvc_teacher_smoke \
  --overwrite-images
```

## Review Existing Drafts

Use `review_only` when draft rows already exist and you want reviewed `step2_target` output:

```bash
python3 teacher/convert_holmes_sft.py \
  --pipeline-stage review_only \
  --draft-jsonl teacher/stage1_g31b_v5_full_balanced/holmes_lpcvc_sft.jsonl \
  --teacher-backend openai_compatible \
  --model <judge-model-name> \
  --output-root /tmp/lpcvc_teacher_reviewed \
  --enable-specialist
```

`review_only` reads images relative to the draft JSONL parent unless another image root is wired in code.

## Full Pipeline Output

The full pipeline keeps an internal trace:

```json
{
  "step2_internal": {
    "overall_likelihood": "AI-Generated",
    "per_criterion_draft": [
      {
        "criterion": "Lighting & Shadows Consistency",
        "proposed_score": 1,
        "support_type": "explicit_holmes",
        "holmes_span": "...",
        "judge_verdict": "accept",
        "final_score": 1
      }
    ]
  }
}
```

It also exports:

```json
{
  "step2_target": {
    "overall_likelihood": "AI-Generated",
    "per_criterion": [
      {
        "criterion": "Lighting & Shadows Consistency",
        "score": 1,
        "evidence": "..."
      }
    ]
  }
}
```

Note that the full teacher export uses `score`, while the student training target currently emits the competition prompt key `aigc score`.

## Quality Policies

The converter is Holmes-first:

- `explicit_holmes`: directly supported by Holmes wording.
- `implied_holmes`: strongly implied by Holmes wording.
- `image_only`: candidate evidence from the image, not final truth by itself.
- `unsupported`: no usable support.

Important rules:

- Real samples should not receive positive artifact scores.
- Unsupported positives are downgraded.
- `image_only` positives are conservative and require review in the full pipeline.
- High-risk criteria are `Text & Symbols`, `Human & Biological Structure Integrity`, `Perspective & Spatial Relationships`, and `Physical & Common Sense Logic`.
- `Material & Object Details` should focus on material/surface realism, not geometry, lighting, text, or anatomy.

## Stats

The local `stats.json` currently reports:

- `requested_rows`: `64997`
- `written_rows`: `32070`
- `processed_rows`: `32070`
- `generator_only_rows`: `32070`
- `batch_retry_count`: `233`
- `teacher_error_count`: `243`
- `json_repair_count`: `437`
- `stage1_material_boundary_downgrade_count`: `3042`
- `real_image_only_block_count`: `2472`
- `criterion_fill_in_count`: `44`

Use these numbers as conversion diagnostics, not model-quality metrics.

## Current Limitations

- No native resume support.
- The local dataset is generator-only and has not passed the full judge/specialist export path.
- `heuristic` is useful for smoke tests, not high-quality teacher labels.
- Full deployment artifacts such as ONNX, AIMET quantization, and Qualcomm AI Hub compilation are outside this teacher folder.
