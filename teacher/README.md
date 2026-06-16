# Teacher Data Conversion

This folder builds Holmes-derived supervision for the student authenticity model. The active teacher design is a Holmes-first multi-teacher pipeline that preserves the fixed overall label from the source image path, rewrites Holmes explanations into the 8 canonical authenticity criteria, and exports supervision that can be consumed directly by the student multitask QLoRA pipeline.

## Files

- `convert_holmes_sft.py`: Holmes conversion pipeline
- `build_derived_dataset.py`: deterministic multi-task supervision builder
- `stage1_g31b_v5_full_balanced/`: local generated teacher dataset
- `derived_deterministic_v1/`: deterministic student-training dataset

## Source Data

Expected Holmes layout:

```text
../dataset/holmes/
  SFTDATA.jsonl
  dataset_huggingface.zip
```

Labels come from the Holmes image path:

- `0_real` -> `Real`
- `1_fake` -> `AI-Generated`

The teacher does not decide the overall label. It converts Holmes supervision into criterion-level evidence under a fixed label.

## Local Teacher Dataset

```text
teacher/stage1_g31b_v5_full_balanced/
  holmes_lpcvc_sft.jsonl
  stats.json
  images/...
```

The filename `holmes_lpcvc_sft.jsonl` is legacy only. The active interpretation is "Holmes-derived 8-criterion teacher supervision."

Current checked-in snapshot:

- rows: `32070`
- labels: `16035 Real`, `16035 AI-Generated`
- note: this on-disk snapshot was generated earlier in `generator_only` mode

Default pipeline policy in code:

- backend: `transformers_gemma4`
- generator model: `google/gemma-4-31B-it`
- judge model: `google/gemma-4-31B-it`
- specialist model: `google/gemma-4-31B-it`
- pipeline stage: `full`
- judge: enabled by default
- specialist: enabled by default

## Canonical Criteria

1. `Lighting & Shadows Consistency`
2. `Edges & Boundaries`
3. `Texture & Resolution`
4. `Perspective & Spatial Relationships`
5. `Physical & Common Sense Logic`
6. `Text & Symbols`
7. `Human & Biological Structure Integrity`
8. `Material & Object Details`

## Active Teacher Flow

The intended teacher flow is:

1. Read Holmes `SFTDATA.jsonl`
2. Resolve the referenced image from `dataset_huggingface.zip`
3. Derive the fixed overall label from the Holmes image path
4. Anchor Holmes evidence into the 8 canonical criteria
5. Run the generator teacher to produce a draft
6. Normalize the draft before review
7. Run the judge teacher to review criterion-level positives
8. Optionally run the specialist teacher on judge-escalated high-risk criteria
9. Produce a single final internal decision per criterion
10. Export final teacher supervision for the student dataset builder

Teacher roles:

- `generator`: rewrites Holmes explanations into criterion-level draft supervision
- `judge`: checks support type, evidence alignment, and positive-score validity
- `specialist`: reviews ambiguous high-risk criteria such as text, anatomy, perspective, and physical/common-sense logic

## Generate Teacher Data

Full multi-teacher generation is the default path:

```bash
python3 teacher/convert_holmes_sft.py \
  --holmes-root ../dataset/holmes \
  --teacher-backend transformers_gemma4 \
  --model google/gemma-4-31B-it \
  --judge-model google/gemma-4-31B-it \
  --specialist-model google/gemma-4-31B-it \
  --pipeline-stage full \
  --balance-label-order \
  --batch-size 1 \
  --max-samples 32070 \
  --output-root teacher/stage1_g31b_v5_full_balanced \
  --overwrite-images
```

Optional modes:

- `--pipeline-stage generator_only`: save draft supervision before judge/specialist review
- `--pipeline-stage review_only`: review an existing draft JSONL with judge/specialist only

## Build Derived Data

```bash
python3 teacher/build_derived_dataset.py \
  --input-jsonl teacher/stage1_g31b_v5_full_balanced/holmes_lpcvc_sft.jsonl \
  --output-root teacher/derived_deterministic_v1
```

The builder reuses existing teacher labels and emits:

- final structured decision target
- evidence trace target
- taxonomy target
- consistency target

The builder accepts both:

- `generator_only` teacher rows with `step2_draft`
- `full` multi-teacher rows with `step2_internal` and `step2_target`

No new teacher, judge, or specialist model is called in this step. It deterministically converts teacher outputs into student-side multitask supervision.

## Teacher Baseline Evaluation

The student evaluator can score teacher outputs directly on the same held-out split used for student checkpoints.
This baseline reads `step2_internal`, falling back to `step2_draft`, then normalizes the teacher row into the same canonical evaluation surface used for student predictions.

```bash
python3 student/src/evaluate.py \
  --prediction_source teacher \
  --teacher_jsonl_path teacher/stage1_g31b_v5_full_balanced/holmes_lpcvc_sft.jsonl \
  --derived_data_path teacher/derived_deterministic_v1/derived.jsonl \
  --split eval \
  --output_path student/outputs/teacher_eval.json
```

## Pipeline Modes

`convert_holmes_sft.py` supports:

- `full`
- `generator_only`
- `review_only`

Supported backends:

- `heuristic`
- `openai_compatible`
- `transformers_gemma4`

## Notes

- The teacher side still contains some legacy internal names from earlier project phases; treat them as implementation history, not active project framing.
- The active design assumes a larger teacher model and a smaller student model: teacher supervision can come from `Gemma 4 31B IT`, while the student training and deployment path can still target `Gemma 4 E2B`.
- The active deployment toolchain lives on the student side and targets Google AI Edge / MediaPipe.
