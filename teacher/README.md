# Teacher Data Conversion

This folder builds Holmes-derived supervision for the student authenticity model. The teacher pipeline is Holmes-first: it rewrites Holmes explanations into the 8 canonical authenticity criteria and preserves the fixed overall label derived from the source image path.

## Files

- `convert_holmes_sft.py`: Holmes conversion pipeline
- `build_derived_dataset.py`: deterministic multi-task supervision builder
- `stage1_g31b_v5_full_balanced/`: local generated teacher dataset
- `derived_deterministic_v1/`: deterministic student-training dataset

## Source Data

Expected Holmes layout:

```text
/ssd4/LPCVC2026/dataset/holmes/
  SFTDATA.jsonl
  dataset_huggingface.zip
```

Labels come from the Holmes image path:

- `0_real` -> `Real`
- `1_fake` -> `AI-Generated`

## Local Teacher Dataset

```text
teacher/stage1_g31b_v5_full_balanced/
  holmes_lpcvc_sft.jsonl
  stats.json
  images/...
```

The filename `holmes_lpcvc_sft.jsonl` is legacy only. The active interpretation is "Holmes-derived 8-criterion draft supervision."

Current local dataset:

- rows: `32070`
- labels: `16035 Real`, `16035 AI-Generated`
- schema: generator-only `step2_draft`

## Canonical Criteria

1. `Lighting & Shadows Consistency`
2. `Edges & Boundaries`
3. `Texture & Resolution`
4. `Perspective & Spatial Relationships`
5. `Physical & Common Sense Logic`
6. `Text & Symbols`
7. `Human & Biological Structure Integrity`
8. `Material & Object Details`

## Generate Draft Data

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

No new teacher or judge model is called in this step.

## Pipeline Modes

`convert_holmes_sft.py` supports:

- `generator_only`
- `full`
- `review_only`

Supported backends:

- `heuristic`
- `openai_compatible`
- `transformers_gemma4`

## Notes

- The teacher side still contains some legacy internal names from earlier project phases; treat them as implementation history, not active project framing.
- The active deployment toolchain lives on the student side and targets Google AI Edge / MediaPipe.
