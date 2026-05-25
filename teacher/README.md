# Holmes SFT -> LPCVC Multi-Teacher Conversion

This folder contains the Holmes-first multi-teacher conversion pipeline that
rewrites Holmes `SFTDATA.jsonl` into LPCVC-style supervision.

The design goal is:

- keep Holmes as the primary source of evidence
- separate generation from review
- preserve internal provenance for debugging
- emit a final export that looks like official LPCVC-style criterion JSON

The output row keeps both:

- `step2_target`: final export-facing JSON
- `step2_internal`: internal trace of draft, review, and final decision

## What changed from v3

`holmes_lpcvc3` used a single-teacher rewrite with rule repair.

This multi-teacher version uses:

- Holmes anchor stage
- generator draft stage
- judge review stage
- optional specialist stage for high-risk criteria
- deterministic internal final score
- export consistency checks

## End-to-end flow

The intended pipeline is:

1. Read `SFTDATA.jsonl`
2. Resolve the referenced image from `dataset_huggingface.zip`
3. Derive the fixed overall label from the Holmes path
4. Anchor Holmes evidence into LPCVC criteria
5. Ask the generator teacher for an internal draft
6. Normalize the draft before review
7. Ask the judge teacher to review criterion-level positives
8. Optionally ask a specialist for high-risk criteria
9. Produce a single internal final decision per criterion
10. Export the final answer from internal decisions only

## Core policy

### Overall label is fixed

- `0_real` -> `Real`
- `1_fake` -> `AI-Generated`

The teacher does not decide the final overall label.

### Fixed criterion order

The output always contains exactly these 8 criteria in this order:

1. `Lighting & Shadows Consistency`
2. `Edges & Boundaries`
3. `Texture & Resolution`
4. `Perspective & Spatial Relationships`
5. `Physical & Common Sense Logic`
6. `Text & Symbols`
7. `Human & Biological Structure Integrity`
8. `Material & Object Details`

### Score semantics

This definition is shared across prompts, normalization rules, judge logic,
specialist logic, and final export.

- `score = 1`: this criterion contains a clear AI-generated artifact or anomaly
- `score = 0`: this criterion does not contain that type of AI artifact, or the
  criterion is not applicable / not assessable

Important clarifications:

- `score = 0` does not mean "no evidence text"
- real images may still include normality evidence while keeping `score = 0`
- `Not assessable due to lack of relevant content` must always map to `score = 0`

### Holmes-first evidence policy

Each criterion is first anchored into one of:

- `explicit_holmes`
- `implied_holmes`
- `unsupported`

The generator may additionally propose `image_only`, but only as a candidate
support type.

`image_only` does not have the same standing as Holmes-backed evidence:

- `explicit_holmes`: generator may propose directly
- `implied_holmes`: generator may propose directly, but the judge must verify
  that the Holmes wording really supports it
- `image_only`: generator may only propose it as a candidate; it must not be
  treated as a final positive unless the judge accepts it and, when needed, the
  specialist also accepts it

## Teacher and rule interaction

### Stage 1: Holmes anchor

The script reads `original_response` and aligns each criterion to Holmes
evidence. This stage does not create a final score. It only establishes
provenance.

### Stage 2: Generator teacher

The generator sees:

- the image
- the original Holmes query
- the original Holmes response
- the fixed overall label
- the 8 fixed LPCVC criteria
- the Holmes anchor result

The generator returns:

- `step1_target`
- `per_criterion_draft[*].criterion`
- `per_criterion_draft[*].proposed_score`
- `per_criterion_draft[*].evidence`
- `per_criterion_draft[*].support_type`
- `per_criterion_draft[*].holmes_span`

Generator responsibility:

- rewrite Holmes evidence into criterion-level supervision
- preserve the fixed overall label
- keep evidence criterion-aligned
- avoid unsupported positives
- treat `image_only` as a candidate proposal rather than an immediately valid
  positive

Generator must not:

- override the overall label
- decide the final exported score by itself
- emit template-only or content-free evidence
- treat `image_only` evidence as equivalent to Holmes-supported evidence

Generator-only relaxation:

- when running `--pipeline-stage generator_only`, fake samples are allowed to
  preserve more `image_only` candidates before review
- this relaxed behavior is intended to improve draft recall for LPCVC criteria
  that Holmes does not explicitly mention
- the relaxation is strongest for lower-risk criteria and remains stricter for
  high-risk criteria such as text, human/bio, perspective, and physical/common
  sense
- this does not change final acceptance rules; it only keeps more candidate
  evidence in the saved draft

Criterion coverage policy:

- `Lighting & Shadows Consistency`
- `Edges & Boundaries`
- `Texture & Resolution`
- `Perspective & Spatial Relationships`
  - These are high-coverage Holmes-aligned criteria.
  - Generator stage should remain Holmes-first.
  - `image_only` should be used sparingly and mainly as a fallback candidate.

- `Text & Symbols`
- `Human & Biological Structure Integrity`
  - These are low-coverage but Holmes-strong criteria.
  - Holmes often omits them entirely, but when Holmes does mention them, the
    support is usually explicit and high-quality.
  - Stage 1 may keep `image_only` candidates when visual evidence is concrete.
  - `Human & Biological Structure Integrity` may be relaxed slightly more than
    `Text & Symbols` because Gemma often identifies strong biological/anatomy
    failures even when Holmes is silent.

- `Material & Object Details`
  - This is the clearest LPCVC-added criterion.
  - Holmes rarely provides an explicit one-to-one source span for it.
  - Stage 1 may be more permissive in preserving concrete candidate evidence
    here, especially for fake samples, but Stage 2 should remain conservative.

- `Physical & Common Sense Logic`
  - This is a hybrid criterion.
  - Holmes often supports it explicitly, but some cases are broad or weakly
    phrased.
  - Stage 1 may preserve candidate positives when the physical or common-sense
    failure is image-grounded and concrete, but Stage 2 should still scrutinize
    it carefully because this criterion is easy to over-expand.

Practical interpretation:

- high-coverage Holmes-aligned criteria: keep Holmes-first behavior
- low-coverage but Holmes-strong criteria: allow limited `image_only`
  candidates when Holmes is silent
- LPCVC-added criteria: allow broader candidate recall in Stage 1
- final acceptance remains conservative and should still be decided by
  Judge/Specialist plus the final decision rules

### Stage 3: Draft normalization

After generation, the script normalizes the draft:

- canonicalizes criterion names
- fills missing criteria
- repairs malformed score values
- blocks invalid real-sample `image_only` positives
- marks `Not assessable...` entries as non-applicable
- expands weak `implied_holmes` evidence into a usable Holmes sentence when
  possible
- records evidence/score conflicts without treating normalization as the final
  score authority

This stage is rule-based and is meant to stabilize teacher output before review.
It should not silently become the place where semantic score decisions are made.

Recommended internal flags for this stage include:

- `artifact_score_conflict`
- `non_applicable`

### Stage 4: Judge teacher

The judge reviews the normalized draft and returns:

- `accept`
- `downgrade_to_0`
- `needs_specialist_check`
- `score_consistency`
- optionally `recommended_score`

Judge responsibility:

- verify evidence/criterion alignment
- verify support type plausibility
- verify whether `proposed_score`, `support_type`, and evidence are mutually
  consistent
- reject weak, generic, unsupported, or redundant positives
- request specialist review when high-risk criteria remain ambiguous

Judge must not:

- rewrite the full JSON answer
- introduce new criteria on its own
- change the fixed overall label

### Stage 5: Specialist teacher

Specialist review is optional and only runs when the judge requests it for
high-risk criteria:

- `Text & Symbols`
- `Human & Biological Structure Integrity`
- `Perspective & Spatial Relationships`
- `Physical & Common Sense Logic`

Specialist responsibility:

- review only the requested criterion
- return a short criterion-specific verdict and evidence
- return a confidence level
- avoid re-judging the whole image

Recommended specialist outputs:

- `criterion`
- `verdict`
- `reason`
- `evidence`
- `confidence: high | medium | low`

### Stage 6: Internal final decision

Each criterion receives a single internal final decision:

- `step2_internal.per_criterion_draft[*].final_score`

This is the only score that should be treated as authoritative before export.

Recommended decision table:

- For `Real`, `final_score` remains `0`
- For `Fake`:
  - `judge = downgrade_to_0` -> `0`
  - `judge = needs_specialist_check` -> specialist decides
  - `judge = accept` and evidence is artifact-specific, support is valid, and
    the criterion is not non-applicable -> `1`
  - otherwise -> `0`

Additional constraints:

- `image_only` can become `1` only after explicit judge acceptance, and after
  specialist acceptance when the criterion is high-risk
- if `artifact_score_conflict = true`, final decision should prefer the evidence
  semantics plus judge recommendation over the generator's raw score

### Stage 7: Export

`step2_target` is generated only from internal final decisions and cleaned
evidence. Export must not independently re-judge the sample.

## Internal schema

`step2_internal` keeps pipeline trace fields such as:

- `proposed_score`
- `support_type`
- `holmes_span`
- `artifact_score_conflict`
- `non_applicable`
- `judge_verdict`
- `judge_reason`
- `score_consistency`
- `recommended_score`
- `specialist_used`
- `specialist_verdict`
- `specialist_confidence`
- `final_score`

The key invariant is:

- `final_score` is the single authoritative score before export

## Official export schema

`step2_target` uses the final export-facing format:

```json
{
  "overall_likelihood": "AI-Generated",
  "per_criterion": [
    {
      "criterion": "Lighting & Shadows Consistency",
      "score": 1,
      "evidence": "Shadows fall in conflicting directions relative to the light source."
    }
  ]
}
```

## Export consistency rules

Before export, the script enforces these invariants:

1. If `judge_verdict = downgrade_to_0`, then `final_score` must be `0`
2. If evidence is `Not assessable due to lack of relevant content`, then
   `final_score` must be `0`
3. For fake samples, if a criterion has Holmes-backed artifact evidence and the
   judge accepts it, `final_score` must not remain `0`
4. For real samples, exported scores must remain `0`
5. `step2_target.per_criterion[*].score` must exactly equal
   `step2_internal.per_criterion_draft[*].final_score`
6. If `final_score = 1`, evidence must not be empty, `Not assessable...`, or a
   generic template sentence
7. If `support_type = image_only` and `final_score = 1`, the sample must retain
   a non-empty judge or specialist justification for why that positive was accepted

## Evidence cleaning rules

The export layer is allowed to clean evidence, but not to change its meaning.

Allowed cleaning:

- remove markdown prefixes like `**Line segments**:`
- remove section-label prefixes
- collapse whitespace
- reduce evidence to a single complete short sentence
- replace malformed or empty evidence with `Not assessable...` when appropriate

Disallowed export behavior:

- re-judging the criterion
- changing the final score independently of internal decisions
- emitting `...` truncation artifacts
- emitting placeholder fragments like `None.`
- collapsing a criterion to a single alias token such as `lighting.`

## Prompt expectations

### Generator prompt requirements

The generator prompt should explicitly state:

- the overall label is fixed
- `score = 1` means the criterion contains an AI artifact
- `score = 0` means no such artifact or not applicable
- obvious fake artifact evidence should not be paired with `0`
- evidence must be short, complete, and criterion-aligned
- `holmes_span` should be a readable supporting phrase or sentence, not a lone alias
- `image_only` may be proposed only as a candidate and does not become a final
  positive without later review

### Judge prompt requirements

The judge prompt should explicitly state:

- `accept` only when evidence matches the criterion
- `downgrade_to_0` for unsupported, generic, weak, or redundant positives
- `needs_specialist_check` for uncertain high-risk criteria
- the overall label must remain fixed
- the judge must assess whether `proposed_score`, evidence, and `support_type`
  are consistent with one another
- the judge should emit `score_consistency`, and may emit `recommended_score`

### Specialist prompt requirements

The specialist prompt should explicitly state:

- only review the requested criterion
- return a short complete evidence sentence
- do not rewrite the whole sample
- do not emit markdown, commentary, or ellipsis-heavy fragments
- return a confidence value alongside the verdict

## CLI reference

Most common arguments:

- `--holmes-root`
- `--output-root`
- `--max-samples`
- `--teacher-backend`
- `--model`
- `--disable-judge`
- `--enable-specialist`
- `--batch-size`

## Quick smoke test

```bash
python -m py_compile convert_holmes_sft.py
python convert_holmes_sft.py \
  --holmes-root /path/to/LPCVC/dataset/holmes \
  --teacher-backend heuristic \
  --enable-specialist \
  --max-samples 10 \
  --output-root /path/to/LPCVC/holmes_lpcvc3_multi_teacher/test_run
```

## Current limitations

- no native resume support
- specialist currently reuses the same configured backend unless extended
- prompt quality can still improve even when rule-based export is stable
- some criterion-mapping policies such as `Overall Hue` -> `Material & Object Details`
  may still need refinement after pilot review
