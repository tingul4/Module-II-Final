# Holmes AI Image Authenticity Project Technical Overview

## 1. Project Goal

本專案的目標不是只做 `Real / AI-Generated` 二元分類，而是建立一條可訓練、可評估、可部署的 **detector-first + structured explanation** pipeline：

1. 先由 detector 做最終真偽判斷。
2. 再由 Gemma student 產生結構化解釋。
3. 解釋必須對齊固定的 8-criterion task contract，而不是自由文字評論。

輸入是單張影像。核心輸出包含：

- `overall_likelihood`
- `per_criterion`

固定 8 個 criteria 依序為：

1. Lighting & Shadows Consistency
2. Edges & Boundaries
3. Texture & Resolution
4. Perspective & Spatial Relationships
5. Physical & Common Sense Logic
6. Text & Symbols
7. Human & Biological Structure Integrity
8. Material & Object Details

這個 task contract 的目的，是把 authenticity decision 從「單一分數」拆成「可審核的 criterion-level evidence aggregation」。

## 2. Why Fine-Tuning Is Needed

直接使用 base VLM 有三個問題：

1. 它不知道本專案要求的固定 8-criterion schema。
2. 它不會自然遵守 `score semantics`：
   - `aigc score = 1` 代表有明確 artifact evidence
   - `aigc score = 0` 代表沒有 artifact 或不適用
3. 它不會穩定輸出可比較的中介結構，例如 evidence provenance、artifact taxonomy、consistency signal。

因此本專案不是只微調「回答格式」，而是要讓 student 學會以下能力：

- 把影像觀察映射到固定 criteria
- 判斷哪些 evidence 足以支持 positive artifact score
- 把 evidence 壓縮成一致、可解析、可比較的 JSON surface
- 學會 artifact 類型、evidence 來源、score-consistency 之間的關係

## 3. Dataset And Supervision Pipeline

### 3.1 Source Dataset

原始來源是 Holmes dataset，teacher pipeline 先把 Holmes response 轉成 canonical supervision，再產出 deterministic derived dataset。

主要檔案：

- source draft: `teacher/stage1_g31b_v5_full_balanced/holmes_lpcvc_sft.jsonl`
- derived dataset: `teacher/derived_deterministic_v1/derived.jsonl`
- derived manifest: `teacher/derived_deterministic_v1/manifest.json`
- split manifest: `teacher/derived_deterministic_v1/derived_split.json`

資料規模：

- total rows: `32,070`
- `AI-Generated`: `16,035`
- `Real`: `16,035`

train/eval split 採 deterministic stratified split：

- stratify key: `final_json_target.overall_likelihood`
- train/eval ratio: `90/10`
- split seed: `42`

實際分割：

- train: `28,862`
- eval: `3,208`
- eval real/fake: `1,604 / 1,604`

### 3.2 Multi-Teacher Pipeline: Generator, Judge, Specialist

teacher pipeline 的目的不是直接產生最終 student label，而是先把 Holmes response 轉成較高品質、較一致的 canonical supervision。這一層本質上是 **teacher-side label engineering**。

目前 teacher 轉換邏輯可分成三個角色：

1. `generator`
   - 讀入 Holmes 原始 response、canonical criteria、以及 Holmes anchor。
   - 先把 Holmes 自由文字重寫成 `per_criterion_draft`。
   - 每個 criterion 會先產出：
     - `proposed_score`
     - `evidence`
     - `support_type`
     - `holmes_span`
   - generator 的任務是 coverage 優先：先把 Holmes response 映射進 canonical structure。

2. `judge`
   - 對 generator draft 做品質控管。
   - 核心工作是判斷某個 criterion draft 是否應接受、降成 0、或需要 specialist 複查。
   - judge 專注在：
     - Holmes-backed evidence 是否真的對應到該 criterion
     - `image_only` evidence 是否過度猜測
     - 是否存在重複、矛盾或不該保留的 positive score

3. `specialist`
   - 只在高風險 criterion 上被 judge 升級呼叫。
   - 目前特別處理的高風險項包括：
     - Text & Symbols
     - Human & Biological Structure Integrity
     - Perspective & Spatial Relationships
     - Physical & Common Sense Logic
   - specialist 的任務不是重做整張圖判斷，而是針對單一 criterion 做更窄、更保守的確認。

這三個角色的設計目的如下：

- `generator` 解決 coverage 與 canonicalization
- `judge` 解決 over-call、criterion misalignment、support mismatch
- `specialist` 解決高風險 criterion 的誤判成本較高問題

因此 derived dataset 並不是單純把 Holmes response 原文搬過來，而是經過一層結構化與保守化處理之後的 supervision。

### 3.3 Derived Dataset Schema

`derived.jsonl` 每一列都包含 image、原始 Holmes response、以及多個 deterministic supervision targets。

主要欄位如下：

| Field | Meaning | Purpose |
| --- | --- | --- |
| `row_id` | derived row 的穩定索引 | train/eval split、teacher/student 對齊、reproducible eval |
| `image` | 相對影像路徑 | 訓練與推論載入影像 |
| `image_root` | 影像根目錄 | 讓 derived row 可在不同 script 中定位影像 |
| `source` | 來源資料標記 | 保留資料 lineage |
| `original_query` | Holmes 原始 query | 保留來源任務上下文 |
| `original_response` | Holmes 原始文字解釋 | 作為 teacher-origin baseline 與 trace provenance 來源 |
| `step1_target` | image 摘要式 key points | teacher pipeline 內部使用的 image summary |
| `final_json_target` | 最終 task contract JSON | student 最終輸出 supervision |
| `evidence_trace_target` | criterion-level中介 trace | 教 student 先做可審核的 evidence decomposition |
| `taxonomy_target` | artifact taxonomy + support_type | 教 student 學 artifact 類型與 evidence provenance |
| `consistency_target` | score/evidence 是否一致 | 教 student 避免 positive score without grounded evidence |
| `quality_flags` | 潛在 supervision 風險標記 | offline QA 與資料檢查 |

### 3.4 What Each Target Means

#### `final_json_target`

這是 student deployment 最接近的 supervision surface。它只保留：

- 8 個 criterion 的最終 evidence
- 8 個 criterion 的 `aigc score`
- `overall_likelihood`

它回答的是最終任務本身：

- 這張圖在哪些 criteria 上有 artifact evidence？
- 綜合之後應判成 `Real` 還是 `AI-Generated`？

#### `evidence_trace_target`

這是更完整的中介 supervision，除了 score 與 evidence 之外，還保留：

- `support_type`
- `holmes_span`
- `artifact_taxonomy`
- `non_applicable`
- `artifact_score_conflict`

它的功能是把「為什麼這個 score 成立」拆得更細。這對 student 很重要，因為 final JSON 只有結果，沒有 provenance。

#### `taxonomy_target`

`taxonomy_target` 是從 `evidence_trace_target` 再抽出較精簡但更可分類的 supervision，欄位只有：

- `criterion`
- `artifact_taxonomy`
- `support_type`

它不是最終 deployment output，但有明確訓練用途：

1. 教 student 把 artifact 描述壓縮成較穩定的型別，例如：
   - `shadow_mismatch`
   - `edge_discontinuity`
   - `texture_repetition`
   - `anatomy_error`
2. 教 student 分清楚 evidence 的來源型態，而不只是輸出一段自由文字。

這個任務的價值不是「多一個 label 比較漂亮」，而是把 explanation learning 拆成：

- artifact 存不存在
- artifact 屬於哪一類
- 這個判斷是 Holmes 明示、Holmes 暗示、image-only，還是根本 unsupported

這有助於 student 建立更穩定的中介表示，也讓後續 evaluation 可以分開看：

- taxonomy 是否對
- support provenance 是否對
- 最終 JSON 是否對

#### `consistency_target`

`consistency_target` 不是在教 model 產生新 evidence，而是在教它判斷：

- 某個 positive score 是否真的有對應的 grounded artifact evidence
- 某個 `overall_likelihood` 是否與 criterion-level signals 一致

它包含：

- `overall_consistent`
- `expected_overall_likelihood`
- 每個 criterion 的 `consistent` 與 `reason`

這個 task 的設計目的，是降低 student 產生「形式上像 explanation，但其實 score 與 evidence 對不起來」的情況。

#### `quality_flags`

`quality_flags` 是資料品質檢查訊號，不直接作為 student output target。它主要標記：

- `real_has_positive_artifact`
- `fake_has_no_positive_artifact`
- `positive_without_evidence`
- `artifact_score_conflict`

用途是 teacher-side QA 與 offline dataset inspection。

## 4. Student Architecture

### 4.1 Active Inference Architecture

目前 active architecture 為：

**CLIP LoRA detector-first classification + Gemma 4 E2B explanation**

兩個模組的角色分工是固定的：

1. `CLIP LoRA detector`
   - 負責 binary authenticity classification
   - checkpoint: `student/outputs/detectors/holmes_clip_lora_vitl14_336/checkpoints/model_epoch_0.94_0.99.pth`
   - output:
     - `detector_score`
     - `detector_label`
   - 在 `detector_student` 模式下，最終 `overall_likelihood` 由 detector 決定

2. `Gemma 4 E2B multi-task SFT`
   - backbone: `google/gemma-4-E2B-it`
   - method: 4-bit QLoRA
   - 負責生成：
     - evidence trace
     - taxonomy / support provenance
     - consistency-related structure
     - final JSON explanation

這個分工背後的設計目的是：

- 把高風險的 binary decision 交給專門的 detector
- 把較長、較結構化、較可解釋的輸出交給 Gemma

也就是說，本專案不假設單一 VLM 同時最擅長分類與長結構化解釋。

### 4.2 Detector-Student Coupling

在 `detector_student` 模式下，Gemma 仍然會產生自己的 `overall_likelihood` 判斷，但它只保留為診斷欄位：

- `student_overall_likelihood`

對外正式輸出的 `overall_likelihood` 會被 detector 覆蓋。這樣做的理由是：

1. deployment 時只保留一個最終 binary authority
2. evaluation 時可以清楚比較：
   - detector 分類能力
   - student explanation 能力
   - detector 與 student 的差異

## 5. Multi-Task SFT Design

### 5.1 Four Training Tasks

student 目前訓練四個任務：

1. `final_json`
2. `evidence_trace`
3. `taxonomy_classification`
4. `consistency_check`

推薦 task mix：

```json
{"final_json": 0.4, "evidence_trace": 0.35, "taxonomy_classification": 0.15, "consistency_check": 0.1}
```

### 5.2 What Each SFT Task Teaches

#### Task 1: `final_json`

輸出 `final_json_target`。

它教 student 直接完成最終任務格式：

- 8-criterion final evidence
- 8-criterion `aigc score`
- `overall_likelihood`

這是 deployment-facing task，也是最接近最後產品行為的 supervision。

#### Task 2: `evidence_trace`

輸出 `evidence_trace_target`。

它教 student 先產生中介 reasoning surface，而不是直接跳 final answer。這個任務要學的是：

- criterion decomposition
- compact evidence extraction
- support provenance
- artifact taxonomy assignment
- 是否 non-applicable
- score/evidence conflict awareness

這個任務存在的理由，是 final JSON 太壓縮，無法完整教會 model 如何形成最終 decision。

#### Task 3: `taxonomy_classification`

輸出 `taxonomy_target`。

它不是單純「再做一次分類」，而是教 student 把 artifact 描述規整成更穩定的中介語彙。

這個 task 希望 student 學到的能力包括：

- 在固定 criterion 下辨認更細的 artifact subtype
- 把自由 evidence 壓縮為 canonical taxonomy label
- 區分 artifact 類型與 evidence provenance

為什麼這有用：

1. final JSON 只告訴你有沒有 artifact，沒有告訴你 artifact 是哪一型。
2. trace 任務同時要做太多事；獨立 taxonomy task 可以把型別學習從長文字生成中拆出來。
3. evaluation 可以獨立量測 taxonomy accuracy，幫助判斷 student 的 explanation 失敗到底是：
   - 看錯 artifact
   - 分錯 criterion
   - 還是只是文字描述沒對齊

#### Task 4: `consistency_check`

輸出 `consistency_target`。

它要學的不是 visual perception 本身，而是 **decision hygiene**：

- positive score 是否真的有 grounded evidence
- overall decision 是否與 criterion-level positives 一致
- 哪些 criterion 雖然提到東西，但不足以支撐 positive score

這個 task 的存在，是因為 VLM 很容易產生「看起來有理由，但其實分數不該那樣給」的回答。consistency supervision 是在矯正這類結構性錯誤。

### 5.3 What `support_type` Is For

`support_type` 是目前最容易被誤解的欄位，但它其實非常重要。它不是裝飾性 metadata，而是 evidence provenance label。

四種值代表：

- `explicit_holmes`: Holmes response 明確說到這個 artifact
- `implied_holmes`: Holmes response 沒直接點名，但強烈暗示這個 criterion
- `image_only`: 這個 artifact 主要來自影像觀察，而非 Holmes wording 直接支撐
- `unsupported`: 沒有足夠 grounded evidence

對 student 而言，學 `support_type` 的目的有三個：

1. 學會區分「有 artifact」與「有沒有足夠證據支撐 artifact」
2. 避免把弱線索都講成強證據
3. 讓 explanation 不只是內容對，還要知道這個內容是 Holmes-backed 還是 image-only

所以 `support_type` supervision 的核心不是為了部署時直接顯示給使用者，而是為了讓 student 形成更保守、更可追溯的 evidence discipline。

### 5.4 QLoRA Adaptation Scope

目前 student 的 QLoRA 不是對整個 Gemma 全量微調，而是只在一組固定的線性投影層上掛 LoRA adapter。

可先用下表理解目前的 adaptation scope：

| Group | Target modules | Role in model | Expected adaptation effect |
| --- | --- | --- | --- |
| Self-attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 控制 token / vision context 的讀取、匹配、聚合與輸出 | 讓模型更會關注 authenticity task 需要的 visual-textual cues，改善 criterion-level evidence aggregation |
| MLP feed-forward | `gate_proj`, `up_proj`, `down_proj` | 控制 hidden feature 的非線性轉換與 task-specific remapping | 讓模型把既有通用表徵轉成更符合 Holmes task 的 score / evidence / taxonomy 輸出 |
| Not adapted | embeddings, norm, full base weights | 保留 base Gemma 的主要語言與多模態對齊能力 | 降低訓練成本與過度破壞 base model 行為的風險 |

active target modules 為：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

若模型實作中這些層以 `.linear` 子模組形式暴露，訓練程式會自動改抓：

- `q_proj.linear`
- `k_proj.linear`
- `v_proj.linear`
- `o_proj.linear`
- `gate_proj.linear`
- `up_proj.linear`
- `down_proj.linear`

也就是說，QLoRA 主要覆蓋兩大區塊：

1. **self-attention projections**
   - `q / k / v / o`
   - 影響模型如何讀取與聚合 multimodal context

2. **MLP feed-forward projections**
   - `gate / up / down`
   - 影響模型如何把內部特徵轉成更符合本任務的判斷與輸出分布

目前沒有把 LoRA 掛到 embedding、norm、或完整全參數更新上。這樣做的原因是：

- attention projections 最直接影響「看哪裡、怎麼聚合上下文」
- MLP projections 最直接影響「把觀察轉成 task-specific decision / explanation」
- 這組 target modules 是 instruction tuning 與 domain adaptation 中常見且成本效益高的折衷

預期效果是：

- 用少量可訓練參數，讓模型學會 Holmes authenticity task 的固定 schema
- 保留 base Gemma 的大部分通用語言與視覺對齊能力
- 降低 full fine-tuning 的顯存與訓練成本

目前 LoRA 超參數為：

- rank `r = 32`
- `lora_alpha = 64`
- `lora_dropout = 0.05`
- `bias = none`

這代表：

- rank 不是極小的超輕量配置，而是保留一定適配容量
- alpha 與 rank 同步放大，讓 adapter 更新有足夠影響力
- dropout 用來稍微抑制 adapter 過擬合
- 不訓練 bias，保持改動集中在低秩增量上

## 6. Deployment Path

### 6.1 Why Deployment Is A Separate Stage

student 訓練出的主要 artifact 是 **LoRA adapter**，不是可直接上 Android runtime 的完整模型。  
因此 deployment 不能直接拿訓練輸出的 adapter 目錄去包成行動端模型，而是必須經過兩個明確步驟：

1. **merge LoRA adapter 回完整 Hugging Face model**
2. **把 merged model 轉成 LiteRT artifacts，再 bundle 成 `model.litertlm`**

這一段的設計原因是：

- QLoRA / PEFT 訓練時保存的是增量權重，不是完整 base model
- LiteRT export toolchain 需要的是完整可載入的 Hugging Face model directory
- Android runtime 最終吃的是 `model.litertlm`，不是 LoRA checkpoint

因此 deployment path 的本質是：

**training artifact normalization -> runtime-specific export -> mobile bundle packaging**

### 6.2 Active Deployment Workflow

目前 repo 的 active deployment path 是：

1. fine-tune Gemma 4 E2B LoRA adapter
2. 選定要部署的 checkpoint
3. merge adapter 回完整 Hugging Face model
4. 驗證 merged model 可正常載入
5. export LiteRT split artifacts
6. bundle 成 `model.litertlm`
7. 把 `model.litertlm` 交給 Android / LiteRT-LM runtime

對應 CLI：

- merge:
  - `student/src/deployment/merge_student.py`
- export:
  - `student/src/deployment/export_litert_model.py`

### 6.3 Step 1: Merge LoRA Back To A Full Hugging Face Model

merge 的目的，是把：

- base model: `google/gemma-4-E2B-it`
- LoRA adapter: `student/outputs/<run>/<checkpoint>`

合併成一個完整的 Hugging Face model directory。

這一步的必要性在於：

1. export toolchain 不理解「base model + adapter」這種訓練期表示法
2. merged model 才是 deployment conversion 的穩定輸入
3. merged model 也方便先做 packaging environment 下的本地載入驗證

目前 merge 指令路徑為：

```bash
python3 student/src/deployment/merge_student.py \
  --base_model google/gemma-4-E2B-it \
  --adapter_path student/outputs/<run>/checkpoint-<step> \
  --output_dir student/merged_models/gemma4_e2b_latest
```

merge 完的輸出不是最終手機端 artifact，而是 **conversion-only intermediate**。

### 6.4 Step 2: Export To LiteRT And Bundle `model.litertlm`

merge 完之後，使用 `student/src/deployment/export_litert_model.py` 做 LiteRT export。

這個 export stage 會做兩件事：

1. 把 merged Hugging Face Gemma 模型拆成 LiteRT 可用的 `.tflite` 子模組
2. 把這些子模組與 tokenizer / metadata bundle 成單一 `model.litertlm`

指令形式如下：

```bash
PYTHONNOUSERSITE=1 python student/src/deployment/export_litert_model.py \
  --merged_model_dir student/merged_models/gemma4_e2b_latest \
  --output_dir student/mobile_artifacts/gemma4_e2b \
  --prefill_seq_len 128 \
  --kv_cache_max_len 512 \
  --trust_remote_code
```

這裡要特別澄清，最終 bundle 檔名是：

- `model.litertlm`

不是 `.literlm`。

### 6.5 What The Export Produces

export 後的 workspace 會包含兩類 artifact。

#### A. Runtime-facing primary artifact

- `model.litertlm`

這是 Android / LiteRT-LM runtime 實際要載入的主 artifact。

#### B. Split LiteRT artifacts

常見會看到：

- `model_quantized.tflite` 或 `model.tflite`
- `vision_encoder_quantized.tflite`
- `vision_adapter_quantized.tflite`
- `embedder_quantized.tflite`
- `per_layer_embedder_quantized.tflite`

另外還會有：

- `conversion_recipe.json`
- `EXPORT_GUIDE.md`

這些 split `.tflite` 檔主要用途是：

- export inspection
- bundle rebuild
- packaging debug

對部署端而言，真正需要交付給 runtime 的核心 artifact 是 `model.litertlm`。

### 6.6 Why Packaging Uses A Separate Environment

training 與 deployment 不共用環境，原因不是習慣問題，而是 toolchain 需求不同。

訓練環境關心的是：

- PEFT / bitsandbytes / QLoRA
- GPU fine-tuning
- evaluator 與 detector runtime

deployment 環境關心的是：

- merged model local load
- LiteRT Torch export
- LiteRT-LM bundling

repo 目前建議 deployment 使用獨立 `uv` 環境：

```bash
uv venv .venv-google-ai-edge --python 3.11
source .venv-google-ai-edge/bin/activate
uv pip install -r requirements.txt
```

這樣做的原因是：

1. 避免 training 依賴與 LiteRT export 依賴衝突
2. 避免 user-site / system-site package 汙染 export toolchain
3. 讓 merge 驗證與 export 失敗時，比較容易定位是環境問題還是模型問題

### 6.7 What Actually Gets Deployed

從系統設計角度看，部署端不是只有一個模型檔。

目前 active architecture 是：

- detector-first classification
- Gemma structured explanation

因此完整產品路徑其實包含兩個部分：

1. **detector**
   - 負責最終 binary classification
   - 目前是 vendored CLIP LoRA detector checkpoint

2. **Gemma LiteRT-LM bundle**
   - 負責 explanation / trace / final JSON generation
   - 透過 `model.litertlm` 交給行動端 runtime

也就是說，`model.litertlm` 對應的是 Gemma student 這條 explanation path，不是整個 detector-first system 的唯一 artifact。

### 6.8 Deployment-Side Inference Contract

行動端若要重現 repo 的 active inference path，需要維持與 `student/src/inference.py` 一致的 two-stage 行為：

1. 送入影像 + `prompts/evidence_trace.txt`
2. 取得 stage-1 trace JSON
3. 把 trace compact 後，連同影像再次送入 `prompts/stage2.txt`
4. 取得 final JSON
5. 若採 `detector_student` 路徑，最終 `overall_likelihood` 由 detector 覆蓋

所以 deployment 不只是「把一個 LLM 打包上手機」，而是要維持：

- prompt contract
- two-stage generation contract
- detector/student decision contract

任何一段改動，都可能讓離線評估與部署表現失去對齊。

### 6.9 Quantization Across Training And Deployment

本專案有兩種不同目的的量化，不能混在一起看。

先用總表整理：

| Stage | Quantized object | Setting | Bit width / recipe | Main purpose | Not the same as |
| --- | --- | --- | --- | --- | --- |
| Training | Gemma base weights | bitsandbytes QLoRA load | `4-bit NF4` | 降低 fine-tuning 顯存需求 | deployment artifact quantization |
| Training | Optimizer states | `paged_adamw_8bit` | `8-bit` | 降低 optimizer memory | model inference quantization |
| Training | LoRA adapters | low-rank trainable delta | not base-weight 4-bit frozen tensor | 以少量參數做 task adaptation | final mobile artifact |
| Deployment | merged Gemma text / vision LiteRT export | `dynamic_wi8_afp32` | dynamic `8-bit` | 優先壓低 runtime memory pressure | training-time QLoRA |
| Deployment | merged Gemma text / vision LiteRT export | `weight_only_wi4_afp32` | weight-only `4-bit` | 優先縮小檔案體積 | training-time QLoRA |

#### A. 訓練期量化: QLoRA base model quantization

訓練 student 時，Gemma base model 會先以 bitsandbytes 4-bit 方式載入：

- `load_in_4bit = True`
- quant type: `nf4`
- compute dtype: `bf16`
- `bnb_4bit_use_double_quant = True`

這代表：

1. **base model 權重**
   - 以 4-bit NF4 形式保存 / 載入
2. **前向與反向主要計算**
   - 使用 `bfloat16`
3. **LoRA adapter 權重**
   - 不是 4-bit 保存的 frozen base weights，而是額外訓練的低秩增量參數

訓練期量化的目的不是為了直接部署，而是為了：

- 降低 Gemma 4 E2B fine-tuning 的顯存需求
- 讓單機 QLoRA 訓練可行
- 在不全量更新 base model 的前提下，保留足夠的 task adaptation 能力

此外，optimizer 使用：

- `paged_adamw_8bit`

這表示 optimizer state 也做了 8-bit 壓縮，進一步降低訓練期記憶體占用。

#### B. 部署期量化: LiteRT export quantization

部署期量化是發生在 **merge 完 LoRA、得到完整 Hugging Face model 之後**。

也就是說順序是：

1. 4-bit QLoRA 訓練 adapter
2. merge adapter 回完整 model
3. 再把 merged model export 成 LiteRT artifact

部署期目前有兩種主要量化路徑：

1. **預設建議路徑**
   - `dynamic_wi8_afp32`
   - text 與 vision export 都可使用這個 recipe

2. **較小檔案的實驗路徑**
   - `weight_only_wi4_afp32`

預設 export CLI 目前使用：

- `--quantize dynamic_wi8_afp32`
- `--vision_quantize dynamic_wi8_afp32`

其設計重點是：

- 以 8-bit 動態量化為主
- 目標優先放在部署側的 runtime memory 壓力，而不是最小檔案體積

`weight_only_wi4_afp32` 的設計重點則是：

- 權重檔案更小
- 但 runtime 可能因顯式 dequantize 成 float 而承受更高記憶體壓力

因此這兩層量化的角色不同：

- **QLoRA 4-bit**
  - 是訓練期的記憶體節省手段
- **LiteRT dynamic 8-bit / weight-only 4-bit**
  - 是部署期的 runtime / storage trade-off 手段

不能把「訓練時的 4-bit」直接理解成「部署時就是 4-bit 模型」。

## 7. Training Setup

目前 student training 重要設定如下：

- base model: `Gemma 4 E2B`
- training method: 4-bit QLoRA
- base weight quantization: `4-bit NF4`
- LoRA target groups: attention projections + MLP projections
- LoRA hyperparameters: `r=32`, `alpha=64`, `dropout=0.05`
- precision: `bf16`
- optimizer: `paged_adamw_8bit`
- gradient accumulation: `8`
- task mix: `final_json 0.40 / evidence_trace 0.35 / taxonomy 0.15 / consistency 0.10`

training-time eval 的設計如下：

- train split 只用 `train_row_ids`
- held-out 評估用 `eval_row_ids`
- 每個 epoch 跑一次正式 evaluator
- 為了控制成本，training-time slice 固定為 balanced `64` samples
- offline comparison 則可指定更大的 fixed slice，例如 `128`

## 8. Evaluation Design

### 7.1 Metric Families

本專案的 evaluation 分成兩大類：

#### Classification / Structure Metrics

- `overall_accuracy`
- `overall_macro_f1`
- `criterion_macro_f1`
- `json_parse_rate`
- `trace_json_parse_rate`
- `support_type_accuracy`
- `taxonomy_accuracy`
- `consistency_score`
- `real_false_positive_rate`

其中：

- `overall_macro_f1` 是 binary `Real / AI-Generated` macro F1
- `criterion_macro_f1` 是 8 個 criteria 的平均 F1

這兩個不能混為一談。

#### Explanation Metrics

- `BLEU-1`
- `ROUGE-L`
- `METEOR`

它們是對 canonical explanation surface 做 teacher/student 文字對齊比較，不是直接評估 binary classification。

### 7.2 Reported Experiments

本文件目前保留兩組實驗結果，用途不同：

1. **full-eval detector classification**
   - mode: `detector`
   - split: held-out `eval`
   - sample count: `3208`
   - detector threshold: `0.5`
   - 用途: 估計目前 CLIP LoRA detector 在完整 eval split 上的真實 binary classification 表現

2. **training-time epoch eval**
   - mode: `detector_student`
   - split: held-out `eval`
   - slice: balanced `64` samples
   - composition: `32 real + 32 fake`
   - detector threshold: `0.5` for binary rescoring
   - 用途: 比較 `epoch_001 / epoch_002 / epoch_003` 的 checkpoint quality

3. **offline threshold stress test**
   - model: `epoch_001`
   - mode: `detector_student`
   - split: held-out `eval`
   - slice: balanced `128` samples
   - composition: `64 real + 64 fake`
   - slice seed: `42`
   - detector threshold: `0.5`
   - 用途: 觀察 detector threshold 調高後，binary metrics 與 explanation metrics 的 trade-off

對應檔案：

- full-eval detector classification JSON: `student/outputs/gemma4_e2b_epoch_eval_rerun_20260612/full_eval/detector_eval_full_eval_thr050.json`
- training-time reports: `student/outputs/gemma4_e2b_epoch_eval_rerun_20260612/training_eval/epoch_00{1,2,3}.json`
- offline threshold stress test JSON: `student/outputs/gemma4_e2b_epoch_eval_rerun_20260612/full_eval/epoch_001_slice128_thr050.json`
- offline threshold stress test Markdown: `student/outputs/gemma4_e2b_epoch_eval_rerun_20260612/full_eval/epoch_001_slice128_thr050.md`

## 9. Current Experimental Results

### 9.1 Full-Eval Detector Classification

先把 classification 與 explanation 拆開看。

由於 detector-only 模式不需要載入 Gemma，也不需要生成 trace / final JSON，因此可以直接對完整 held-out `eval` split 的 `3208` 筆資料做 full eval。這組數字是目前最應該拿來當 **binary classification 正式結果** 的數字。

設定：

- model: CLIP LoRA detector only
- split: held-out `eval`
- sample count: `3208`
- detector threshold: `0.5`
- output: `student/outputs/gemma4_e2b_epoch_eval_rerun_20260612/full_eval/detector_eval_full_eval_thr050.json`

結果：

| Metric | Value |
| --- | ---: |
| sample_count | 3208 |
| detector_threshold | 0.5000 |
| overall_accuracy | 0.5549 |
| overall_macro_f1 | 0.5080 |
| average_precision | 0.6239 |
| real_false_positive_rate | 0.1365 |
| wall_time_sec | rescored from saved predictions |
| sec_per_sample | rescored from saved predictions |

這組結果的重要性在於：

- `overall_accuracy = 0.5549`
- `overall_macro_f1 = 0.5080`

這兩個數字來自完整 `3208` 筆 held-out eval rows，而不是小型 sample slice，所以它們比先前 `128-slice` 的 threshold stress test 更適合當 **正式 classification 報告值**。

但它們也揭露了一個明確問題：

- `threshold=0.5` 明顯壓低了真圖誤報
- `real_false_positive_rate = 0.1365`
- 但同時也犧牲了不少 fake recall，因此整體 `overall_macro_f1` 降到 `0.5080`

因此目前比較誠實的說法不是「分類已經很強」，而是：

- detector 在完整 eval split 上已經明顯優於隨機猜測
- `threshold=0.5` 較保守，較符合你目前希望降低真圖誤報的設定
- 但 binary classification 仍未達到可宣稱成熟部署的程度

### 9.2 Epoch-by-Epoch Formal Eval

下表只比較正式的 `epoch_001 / 002 / 003` report，不採用舊的 `epoch_001_smoke` artifact。

其中：

- `overall_accuracy` 與 `overall_macro_f1` 已統一用 `threshold=0.5` 從既有 detector scores 重算
- `json_parse_rate`、`criterion_macro_f1`、`support_type_accuracy`、`taxonomy_accuracy`、`BLEU-1`、`ROUGE-L`、`METEOR` 不受 threshold 影響，因此沿用原始 epoch report

| Epoch | overall_accuracy | overall_macro_f1 | json_parse_rate | criterion_macro_f1 | support_type_accuracy | taxonomy_accuracy | BLEU-1 | ROUGE-L | METEOR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch_001 | 0.5625 | 0.5039 | 0.7188 | 0.1912 | 0.4603 | 0.5278 | 0.4128 | 0.3394 | 0.4285 |
| epoch_002 | 0.5625 | 0.5039 | 0.8594 | 0.4398 | 0.5694 | 0.5437 | 0.5211 | 0.4182 | 0.5428 |
| epoch_003 | 0.5625 | 0.5039 | 0.7969 | 0.3861 | 0.5655 | 0.5516 | 0.4923 | 0.3881 | 0.4989 |

關鍵觀察：

- `overall_accuracy` 與 `overall_macro_f1` 三個 epoch 完全一樣，這是因為 `detector_student` 模式下最終 `overall_likelihood` 由 detector 覆蓋，binary label 幾乎不受 Gemma epoch 影響。
- 在 `threshold=0.5` 下，這組 64-sample slice 的 binary 指標是 `overall_accuracy = 0.5625`、`overall_macro_f1 = 0.5039`
- 真正反映 multi-task SFT 是否進步的指標，是：
  - `json_parse_rate`
  - `criterion_macro_f1`
  - `support_type_accuracy`
  - `taxonomy_accuracy`
  - `BLEU-1 / ROUGE-L / METEOR`
- 以正式 epoch eval 而言，**`epoch_002` 是目前最好的 explanation-side checkpoint**。

### 9.3 Best Epoch Interpretation

`epoch_002` 的意義不是把 binary classification 拉高，而是把 student 的 explanation / structure 能力推到目前三個 epoch 中最好的平衡點：

- `json_parse_rate = 0.8594`
- `criterion_macro_f1 = 0.4398`
- `support_type_accuracy = 0.5694`
- `taxonomy_accuracy = 0.5437`
- `BLEU-1 = 0.5211`
- `ROUGE-L = 0.4182`
- `METEOR = 0.5428`

這代表：

1. student 的 structured output 穩定度在 `epoch_002` 明顯優於 `epoch_001`
2. criterion-level artifact reasoning 在 `epoch_002` 才開始形成可見訊號
3. 到 `epoch_003` 時，binary label 沒變，但 explanation-side quality 有輕微回落，顯示 explanation 任務可能在第 2 個 epoch 左右就接近最佳點

### 9.4 Why Explanation Metrics Still Use Slices

`BLEU-1 / ROUGE-L / METEOR` 目前仍維持用 sample slice 做 evaluation，原因不是理論上不能跑 full eval，而是計算成本差異非常大：

- detector-only full eval:
  - 不生成文字
  - 只做影像前向推論
  - `3208` 筆只花約 `68` 秒

- student explanation eval:
  - 每筆都要做 trace generation
  - 還要做 final JSON generation
  - 再做 parse 與 explanation metric 計算
  - full eval 成本會比 detector-only 高很多

所以目前策略是：

- **classification**
  - 用 detector-only full eval 拿正式 `accuracy / macro F1`

- **explanation / structure**
  - 用固定 seed 的 balanced slice 持續比較不同 checkpoint

這樣做的好處是：

- binary classification 可以拿到完整資料集上的準確數字
- explanation 指標仍保有可比較性與可接受的計算成本

### 9.5 Offline Threshold Stress Test (`epoch_001`, slice=128, threshold=0.5)

這組結果不是用來選 checkpoint，而是用來觀察 detector threshold 調高後的代價。

| Metric | Value |
| --- | ---: |
| sample_count | 128 |
| detector_threshold | 0.5000 |
| json_parse_rate | 0.9063 |
| trace_json_parse_rate | 1.0000 |
| overall_accuracy | 0.5313 |
| overall_macro_f1 | 0.4747 |
| criterion_macro_f1 | 0.0071 |
| support_type_accuracy | 0.2793 |
| taxonomy_accuracy | 0.5684 |
| consistency_score | 0.9990 |
| real_false_positive_rate | 0.1406 |
| BLEU-1 | 0.4395 |
| ROUGE-L | 0.4320 |
| METEOR | 0.4775 |
| sec_per_sample | 35.32 |

這組數字要特別小心解讀：

- 它的 primary purpose 是看 `threshold=0.5` 對 binary metrics 的影響
- 它只用了 `epoch_001`
- `criterion_macro_f1 = 0.0071` 是這一組特定 offline run 的結果，不代表 `epoch_002` 或 `epoch_003` 的 explanation-side最佳表現
- 因此它比較像 **failure-mode / calibration study**，不應直接拿來當整個專案的唯一最終 headline result

### 9.6 What These Scores Mean

從目前結果看，可以把系統能力拆成三層：

1. **格式與中介結構能力**
   - 這部分其實已經有不錯進展。
   - `trace_json_parse_rate` 幾乎穩定在 `0.984+`
   - `json_parse_rate` 在最佳 epoch 已達 `0.8594`

2. **解釋對齊能力**
   - `epoch_002` 的 `BLEU-1 / ROUGE-L / METEOR` 顯示 student 已能產出中等程度對齊的 canonical explanation surface
   - 這代表它不是亂講，而是已經部分學到 teacher-style explanation 的詞面與結構

3. **高信度 artifact reasoning 能力**
   - 這部分還不夠成熟
   - `criterion_macro_f1` 雖然在 `epoch_002` 升到 `0.4398`，但距離可宣稱為強健 artifact detector 還有距離
   - `support_type_accuracy` 和 `taxonomy_accuracy` 也顯示 provenance 與 subtype learning 仍在早期階段

## 10. Can These Be Final Report Results?

可以，但**不能全部都當最終 headline KPI**。

### 10.1 Suitable As Final Report Results

以下幾類很適合放進最終報告，因為它們能誠實反映目前系統已經做成的能力：

- 完整 held-out eval split 上的 detector-only `overall_accuracy` 與 `overall_macro_f1`
- detector-first + multi-task SFT 的完整 pipeline 已打通
- held-out eval split 與 epoch-level formal eval 已建立
- `epoch_002` 在 explanation-side 指標上優於 `epoch_001` 與 `epoch_003`
- `json_parse_rate`
- `trace_json_parse_rate`
- `BLEU-1`
- `ROUGE-L`
- `METEOR`
- `taxonomy_accuracy`

這些指標比較容易被解釋成：

- binary classification 在完整 eval split 上的真實表現
- student 是否能穩定輸出結構化結果
- student 是否有學到 teacher-style explanation surface
- student 是否有開始學到 artifact subtype taxonomy

### 10.2 Not Suitable As Sole Final Headline

以下幾類不建議單獨拿大標呈現，否則會很難解釋，甚至會讓讀者誤會系統整體失敗：

- `overall_accuracy = 0.5313` 與 `overall_macro_f1 = 0.4747`
  - 這是 `epoch_001 + threshold=0.5 + 128-slice` 的 calibration stress test 結果
  - 現在已經有完整 `3208` 筆 eval rows 的 detector-only 分數，因此這組更不該被拿來當主 classification headline

- `criterion_macro_f1 = 0.0071`
  - 這個數字太低，而且只代表某一個特定 offline run
  - 如果直接當主結果，會掩蓋其實 `epoch_002` 已有 `0.4398` 的進展

- `support_type_accuracy = 0.2793`
  - 這是困難 supervision 的 early-stage diagnostic
  - 適合放分析段落，不適合當首頁 headline

### 10.3 Better Final Report Framing

比較好的報告寫法應該是：

1. **主結果**
   - 系統已完成 Holmes-derived supervision pipeline
   - student 已具備穩定 structured explanation generation 能力
   - detector 在完整 eval split 上的 `overall_accuracy = 0.5549`、`overall_macro_f1 = 0.5080`（`threshold=0.5`）
   - 最佳 checkpoint 為 `epoch_002`

2. **量化結果**
   - classification 用 full-eval detector-only 結果
   - 用 `epoch_002` 的 explanation-side metrics 當主要 student 結果
   - 把 `epoch_001 threshold=0.5` 結果放成 detector calibration / threshold sensitivity study

3. **誠實揭露限制**
   - binary classification 仍高度受 detector threshold calibration 影響
   - criterion-level artifact reasoning 尚未完全成熟
   - `support_type` 仍是最難學的 supervision 之一

## 11. Design Takeaways

從目前結果看，這個系統的強弱點已經相對清楚：

### 11.1 What Is Working

- teacher-side canonicalization 已成功把 Holmes response 轉成可訓練的 deterministic supervision
- student 已能學到固定 schema 與 canonical explanation surface
- detector-first 架構讓分類與解釋責任分離，分析與部署都較清楚
- multi-task SFT 已建立出可診斷的中介層，而不是只有 final answer
- `epoch_002` 顯示 taxonomy 與 criterion-level reasoning 的 supervision 確實有學習效果
- detector-only full eval 讓 classification 指標可以用完整 eval split 直接量測，不再只能依賴 slice 估計

### 11.2 What Is Not Working Yet

- 即使固定用 `threshold=0.5`，完整 `eval` split 上的 `overall_macro_f1 = 0.5080` 仍不夠高，代表 detector 本身仍有可觀的 calibration / recall 問題
- student 的 criterion-level artifact detection 仍未達可直接部署宣稱的水準
- `support_type` 與 taxonomy 雖然有 supervision，但 student 尚未充分學會 provenance discipline
- 目前的 explanation 比較像「結構對了、詞面部分對齊」，還不是高信度 grounded explanation

## 12. Recommended Next Steps

1. **用 `epoch_002` 當主 student checkpoint 做更完整 offline eval**
   - 不要再以 `epoch_001 threshold=0.5` 當唯一主結果
   - 重新在較大 fixed slice 或 full eval split 上驗證 `epoch_002`

2. **把 detector calibration 與 student explanation 分開報**
   - detector 報完整 eval split 的 `overall_accuracy / overall_macro_f1 / real_false_positive_rate`
   - student 報 parse / taxonomy / criterion / BLEU-ROUGE-METEOR

3. **特別檢查 `support_type` supervision 是否過難**
   - 如果 `support_type_accuracy` 長期偏低，代表 provenance task 可能需要簡化或重設 target granularity

4. **優先追 criterion-level quality，而不是只追 wording**
   - 在 `criterion_macro_f1` 沒有穩定前，單純提高 BLEU/ROUGE 並不代表 authenticity reasoning 真的更好

## 13. One-Sentence Summary

本專案目前已建立完整的 Holmes-to-derived supervision pipeline 與 detector-first student stack；在統一使用 `threshold=0.5` 的設定下，完整 `eval` split 上的 detector-only classification 為 `accuracy 0.5549 / macro F1 0.5080`，而三個 epoch 的正式 held-out eval 中 `epoch_002` 則展現最佳 explanation-side 表現，因此目前最適合把結果寫成「結構化 explanation 與 taxonomy learning 已成立、binary authenticity decision 已具初步能力但仍未成熟」的技術報告，而不是直接宣稱整體分類已成熟。
