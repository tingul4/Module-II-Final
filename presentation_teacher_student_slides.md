# 期末簡報逐頁文案：教師模型與學生模型方法分析

這份文案預設講述時間為 13 到 15 分鐘，保留 2 到 3 分鐘問答。主軸聚焦在教師端資料轉換、學生端微調與推論、最新的 detector 融合實驗，並簡述 LiteRT / `.litertlm` 部署端流程。

## Slide 1. 題目與任務定義

### 投影片文案

- 題目：Holmes-derived AI Image Authenticity Project
- 任務：輸入一張圖片，輸出 8 個固定 authenticity criteria 的 JSON 判斷結果
- 輸出欄位：`per_criterion` 與 `overall_likelihood`
- 決策規則：只要任一 criterion 的 `aigc score = 1`，整體就是 `AI-Generated`
- 重點：這不是一般二分類，而是可解釋的結構化判斷

### 講者備註

我們的目標不是只回答這張圖是真是假，而是要指出模型是根據哪一類視覺證據做出判斷。專案固定使用 8 個 criterion，讓輸出格式一致，也讓後續訓練、評估與部署比較容易對齊。

## Slide 2. 為什麼要做 Teacher -> Student

### 投影片文案

- 原始 Holmes supervision 有豐富解釋，但不是可直接訓練的最終目標格式
- 我們先用較大的 teacher model 把 Holmes 的自然語言解釋轉成固定 schema
- 再用 student model 學會穩定輸出這個 schema
- 核心價值：把自然語言說明轉成可監督、可評估的結構化任務

### 講者備註

如果直接拿 Holmes 的原始說明去做 end-to-end 學習，輸出格式會不穩，評估也很難一致。所以這個專案先用較大的 Gemma 4 31B teacher 做 supervision conversion，再讓較小的 student 去學固定格式的輸出。這樣的 teacher-student 分工比較符合能力與成本配置。

## Slide 3. 教師端整體做法

### 投影片文案

- 主要腳本：`teacher/convert_holmes_sft.py`
- 輸入：Holmes `SFTDATA.jsonl` 與對應圖片
- 標籤不是重新判定，而是直接由圖片路徑繼承
- `0_real -> Real`，`1_fake -> AI-Generated`
- 核心任務：Holmes-first rewrite，把 Holmes 解釋映射到固定 8 準則
- 主力 teacher 設定：`google/gemma-4-31B-it`
- 流程包含 `generator`、`judge`、`specialist`

### 講者備註

這裡最重要的是，教師端不是重新替資料打標，而是保留 Holmes 的原始標籤，再把原始 explanation 轉成我們專案需要的結構。實作上我們把大 teacher 拆成三個角色，分別負責生成、審查與高風險 criterion 的專項複核，所以 teacher 端本質上是資料轉換與品質控制流程。

## Slide 4. 教師端細節：怎麼把 Holmes 變成 8 準則 supervision

### 投影片文案

- 先從 Holmes response 抽出 anchor，對每個 criterion 建立對應線索
- support type 分成 4 類：
- `explicit_holmes`
- `implied_holmes`
- `image_only`
- `unsupported`
- generator 先產出每個 criterion 的 `proposed_score`、`evidence`、`support_type`、`holmes_span`
- judge 再審查 evidence 與 criterion 是否對齊
- specialist 專門處理高風險 criterion：文字、人體、生物結構、透視、物理常識
- finalizer 根據 judge / specialist 結果輸出每個 criterion 的最終內部決策

### 講者備註

這一頁想傳達的是，teacher 端不是一次生成就直接收下。它其實是完整多教師流程：先做 Holmes anchor，再用 generator 生成 draft，接著由 judge 複核，必要時再交 specialist。最後 finalizer 會把 image-only positives、重複 positives、與 Real 圖上的不合理正例一起清掉，所以這部分比較像 supervision engineering，而不是一般 caption rewrite。

### 圖示建議

- 畫成流程圖：`Holmes response -> anchor extraction -> Gemma 4 31B generator -> Gemma 4 31B judge -> Gemma 4 31B specialist -> final structured supervision`

## Slide 5. 教師端產物與資料規模

### 投影片文案

- 第一層產物：Holmes-derived teacher supervision
- 第二層產物：deterministic derived dataset
- `teacher/stage1_g31b_v5_full_balanced/stats.json`
- 最終寫出 `32070` 筆資料
- `teacher/derived_deterministic_v1/manifest.json`
- `16035 Real / 16035 AI-Generated`，資料完全平衡
- full teacher row 會保留：
- `step2_target`
- `step2_internal`
- 每筆 derived row 包含：
- `final_json_target`
- `evidence_trace_target`
- `taxonomy_target`
- `consistency_target`
- `quality_flags`

### 講者備註

這個資料流的重點是，teacher 端先把大模型的判斷收斂成 deterministic supervision，student 端再用 derived dataset 做 QLoRA。也就是說，學生不是直接學大模型原始輸出，而是學整理過的最終 teacher decision、evidence trace、taxonomy 跟 consistency 任務。

## Slide 6. 學生端訓練目標：不是只學 final JSON

### 投影片文案

- 學生主幹模型：`google/gemma-4-E2B-it`
- 訓練方式：multi-task QLoRA SFT
- 不是單一任務微調，而是同時學 4 個任務
- `final_json`
- `evidence_trace`
- `taxonomy_classification`
- `consistency_check`
- 推薦 task mix：`0.4 / 0.35 / 0.15 / 0.1`

### 講者備註

我們不希望模型只背 final answer，所以把 supervision 拆成四個層次。它先學怎麼組織 evidence，再學 taxonomy 與 consistency，最後才學 final JSON。這樣的設計也讓 teacher 端輸出的資訊能被完整利用，不會浪費掉 judge 與 specialist 已經整理好的結構。

## Slide 7. 學生端模型與訓練設定

### 投影片文案

- 主要腳本：`student/src/train.py`
- 訓練法：`4-bit NF4 QLoRA + PEFT LoRA`
- LoRA target modules 包含 attention 與 MLP projection
- `q_proj / k_proj / v_proj / o_proj`
- `gate_proj / up_proj / down_proj`
- 主要設定：
- `bf16`
- `paged_adamw_8bit`
- `gradient accumulation = 8`
- `gradient checkpointing`
- `lr = 1e-4`
- `epochs = 3`
- loss masking：assistant 之前的 token 設為 `-100`

### 講者備註

這裡可以把學生模型定位成一個在 Gemma 4 E2B 上做輕量化 adaptation 的方案。專案不是 full fine-tune，而是 QLoRA。這樣比較符合資源限制，也比較容易快速迭代。程式裡也保留了 visual expert distillation 的鉤子，但目前主線 run 的 `distill_weight = 0.0`，不是這次主要成果。

## Slide 8. 學生端微調流程與 checkpoint 選擇

### 投影片文案

- 微調主線 run：`student/outputs/gemma4_e2b_round1_20260527`
- 在 full run 前先做 bounded smoke run：
- `student/outputs/gemma4_e2b_smoke_fix`
- 正式訓練過程會固定步數輸出：
- `training.log`
- `training_eval/step_<N>.json`
- `training_eval/step_<N>.html`
- 固定步數 sample eval 會檢查：
- final JSON parse status
- predicted / gold `overall_likelihood`
- raw prediction text
- 當前 deployment candidate 明確選為：`checkpoint-4000`

### 講者備註

這一頁想強調我們不是只跑一次 full training 然後直接拿最後 checkpoint。流程上先做 smoke run，確認 Gemma 4 E2B QLoRA 可以正常載入、開始訓練、寫 checkpoint，再跑正式訓練。訓練中每隔固定步數還會做 sample evaluation，所以後續選 checkpoint 時，不是只看 loss，而是會對照中間推論結果與離線切片表現。

## Slide 9. 學生端推論流程

### 投影片文案

- 主要腳本：`student/src/inference.py`
- 推論採兩階段，不是單步輸出
- Stage 1：先生成 `evidence_trace`
- Stage 2：把壓縮後的 trace 餵回模型，生成 `final_json`
- 若 trace JSON 解析失敗，系統會用更大的 token budget 重試一次
- 重點：先形成中間證據，再做最終決策

### 講者備註

這是整個學生端最值得強調的設計。因為如果直接要模型吐 final JSON，常見問題是格式不穩定，或理由很空。現在改成先產生 evidence trace，再用 trace 合成 final JSON，等於把推論拆成比較可控的兩段。

### 圖示建議

- 畫成流程圖：`Image + evidence_trace prompt -> evidence_trace -> compact trace -> final_json prompt -> final_json`

## Slide 10. 部署端流程：從 checkpoint 到 Android on-device

### 投影片文案

- 目標不是只停在本地推論，而是把 student checkpoint 轉成可在手機執行的 artifact
- 部署主線：
- `checkpoint-4000`
- `merge_student.py`
- `export_litert_model.py`
- `model.litertlm + split LiteRT .tflite assets`
- 官方部署順序：
- LoRA adapter merge 成完整 Hugging Face model
- 轉出 LiteRT artifact
- 打包 `.litertlm`
- 串接 Android app runtime
- 目前狀態：
- LiteRT export 已驗證
- Android runtime integration 仍是 pending

### 講者備註

這一頁只要講清楚我們不是做完 training 就結束，而是有把模型往手機端部署路徑推進。實際流程是先把 LoRA adapter merge 回完整模型，再透過 LiteRT export 轉成 Android 端可以吃的 artifact，最後打包成 `.litertlm` 與對應的 split `.tflite`。目前這條鏈在 workstation 上已經驗證到 export 成功，但 Android runtime 的最終整合還不是這次報告主角。

### 圖示建議

- 畫成流程圖：`Gemma checkpoint -> merge_student.py -> merged HF model -> export_litert_model.py -> model.litertlm + split tflite -> Android app`

## Slide 11. 評估與訓練監控怎麼做

### 投影片文案

- 訓練過程會固定步數輸出 `training_eval/step_<N>.json` 與 `.html`
- `training.log` 會持續記錄：
- `epoch`
- `epoch_step`
- `global_step`
- `loss`
- `lr`
- `grad_norm`
- `ETA`
- 離線評估指標包含：
- JSON parse rate
- trace JSON parse rate
- overall accuracy
- macro F1
- support type / taxonomy / consistency

### 講者備註

這個專案不只看最後分類對不對，也看中間結構有沒有成形。也就是說，evaluation 不是單一 accuracy，而是同時看 parse rate、criterion-level F1，還有 trace 的 support type 與 consistency 表現。這樣比較能反映模型到底是在理解任務，還是在碰運氣。

## Slide 12. Lightweight eval 設計

### 投影片文案

- 這次 Gemma 主線不是直接跑 full validation，而是先做 lightweight pseudo-validation
- 固定切片：`128` 筆，`64 Real / 64 AI-Generated`
- 切片來源：`teacher/derived_deterministic_v1/derived.jsonl`
- 固定 seed：`42`
- 比較 3 個 Gemma job：
- `ckpt4000_two_stage`
- `ckpt4000_single_stage`
- `ckpt6015_two_stage`
- 每個 job 都輸出：
- report JSON / HTML
- per-sample prediction JSONL
- raw log

### 講者備註

這一頁主要是先說明實驗設計，不要讓後面的數字看起來像是 full benchmark。因為兩階段推論成本高，所以我先用固定的 128 筆 balanced slice 做 checkpoint 與 inference-mode 比較。這樣的好處是速度可控，而且不同方法都能在完全相同的樣本上對照。

## Slide 13. Gemma 主線結果與問題定位

### 投影片文案

- 原始 lightweight report：
- `ckpt4000_single_stage`: `acc = 0.492`
- `ckpt4000_two_stage`: `acc = 0.414`
- `ckpt6015_two_stage`: `acc = 0.352`
- 問題不是 parse 壞掉，而是 Gemma 幾乎不願意判 `AI-Generated`
- `ckpt4000_single_stage` prediction 分布：
- `Real = 118`
- `AI-Generated = 6`
- `Uncertain = 2`
- `Parse fail = 2`
- 對 binary task contract 做 normalization 後：
- `ckpt4000_single_stage` 提升到 `acc = 0.516`, `macro F1 = 0.391`

### 講者備註

這一頁的重點不是說 Gemma 已經很好，而是把問題講清楚。原始 report 看起來比亂猜還差，但深入檢查後發現，除了模型本身偏向判 Real，還有一部分是 evaluation contract 沒對齊。因為這個任務是 binary，只要任一 criterion 為正，就應該把 `overall_likelihood` 視為 `AI-Generated`。做完 normalization 後，Gemma 的觀察分數有回升，但整體分類能力仍然不夠強。

## Slide 14. 為什麼新增 detector：CLIP LoRA 比 Gemma 更適合先做二分類

### 投影片文案

- 新增外部 detector baseline：`AIGI-Holmes` CLIP LoRA
- checkpoint：
- `/ssd4/LPCVC2026/bk/AIGI-Holmes/checkpoints_clip_lora/clip_lora_holmes1/model_epoch_0.94_0.99.pth`
- 在相同 `128` 筆切片上：
- threshold `0.50`：`acc = 0.547`, `macro F1 = 0.480`
- threshold `0.34`：`acc = 0.695`, `macro F1 = 0.687`
- 結論：detector score 很有用，但 threshold 必須校準
- 新增的 detector 執行路徑已固定到 GPU 相容環境：
- `student/src/run_clip_detector_eval.py`

### 講者備註

因為 Gemma 主線最大的問題是分類能力不夠穩，所以我先找一個已經訓練好的外部 detector 當 baseline。結果很清楚，同一批 128 筆資料上，CLIP LoRA 的 binary classification 明顯比 Gemma 好。這代表如果我們的短期目標是先把整體分類拉起來，用 detector-first 的方向是合理的。

## Slide 15. Detector-first fusion 實驗與目前最佳配置

### 投影片文案

- 先保留 Gemma 產生 8 criteria explanation
- 再把 CLIP LoRA detector 當成最終分類主訊號
- 最佳觀察規則：
- `score = 0.95 * detector_score + 0.05 * gemma_positive_flag`
- threshold `0.32`
- 在同一個 `128` 筆切片上：
- best detector-only：`acc = 0.695`, `macro F1 = 0.687`
- best fusion：`acc = 0.703`, `macro F1 = 0.694`
- 結論：
- binary classification 應該 detector-first
- Gemma 目前主要負責結構化 explanation，不適合單獨決定最終真假

### 講者備註

這一頁要講的是目前最務實的系統設計。不是把 Gemma 拿掉，而是把角色分工講清楚。CLIP LoRA 先負責把真假判斷做穩，Gemma 再負責 8 個 criterion 的 explanation 與 JSON 輸出。這樣的架構比讓 Gemma 單獨扛分類更合理，而且實驗上也確實比較好。

## Slide 16. GPU 驗證、限制與不能過度宣稱的地方

### 投影片文案

- detector 的 GPU 路徑已重新驗證正常
- 問題不是 `AIGI-Holmes` 本身，而是舊的 system `python3`
- `torch 1.13.1+cu117` 不支援 Blackwell `sm_120`
- 在錯誤 runtime 下，連 `matmul / conv / sigmoid` 都會出現全 0
- 正確 runtime：`.venv-google-ai-edge` + `torch 2.9.0+cu128`
- 在正確 GPU 環境下，detector 的 CPU / GPU 分數對齊
- 目前仍不能過度宣稱：
- 這是 lightweight pseudo-validation，不是 full validation
- threshold `0.32 / 0.34` 是在同一個切片上調的，可能 overfit
- Gemma 的 criterion-level reasoning 仍需要更大切片驗證

### 講者備註

這一頁有兩個作用。第一，先把 GPU 問題講清楚，避免被問到為什麼一開始 detector 會出現全 0。現在已經定位到是 PyTorch 與 Blackwell 的 runtime 相容性，不是模型本身壞掉。第二，要保留研究誠實度：目前 detector-first 方向是對的，但結果仍來自 128 筆 lightweight slice，不能說已經完全定型。

## Slide 17. 收尾：本專題目前真正的技術貢獻

### 投影片文案

- 第一，把 Holmes explanation 轉成固定 8 準則的結構化 supervision
- 第二，建立 multi-task student pipeline，同時學 evidence、taxonomy、consistency、final JSON
- 第三，完成 Gemma 4 E2B 的微調、checkpoint 選擇、兩階段推論與離線評估流程
- 第四，透過 detector comparison 與 fusion experiment，找出目前更合理的系統分工
- 如果要一句話總結：
- 我們做的是資料轉換規格、多任務訓練設計、兩階段推論框架，以及 detector-first 的分類改進方案
- 部署鏈已驗證，但不在本次報告重點

### 講者備註

如果老師最後問「你們真正做了什麼」，我建議就用這一頁回答。不要只說我們微調了 Gemma，也不要只說我們加了一個 detector。更準確的說法是，我們把原本不容易直接訓練的 Holmes supervision，轉成一條可訓練、可評估的完整方法鏈，並且用實驗把目前最合理的分類架構找出來。

## 備用問答

### Q1. 為什麼不直接做二分類？

- 因為這個任務不是只要真假答案，還要能指出是哪一類 artifact 造成判斷
- 固定 8 個 criterion 的結構化輸出，比單純二分類更可解釋，也更方便後續分析錯誤

### Q2. 為什麼要多任務，而不是只學 final JSON？

- 因為只學 final JSON 容易讓模型直接背格式，卻沒有學穩 evidence 結構
- multi-task 讓模型同時學 evidence、taxonomy、consistency，能提高中間表示的可控性

### Q3. 為什麼後來又加 detector？

- 因為在相同切片上，Gemma 單獨做 binary classification 的能力不夠穩
- CLIP LoRA detector 在 overall accuracy 與 macro F1 都明顯更好
- 所以目前最合理的角色分工是 detector 先做真假判斷，Gemma 再負責 explanation

### Q4. 現在結果能不能直接當最終結論？

- 還不能，因為目前是 `128` 筆 lightweight pseudo-validation
- 但它已經足夠用來做架構選型，也足夠說明 detector-first 比 Gemma-only 更合理

## 引用來源

- `teacher/stage1_g31b_v5_full_balanced/stats.json`
- `teacher/derived_deterministic_v1/manifest.json`
- `student/README.md`
- `student/src/train.py`
- `student/src/inference.py`
- `student/src/merge_student.py`
- `student/src/export_litert_model.py`
- `student/src/run_lightweight_eval.py`
- `student/src/run_clip_detector_eval.py`
- `student/src/analyze_detector_fusion.py`
- `student/src/task_utils.py`
- `student/reports/phase1/baseline7014_holmes_val_balanced24_eval.json`
- `student/reports/phase1/phase1_round1_final_holmes_val_balanced24_eval.json`
- `student/reports/gemma4_e2b_round1_checkpoint4000_eval4.json`
- `student/outputs/gemma4_lightweight_eval/eval_reports/ckpt4000_single_stage.json`
- `student/outputs/gemma4_lightweight_eval/eval_reports/ckpt4000_two_stage.json`
- `student/outputs/gemma4_lightweight_eval/eval_reports/ckpt6015_two_stage.json`
- `student/outputs/gemma4_lightweight_eval/detector_clip_lora_gpu.json`
- `student/outputs/gemma4_lightweight_eval/detector_fusion_analysis.json`
- `student/outputs/gemma4_lightweight_eval/experiment_analysis.md`
