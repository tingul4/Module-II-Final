import json
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HolmesSFTDataset(Dataset):
    def __init__(self, jsonl_path, prompt_dir, processor, max_length=2048):
        self.data = []
        base_dir = os.path.dirname(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                # Resolve image path
                img_path = os.path.join(base_dir, item["image"])
                if not os.path.exists(img_path):
                    # fallback
                    img_path = os.path.join(
                        "/ssd4/LPCVC2026/dataset/holmes", item["image"]
                    )
                item["full_image_path"] = img_path
                self.data.append(item)

        self.processor = processor
        self.max_length = max_length
        self.prompt_dir = prompt_dir

        self.prompts = self._load_prompts()

    def _load_prompts(self):
        prompts = {}
        stage1_path = os.path.join(self.prompt_dir, "stage1.txt")
        stage2_path = os.path.join(self.prompt_dir, "stage2.txt")

        if os.path.exists(stage1_path):
            with open(stage1_path, "r", encoding="utf-8") as f:
                # Remove quotes and empty lines
                prompts["stage1"] = [line.strip().strip('"') for line in f if line.strip()]

        if os.path.exists(stage2_path):
            with open(stage2_path, "r", encoding="utf-8") as f:
                prompts["stage2"] = f.read().strip()

        return prompts

    def __len__(self):
        return len(self.data)

    def _format_step2_json(self, step2_draft):
        """
        Convert the dataset's step2_draft into the competition's strict schema.
        Dataset schema: per_criterion_draft [{criterion, proposed_score, evidence}]
        Competition schema: per_criterion [{criterion, evidence, aigc score}]
        """
        per_criterion = []
        draft_list = step2_draft.get("per_criterion_draft", [])
        
        for d in draft_list:
            per_criterion.append({
                "criterion": d["criterion"],
                "evidence": d["evidence"],
                "aigc score": d.get("proposed_score", 0)
            })
        
        return {
            "per_criterion": per_criterion,
            "overall_likelihood": step2_draft.get("overall_likelihood", "Uncertain")
        }

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            image = Image.open(item["full_image_path"]).convert("RGB")
            # 限制解析度
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        except Exception as e:
            # print(f"Error loading image {item['full_image_path']}: {e}")
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        # Multi-task sampling: 50% Stage 1 (Analysis), 50% Stage 2 (JSON)
        # We can adjust this ratio.
        task_type = random.choice(["stage1", "stage2"])
        
        if task_type == "stage1" and self.prompts.get("stage1"):
            # Stage 1: Detailed analysis based on one of the prompts
            user_prompt = random.choice(self.prompts["stage1"])
            # Use original_response as the gold standard analysis
            target_text = item.get("original_response", "")
        else:
            # Stage 2: Synthesis into JSON
            stage2_template = self.prompts.get("stage2", "")
            excerpts = item.get("original_response", "")
            user_prompt = f"{stage2_template}\n\nHere are the analytical answers:\n{excerpts}"
            
            # Use step2_draft transformed to correct schema
            step2_data = item.get("step2_draft", {})
            formatted_json = self._format_step2_json(step2_data)
            target_text = json.dumps(formatted_json, ensure_ascii=False)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
        ]

        if self.processor is not None:
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=False,
                )
                
                # --- 新增: Instruction Masking (只對 Assistant 回覆計算 Loss) ---
                # 複製 input_ids 作為 labels
                labels = inputs["input_ids"].clone()
                
                # 找到 Assistant 回覆開始的位置
                # Qwen2 的 chat template 格式通常是 <|im_start|>assistant\n
                # 我們可以透過尋找 assistant token 序列來屏蔽前面的部分
                # 這裡提供一個通用的做法：
                # 1. 建立只有 User 內容的 prompt
                prompt_messages = messages[:-1]
                prompt_inputs = self.processor.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True, # 包含 <|im_start|>assistant\n
                )
                prompt_length = len(prompt_inputs)
                
                # 將 User Prompt 區域的 labels 設為 -100
                labels[0, :prompt_length] = -100
                inputs["labels"] = labels

            except Exception as e:
                # print(f"Error in apply_chat_template: {e}")
                prompt = f"<|image|>\nUser: {user_prompt}\nAssistant: {target_text}"
                inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                inputs["labels"] = inputs["input_ids"].clone()

            no_squeeze = {
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
            }
            inputs_dict = {
                k: (v if k in no_squeeze else (v.squeeze(0) if v.dim() > 0 else v))
                for k, v in inputs.items()
            }
            return inputs_dict
        else:
            return {"messages": messages, "image": image}


def get_dataloader(
    jsonl_path, prompt_dir, processor, batch_size=2, shuffle=True, num_workers=4
):
    dataset = HolmesSFTDataset(jsonl_path, prompt_dir, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader
