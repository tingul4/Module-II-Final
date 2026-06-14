import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.task_utils import (
    CRITERIA,
    compact_json_dumps,
    compact_trace_payload,
    json_dumps,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEFAULT_TASK_MIX = {
    "final_json": 0.40,
    "evidence_trace": 0.35,
    "taxonomy_classification": 0.15,
    "consistency_check": 0.10,
}

DEFAULT_PROMPTS = {
    "evidence_trace": (
        "Analyze the image and return ONLY a JSON object with the following schema:\n"
        "{\n"
        '  "overall_likelihood": "Real" | "Uncertain" | "AI-Generated",\n'
        '  "per_criterion": [{"criterion": "...", "score": 0 or 1, "evidence": "...", '
        '"support_type": "explicit_holmes | implied_holmes | image_only | unsupported", '
        '"holmes_span": "...", "artifact_taxonomy": "...", "non_applicable": true/false, '
        '"artifact_score_conflict": true/false}]\n'
        "}\n"
        f"Use these exact criteria in order: {', '.join(CRITERIA)}."
    ),
    "taxonomy_classification": (
        "Analyze the image and return ONLY a JSON object that lists the exact canonical criteria in order.\n"
        'For each criterion output {"criterion": "...", "artifact_taxonomy": "...", "support_type": "..."}.\n'
        "Use artifact_taxonomy=none when no grounded artifact is visible."
    ),
    "consistency_check": (
        "Analyze the image and return ONLY a JSON object assessing whether each canonical criterion would have a score"
        ' that is consistent with its evidence. Use the schema {"overall_consistent": true/false, '
        '"expected_overall_likelihood": "Real" | "AI-Generated", '
        '"per_criterion": [{"criterion": "...", "consistent": true/false, "reason": "..."}]}.'
    ),
}


class DerivedMultiTaskDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        prompt_dir,
        processor,
        max_length=2048,
        task_mix: Optional[Dict[str, float]] = None,
        trace_evidence_words: int = 14,
        trace_holmes_span_words: int = 12,
        seed: int = 42,
        allowed_row_ids: Optional[set[int]] = None,
    ):
        self.data = []
        self.epoch = 0
        self.seed = seed
        self.processor = processor
        self.max_length = max_length
        self.prompt_dir = prompt_dir
        self.task_mix = task_mix or dict(DEFAULT_TASK_MIX)
        self.trace_evidence_words = int(trace_evidence_words)
        self.trace_holmes_span_words = int(trace_holmes_span_words)
        self.task_names = list(self.task_mix.keys())
        self.task_probs = self._normalize_task_mix(self.task_mix)
        self.prompts = self._load_prompts()
        jsonl_path = Path(jsonl_path)
        base_dir = jsonl_path.parent
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                row_id = int(item.get("row_id", len(self.data)))
                if allowed_row_ids is not None and row_id not in allowed_row_ids:
                    continue
                image_path = Path(item["image"])
                if not image_path.is_absolute():
                    image_root = item.get("image_root")
                    if image_root:
                        image_path = Path(image_root) / item["image"]
                    else:
                        image_path = base_dir / item["image"]
                item["row_id"] = row_id
                item["full_image_path"] = os.fspath(image_path)
                self.data.append(item)

    def _normalize_task_mix(self, task_mix: Dict[str, float]) -> List[float]:
        total = sum(max(0.0, float(value)) for value in task_mix.values())
        if total <= 0:
            raise ValueError("task_mix must contain positive weights")
        return [max(0.0, float(task_mix[name])) / total for name in self.task_names]

    def _load_prompts(self):
        prompts = {}
        prompt_files = {
            "final_json": "stage2.txt",
            "evidence_trace": "evidence_trace.txt",
            "taxonomy_classification": "taxonomy.txt",
            "consistency_check": "consistency.txt",
        }
        for task_name, filename in prompt_files.items():
            path = os.path.join(self.prompt_dir, filename)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as handle:
                    prompts[task_name] = handle.read().strip()
            else:
                if task_name == "final_json":
                    fallback = os.path.join(self.prompt_dir, "stage2.txt")
                    with open(fallback, "r", encoding="utf-8") as handle:
                        prompts[task_name] = handle.read().strip()
                else:
                    prompts[task_name] = DEFAULT_PROMPTS[task_name]
        return prompts

    def __len__(self):
        return len(self.data)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _pick_task(self, idx: int) -> str:
        rng = random.Random(self.seed + self.epoch * 1_000_003 + idx)
        return rng.choices(self.task_names, weights=self.task_probs, k=1)[0]

    def _load_image(self, path: str) -> Image.Image:
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            return image
        except Exception:
            return Image.new("RGB", (224, 224), color=(0, 0, 0))

    def _build_prompt_and_target(self, item: Dict[str, object], task_name: str):
        final_json_text = json_dumps(item["final_json_target"])
        compact_trace = compact_trace_payload(
            item["evidence_trace_target"],
            evidence_words=self.trace_evidence_words,
            holmes_span_words=self.trace_holmes_span_words,
        )
        evidence_trace_text = compact_json_dumps(compact_trace)
        taxonomy_text = json_dumps(item["taxonomy_target"])
        consistency_text = json_dumps(item["consistency_target"])

        if task_name == "evidence_trace":
            return self.prompts["evidence_trace"], evidence_trace_text
        if task_name == "taxonomy_classification":
            return self.prompts["taxonomy_classification"], taxonomy_text
        if task_name == "consistency_check":
            return self.prompts["consistency_check"], consistency_text

        user_prompt = (
            f"{self.prompts['final_json']}\n\n"
            "Here is the structured evidence trace for this image:\n"
            f"{evidence_trace_text}\n\n"
            "Use the trace to synthesize the final structured decision JSON."
        )
        return user_prompt, final_json_text

    def _tokenize_messages(self, messages):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=False,
        )
        labels = inputs["input_ids"].clone()
        prompt_inputs = self.processor.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_length = len(prompt_inputs)
        labels[0, :prompt_length] = -100
        inputs["labels"] = labels
        return inputs

    def __getitem__(self, idx):
        item = self.data[idx]
        task_name = self._pick_task(idx)
        image = self._load_image(item["full_image_path"])
        user_prompt, target_text = self._build_prompt_and_target(item, task_name)

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

        inputs = self._tokenize_messages(messages)
        no_squeeze = {
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        }
        inputs_dict = {
            key: (value if key in no_squeeze else value.squeeze(0))
            for key, value in inputs.items()
        }

        inputs_dict["task_name"] = task_name
        inputs_dict["row_id"] = torch.tensor(int(item.get("row_id", idx)), dtype=torch.long)
        return inputs_dict


class LegacyHolmesSFTDataset(Dataset):
    def __init__(self, jsonl_path, processor=None):
        self.data = []
        self.processor = processor
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


HolmesSFTDataset = DerivedMultiTaskDataset


def get_default_task_mix():
    return dict(DEFAULT_TASK_MIX)
