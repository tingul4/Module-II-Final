import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


class DetectorConfig:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}

    def get(self, key: str, default: Any = None) -> Any:
        current: Any = self.payload
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        current = self.payload
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def apply_overrides(self, args: argparse.Namespace) -> None:
        overrides = {
            "data.train_dataroot": getattr(args, "train_dataroot", None),
            "data.test_dataroot": getattr(args, "test_dataroot", None),
            "models.pretrained.clip_weights": getattr(args, "clip_weights", None),
            "models.checkpoints_dir": getattr(args, "checkpoints_dir", None),
            "training.lr": getattr(args, "lr", None),
            "training.batch_size": getattr(args, "batch_size", None),
            "training.niter": getattr(args, "epochs", None),
            "training.gpu_ids": getattr(args, "gpu_ids", None),
            "training.name": getattr(args, "run_name", None),
        }
        for key, value in overrides.items():
            if value not in (None, ""):
                self.set(key, value)

    def save(self, output_path: str | Path) -> None:
        Path(output_path).write_text(
            yaml.safe_dump(self.payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

