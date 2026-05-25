import json
import logging
import os
import random
import subprocess
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed: int = 42) -> None:
    """強制設定所有的隨機數種子以確保實驗的可重現性 (Reproducibility)"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多個GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Seed set strictly to {seed}.")

class StrictExperimentLogger:
    def __init__(self, log_dir_base: str = "logs", exp_name: str = "exp"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(log_dir_base, f"{exp_name}_{self.timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 設置 Python 內建 logging
        log_file = os.path.join(self.exp_dir, "training.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging initialized at {self.exp_dir}")
        
        # 初始化 TensorBoard
        self.writer = SummaryWriter(log_dir=self.exp_dir)
        
        # 紀錄實驗參數
        self.hparams = {}

    def log_hparams(self, hparams: dict):
        self.hparams = hparams
        logging.info(f"Hyperparameters: {json.dumps(hparams, indent=2)}")
        # 可以用 Tensorboard 的 hparams 等等紀錄，但這邊為了寫入 markdown，先存在 dict
        with open(os.path.join(self.exp_dir, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=4)

    def log_metrics(self, metrics: dict, step: int, tag: str = "train"):
        for k, v in metrics.items():
            self.writer.add_scalar(f"{tag}/{k}", v, step)
            
    def _get_git_commit(self):
        try:
            return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
        except Exception:
            return "unknown"

    def finalize_and_append_to_context(self, final_scores: dict, context_md_path: str = ".context/04_EXPERIMENTS.md"):
        """訓練結束時呼叫，自動將結果附加到EXPERIMENTS.md"""
        self.writer.close()
        
        git_hash = self._get_git_commit()
        seed = self.hparams.get("seed", "Unknown")
        model = self.hparams.get("model", "Unknown")
        
        summary = f"\n**Exp {self.timestamp} ({model}):**\n"
        summary += f"- **Date**: {datetime.now().strftime('%Y-%m-%d')}\n"
        summary += f"- **Git Hash**: {git_hash}\n"
        summary += f"- **Seed**: {seed}\n"
        summary += f"- **Hyperparameters**: {json.dumps(self.hparams)}\n"
        summary += f"- **Final Scores**: {json.dumps(final_scores)}\n"
        summary += f"- **Log Dir**: `{self.exp_dir}`\n"
        summary += "- **Result & Observation**: (Please describe the outcome here)\n"
        summary += "- **Conclusion**: (Is this successful? What's the next step?)\n\n---\n"
        
        # 將 summary 寫入附屬的 json 檔案備查
        summary_dict = {
            "timestamp": self.timestamp,
            "git_hash": git_hash,
            "seed": seed,
            "hparams": self.hparams,
            "final_scores": final_scores,
            "log_dir": self.exp_dir
        }
        with open(os.path.join(self.exp_dir, "experiments_summary.json"), "w") as f:
            json.dump(summary_dict, f, indent=4)
            
        # 嘗試寫入 .context/04_EXPERIMENTS.md
        try:
            with open(context_md_path, "a", encoding="utf-8") as f:
                f.write(summary)
            logging.info(f"Successfully appended experiment summary to {context_md_path}")
        except Exception as e:
            logging.error(f"Failed to append to {context_md_path}: {e}")
