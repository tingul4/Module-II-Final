import argparse
from pathlib import Path

from model_utils import merge_lora_adapter


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a LoRA student adapter into a full Hugging Face model directory.")
    parser.add_argument("--base_model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "student" / "merged_models" / "gemma4_e2b_latest"),
    )
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    merged_dir, config_path = merge_lora_adapter(
        args.base_model,
        args.adapter_path,
        args.output_dir,
        local_files_only=args.local_files_only,
    )
    print(f"Merged model saved to: {merged_dir}")
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()
