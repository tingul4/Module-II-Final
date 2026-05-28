import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run lightweight EDA on a Holmes-derived authenticity dataset JSONL.")
    parser.add_argument(
        "--jsonl_path",
        type=Path,
        default=REPO_ROOT / "teacher" / "stage1_g31b_v5_full_balanced" / "holmes_lpcvc_sft.jsonl",
    )
    parser.add_argument("--base_dir", type=Path, default=None)
    parser.add_argument("--output_file", type=Path, default=REPO_ROOT / "eda_results.md")
    return parser.parse_args()


def resolve_image_path(base_dir: Path, item: dict) -> Path:
    image_value = item.get("image") or item.get("full_image_path", "")
    image_path = Path(str(image_value))
    if image_path.is_absolute() and image_path.exists():
        return image_path
    candidate = base_dir / image_path
    if candidate.exists():
        return candidate
    return candidate


def main():
    args = parse_args()
    base_dir = args.base_dir or args.jsonl_path.parent

    resolutions = []
    aspect_ratios = []
    text_lengths = []
    images_not_found = 0

    print(f"Starting EDA for {args.jsonl_path} ...")
    with args.jsonl_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            item = json.loads(line)
            image_path = resolve_image_path(base_dir, item)

            if image_path.exists():
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                        resolutions.append(width * height)
                        aspect_ratios.append(width / height)
                except Exception:
                    pass
            else:
                images_not_found += 1

            target = item.get("step1_target", item.get("step2_target", item.get("original_response", "")))
            text_str = target if isinstance(target, str) else json.dumps(target, ensure_ascii=False)
            text_lengths.append(len(text_str))

            if idx > 0 and idx % 5000 == 0:
                print(f"Processed {idx} items")

    results = []
    results.append("# Dataset Exploratory Data Analysis (EDA)")
    results.append("\n## Source")
    results.append(f"- **Dataset**: `{args.jsonl_path}`")

    results.append("\n## Image Analysis")
    results.append(f"- **Total valid images analyzed**: {len(resolutions)}")
    results.append(f"- **Images not found**: {images_not_found}")

    if resolutions:
        results.append("- **Image Resolution (Pixels)**:")
        results.append(f"  - Mean: {np.mean(resolutions):.0f}")
        results.append(f"  - Median: {np.percentile(resolutions, 50):.0f}")
        results.append(f"  - Max: {np.max(resolutions)}")
        results.append(f"  - Min: {np.min(resolutions)}")
        results.append("- **Aspect Ratio (W/H)**:")
        results.append(f"  - Mean: {np.mean(aspect_ratios):.2f}")
        results.append(f"  - Max: {np.max(aspect_ratios):.2f}")
        results.append(f"  - Min: {np.min(aspect_ratios):.2f}")

    if text_lengths:
        results.append("\n## Text Target Analysis")
        results.append("- **Text target length (characters)**:")
        results.append(f"  - Mean: {np.mean(text_lengths):.1f}")
        results.append(f"  - Max: {np.max(text_lengths)}")
        results.append(f"  - Min: {np.min(text_lengths)}")
        results.append("- **Text length percentiles**:")
        results.append(f"  - 50%: {np.percentile(text_lengths, 50)}")
        results.append(f"  - 90%: {np.percentile(text_lengths, 90)}")
        results.append(f"  - 99%: {np.percentile(text_lengths, 99)}")

    args.output_file.write_text("\n".join(results), encoding="utf-8")
    print(f"\nEDA complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
