import json
import os
from PIL import Image
import numpy as np


def main():
    jsonl_path = "/ssd4/LPCVC2026/holmes_lpcvc3_multi_teacher/stage1_g31b_v5_full_balanced/holmes_lpcvc_sft.jsonl"
    base_dir = (
        "/ssd4/LPCVC2026/holmes_lpcvc3_multi_teacher/stage1_g31b_v5_full_balanced"
    )

    resolutions = []
    aspect_ratios = []
    text_lengths = []
    images_not_found = 0

    print("Starting EDA...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            # 支援不同鍵值的寫法
            img_val = item.get("image") or item.get("full_image_path", "")
            img_path = str(img_val)

            # 若為相對路徑，轉換為絕對路徑
            if not os.path.exists(img_path):
                img_path = os.path.join(base_dir, img_path)

            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        resolutions.append(w * h)
                        aspect_ratios.append(w / h)
                except Exception:
                    pass
            elif img_path:
                img_path = os.path.join("/ssd4/LPCVC2026/dataset", img_val)
                if os.path.exists(img_path):
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                            resolutions.append(w * h)
                            aspect_ratios.append(w / h)
                    except Exception:
                        pass
                else:
                    images_not_found += 1

            # Text length (包含 step1_target, step2_target 或是 original_response)
            target = item.get(
                "step1_target",
                item.get("step2_target", item.get("original_response", "")),
            )
            text_str = (
                target
                if isinstance(target, str)
                else json.dumps(target, ensure_ascii=False)
            )
            text_lengths.append(len(text_str))

            if i > 0 and i % 5000 == 0:
                print(f"Processed {i} items")

    # 統整結果並輸出成 Markdown 格式
    res = []
    res.append("# Dataset Exploratory Data Analysis (EDA)")
    res.append("\n## Source")
    res.append(f"- **Dataset**: `{jsonl_path}`")

    res.append("\n## Image Analysis")
    res.append(f"- **Total valid images analyzed**: {len(resolutions)}")
    res.append(f"- **Images not found**: {images_not_found}")

    if resolutions:
        res.append(f"- **Image Resolution (Pixels)**:")
        res.append(f"  - Mean: {np.mean(resolutions):.0f}")
        res.append(f"  - Median: {np.percentile(resolutions, 50):.0f}")
        res.append(f"  - Max: {np.max(resolutions)}")
        res.append(f"  - Min: {np.min(resolutions)}")
        res.append(f"- **Aspect Ratio (W/H)**:")
        res.append(f"  - Mean: {np.mean(aspect_ratios):.2f}")
        res.append(f"  - Max: {np.max(aspect_ratios):.2f}")
        res.append(f"  - Min: {np.min(aspect_ratios):.2f}")

    res.append("\n## Text Target Analysis")
    res.append(f"- **Text target length (characters)**:")
    res.append(f"  - Mean: {np.mean(text_lengths):.1f}")
    res.append(f"  - Max: {np.max(text_lengths)}")
    res.append(f"  - Min: {np.min(text_lengths)}")
    res.append("- **Text length percentiles**:")
    res.append(f"  - 50%: {np.percentile(text_lengths, 50)}")
    res.append(f"  - 90%: {np.percentile(text_lengths, 90)}")
    res.append(f"  - 99%: {np.percentile(text_lengths, 99)}")

    md_content = "\n".join(res)

    output_file = "/ssd4/LPCVC2026/student/eda_results.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nEDA complete! Results saved to {output_file}")


if __name__ == "__main__":
    main()
