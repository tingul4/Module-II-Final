import argparse
import html
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
AIGI_ROOT = REPO_ROOT.parent / "bk" / "AIGI-Holmes" / "Baselines_AIGI"
CLIP_MODULE_ROOT = AIGI_ROOT / "models" / "clip"
DEFAULT_DATASET = REPO_ROOT / "teacher" / "derived_deterministic_v1" / "derived.jsonl"
DEFAULT_ROW_IDS = REPO_ROOT / "student" / "outputs" / "gemma4_lightweight_eval" / "eval_slice_row_ids.txt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "student" / "outputs" / "gemma4_lightweight_eval"
DEFAULT_CHECKPOINT = REPO_ROOT.parent / "bk" / "AIGI-Holmes" / "checkpoints_clip_lora" / "clip_lora_holmes1" / "model_epoch_0.94_0.99.pth"
DEFAULT_CLIP_WEIGHTS = REPO_ROOT.parent / "bk" / "AIGI-Holmes" / "pretrained" / "clip" / "ViT-L-14-336px.pt"

if str(AIGI_ROOT) not in sys.path:
    sys.path.insert(0, str(AIGI_ROOT))
if str(CLIP_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIP_MODULE_ROOT))

from models.clip_models import CLIPModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run AIGI-Holmes CLIP LoRA detector on a fixed derived-data slice.")
    parser.add_argument("--derived_data_path", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--row_ids_path", type=str, default=str(DEFAULT_ROW_IDS))
    parser.add_argument("--checkpoint_path", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--clip_weights", type=str, default=str(DEFAULT_CLIP_WEIGHTS))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output_prefix", type=str, default="detector_clip_lora_gpu")
    return parser.parse_args()


def load_row_ids(path: Path) -> set[int]:
    return {int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def iter_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve_image_path(row: Dict[str, object]) -> Path:
    image_root = Path(str(row["image_root"]))
    image_rel = Path(str(row["image"]))
    return REPO_ROOT / image_root / image_rel


def load_slice_rows(dataset_path: Path, row_ids_path: Path) -> List[Dict[str, object]]:
    wanted = load_row_ids(row_ids_path)
    rows = []
    for row in iter_rows(dataset_path):
        row_id = int(row["row_id"])
        if row_id not in wanted:
            continue
        label = row["final_json_target"]["overall_likelihood"]
        rows.append(
            {
                "row_id": row_id,
                "image_path": str(resolve_image_path(row)),
                "gold_label": 1 if label == "AI-Generated" else 0,
                "gold_label_text": label,
            }
        )
    rows.sort(key=lambda item: item["row_id"])
    if len(rows) != len(wanted):
        raise ValueError(f"row_id resolution mismatch: expected {len(wanted)}, got {len(rows)}")
    return rows


def require_supported_cuda(device: str):
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    index = torch.device(device).index or 0
    props = torch.cuda.get_device_properties(index)
    capability = f"sm_{props.major}{props.minor}"
    arch_list = set(torch.cuda.get_arch_list())
    if capability not in arch_list:
        raise RuntimeError(
            f"PyTorch {torch.__version__}+cuda{torch.version.cuda} does not advertise {capability}. "
            f"Detected GPU={props.name}. Supported arch list={sorted(arch_list)}"
        )


def sanity_check_device(device: str):
    if not device.startswith("cuda"):
        return
    probe = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
    sigmoid = probe.sigmoid().detach().cpu().tolist()
    if max(sigmoid) == 0.0:
        raise RuntimeError(f"CUDA sanity check failed on {device}: sigmoid probe returned all zeros")
    mat = torch.randn(4, 4, device=device) @ torch.randn(4, 4, device=device)
    if float(mat.abs().sum().item()) == 0.0:
        raise RuntimeError(f"CUDA sanity check failed on {device}: matmul probe returned all zeros")


def build_model(checkpoint_path: Path, clip_weights: Path, device: str):
    opt = type("Opt", (), {"clip_weights": str(clip_weights)})()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = CLIPModel("ViT-L/14@336px", 1, opt)
    model.fc.load_state_dict(checkpoint["fc"])
    model.load_state_dict(checkpoint["lora"], strict=False)
    model = model.to(device)
    model.eval()
    return model


def average_precision(y_true: List[int], y_score: List[float]) -> float:
    positives = sum(y_true)
    if positives == 0:
        return 0.0
    order = sorted(range(len(y_true)), key=lambda idx: y_score[idx], reverse=True)
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0
    for idx in order:
        if y_true[idx]:
            tp += 1
        else:
            fp += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / positives
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def compute_metrics(y_true: List[int], y_score: List[float], threshold: float) -> Dict[str, float]:
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision_fake = tp / (tp + fp) if (tp + fp) else 0.0
    precision_real = tn / (tn + fn) if (tn + fn) else 0.0
    fake_f1 = 2 * precision_fake * fake_recall / (precision_fake + fake_recall) if (precision_fake + fake_recall) else 0.0
    real_f1 = 2 * precision_real * real_recall / (precision_real + real_recall) if (precision_real + real_recall) else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": (real_f1 + fake_f1) / 2.0,
        "average_precision": average_precision(y_true, y_score),
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "pred_fake_count": sum(y_pred),
        "pred_real_count": len(y_pred) - sum(y_pred),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def render_html(report: Dict[str, object]) -> str:
    rows = []
    for key in (
        "sample_count",
        "threshold",
        "accuracy",
        "macro_f1",
        "average_precision",
        "real_recall",
        "fake_recall",
        "pred_real_count",
        "pred_fake_count",
        "wall_time_sec",
        "sec_per_sample",
        "device",
        "torch_version",
        "torch_cuda_version",
    ):
        value = report.get(key)
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(f'{value:.6f}' if isinstance(value, float) else str(value))}</td>"
            "</tr>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>CLIP Detector Eval</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:8px 10px;text-align:left;}th{background:#f4f4f4;}</style>"
        "</head><body>"
        "<h1>CLIP LoRA Detector Evaluation</h1>"
        "<table>"
        + "".join(rows)
        + "</table></body></html>"
    )


def main():
    args = parse_args()
    dataset_path = Path(args.derived_data_path)
    row_ids_path = Path(args.row_ids_path)
    checkpoint_path = Path(args.checkpoint_path)
    clip_weights = Path(args.clip_weights)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    require_supported_cuda(args.device)
    sanity_check_device(args.device)

    rows = load_slice_rows(dataset_path, row_ids_path)
    model = build_model(checkpoint_path, clip_weights, args.device)

    start_time = time.time()
    scores = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            batch = []
            for row in batch_rows:
                image = Image.open(row["image_path"]).convert("RGB")
                batch.append(model.preprocess(image))
            batch_tensor = torch.stack(batch).to(args.device)
            batch_scores = model(batch_tensor).sigmoid().flatten().detach().cpu().tolist()
            scores.extend(batch_scores)
    elapsed = time.time() - start_time

    predictions = []
    for row, score in zip(rows, scores):
        predictions.append(
            {
                **row,
                "detector_score": score,
                f"detector_pred_threshold_{str(args.threshold).replace('.', '_')}": int(score >= args.threshold),
            }
        )

    metrics = compute_metrics([row["gold_label"] for row in rows], scores, args.threshold)
    report = {
        "sample_count": len(rows),
        "threshold": args.threshold,
        "device": args.device,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "checkpoint_path": str(checkpoint_path),
        "clip_weights": str(clip_weights),
        "row_ids_path": str(row_ids_path),
        "derived_data_path": str(dataset_path),
        "wall_time_sec": elapsed,
        "sec_per_sample": elapsed / len(rows) if rows else 0.0,
        **metrics,
    }

    output_jsonl = output_root / f"{args.output_prefix}.jsonl"
    output_json = output_root / f"{args.output_prefix}.json"
    output_html = output_root / f"{args.output_prefix}.html"
    output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_html.write_text(render_html(report), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
