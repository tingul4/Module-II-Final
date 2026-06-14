import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from PIL import Image

from detectors.holmes_clip_lora.clip_model import CLIPBinaryModel
from detectors.holmes_clip_lora.metrics import compute_binary_metrics


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "student" / "outputs" / "detectors" / "holmes_clip_lora_vitl14_336"
DEFAULT_CHECKPOINT = DEFAULT_ARTIFACT_ROOT / "checkpoints" / "model_epoch_0.94_0.99.pth"
DEFAULT_THRESHOLD = 0.34
DEFAULT_MODEL_NAME = "ViT-L/14@336px"


@dataclass
class DetectorBundle:
    model: CLIPBinaryModel
    device: str
    threshold: float


def resolve_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def default_checkpoint_path() -> str:
    return str(DEFAULT_CHECKPOINT)


def load_detector(
    checkpoint_path: str | Path,
    clip_weights_path: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    device: str | None = None,
) -> DetectorBundle:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"detector checkpoint not found: {checkpoint_path}")
    if not clip_weights_path or not Path(clip_weights_path).exists():
        raise FileNotFoundError(
            "detector base CLIP weights not found. Pass --detector_clip_weights <ViT-L-14-336px.pt>."
        )

    resolved_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = CLIPBinaryModel(DEFAULT_MODEL_NAME, clip_weights=clip_weights_path)
    if "fc" not in checkpoint or "lora" not in checkpoint:
        raise ValueError("detector checkpoint must contain both 'fc' and 'lora' state dicts")
    model.fc.load_state_dict(checkpoint["fc"])
    model.load_state_dict(checkpoint["lora"], strict=False)
    model = model.to(resolved_device)
    model.eval()
    return DetectorBundle(model=model, device=resolved_device, threshold=float(threshold))


def score_pil_images(bundle: DetectorBundle, images: Sequence[Image.Image]) -> List[float]:
    batch = [bundle.model.preprocess(image.convert("RGB")) for image in images]
    batch_tensor = torch.stack(batch).to(bundle.device)
    with torch.no_grad():
        return bundle.model(batch_tensor).sigmoid().flatten().detach().cpu().tolist()


def score_image_paths(bundle: DetectorBundle, image_paths: Iterable[str]) -> List[float]:
    images = []
    for image_path in image_paths:
        with Image.open(image_path).convert("RGB") as image:
            images.append(image.copy())
    return score_pil_images(bundle, images)


def score_single_image(bundle: DetectorBundle, image_path: str) -> float:
    return score_image_paths(bundle, [image_path])[0]


def label_from_score(score: float, threshold: float) -> str:
    return "AI-Generated" if float(score) >= float(threshold) else "Real"


def prediction_payload(score: float, threshold: float) -> dict:
    return {
        "detector_score": float(score),
        "detector_threshold": float(threshold),
        "detector_label": label_from_score(score, threshold),
    }


def evaluate_scores(row_ids: List[int], gold_labels: List[int], scores: List[float], threshold: float) -> dict:
    metrics = compute_binary_metrics(gold_labels, scores, threshold)
    predictions = []
    for row_id, gold_label, score in zip(row_ids, gold_labels, scores):
        predictions.append(
            {
                "row_id": int(row_id),
                "gold_label": int(gold_label),
                **prediction_payload(score, threshold),
            }
        )
    return {
        **metrics,
        "threshold": float(threshold),
        "predictions": predictions,
    }
