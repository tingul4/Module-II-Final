import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def build_train_transform(preprocess):
    base_transforms = list(getattr(preprocess, "transforms", []))
    if not base_transforms:
        raise ValueError("CLIP preprocess did not expose torchvision transforms")
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(336, scale=(0.9, 1.0)),
            *base_transforms[2:],
        ]
    )


class FileListImageDataset(Dataset):
    def __init__(self, rows: List[Tuple[str, int]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image_path, label = self.rows[index]
        with Image.open(image_path).convert("RGB") as image:
            tensor = self.transform(image.copy())
        return tensor, label


def create_imagefolder_loader(root: str | Path, transform, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    dataset = datasets.ImageFolder(str(root), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def create_filelist_loader(
    rows: List[Tuple[str, int]],
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = FileListImageDataset(rows, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def label_from_overall_likelihood(overall_likelihood: str) -> int:
    return 1 if str(overall_likelihood) == "AI-Generated" else 0


def resolve_derived_image_path(item: dict, base_dir: Path) -> Path:
    image_path = Path(str(item["image"]))
    if image_path.is_absolute():
        return image_path
    image_root = item.get("image_root")
    if image_root:
        return Path(str(image_root)) / image_path
    return base_dir / image_path


def load_derived_filelist(
    jsonl_path: str | Path,
    *,
    allowed_row_ids: Optional[set[int]] = None,
) -> List[Tuple[str, int]]:
    jsonl_path = Path(jsonl_path)
    rows: List[Tuple[str, int]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for fallback_row_id, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            row_id = int(item.get("row_id", fallback_row_id))
            if allowed_row_ids is not None and row_id not in allowed_row_ids:
                continue
            image_path = resolve_derived_image_path(item, jsonl_path.parent)
            label = label_from_overall_likelihood(item["final_json_target"]["overall_likelihood"])
            rows.append((str(image_path), label))
    return rows


def summarize_binary_labels(rows: Iterable[Tuple[str, int]]) -> dict[str, int]:
    counts = Counter(label for _, label in rows)
    return {
        "Real": int(counts.get(0, 0)),
        "AI-Generated": int(counts.get(1, 0)),
    }
