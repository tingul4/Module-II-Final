from pathlib import Path
from typing import Iterable, List, Tuple

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


def create_filelist_loader(rows: List[Tuple[str, int]], transform, batch_size: int, num_workers: int) -> DataLoader:
    dataset = FileListImageDataset(rows, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

