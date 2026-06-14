import os

import torch.nn as nn

from detectors.holmes_clip_lora.clip import clip


CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768,
    "ViT-L/14@336px": 768,
}


class CLIPBinaryModel(nn.Module):
    def __init__(self, name: str, clip_weights: str | None = None, num_classes: int = 1):
        super().__init__()
        load_target = clip_weights if clip_weights and os.path.exists(clip_weights) else name
        self.model, self.preprocess = clip.load(load_target, device="cpu")

        embed_dim = getattr(self.model.visual, "output_dim", None)
        if embed_dim is None:
            embed_dim = CHANNELS.get(name)
        if embed_dim is None and hasattr(self.model.visual, "input_resolution"):
            embed_dim = 768 if self.model.visual.input_resolution >= 336 else 512
        if embed_dim is None:
            raise ValueError(f"Unable to determine CLIP embedding dimension for {name}")

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_feature: bool = False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        return self.fc(features)

