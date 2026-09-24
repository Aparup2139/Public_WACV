from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD
from .evaluation import predict_probability_map

# ponytail: bounds the prefer_changed scan to this many shuffled candidates
# instead of the whole dataset (which could be huge just to render `n`
# examples). If a dataset ever has changed samples too sparse to turn up
# within this many draws, the upgrade path is to scan the full dataset index
_PREFER_CHANGED_SCAN_LIMIT = 100


def denorm(image: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    mean_tensor = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=image.device).view(-1, 1, 1)
    restored = (image * std_tensor + mean_tensor).clamp(0, 1)
    return restored.permute(1, 2, 0).cpu().numpy()


def qg_show(
    model,
    dataset,
    threshold: float,
    device,
    n: int = 3,
    title: str = "",
    prefer_changed: bool = True,
    tile: int = 224,
    stride: int = 224,
    tta: bool = False,
    d4_tta: bool = False,
    amp: bool = True,
    amp_dtype: str = "float16",
):
    indices = list(range(len(dataset)))
    if prefer_changed:
        random.shuffle(indices)
        changed = []
        for i in indices[:_PREFER_CHANGED_SCAN_LIMIT]:
            if dataset[i][2].sum() > 0:
                changed.append(i)
            if len(changed) >= n:
                break
        indices = changed if len(changed) >= n else indices
    chosen = random.sample(indices, k=min(n, len(indices)))

    figure, axes = plt.subplots(len(chosen), 4, figsize=(12, 3 * len(chosen)))
    axes = np.atleast_2d(axes)
    for row, index in enumerate(chosen):
        image_a, image_b, target, name = dataset[index]
        probability = predict_probability_map(
            model, image_a, image_b, device, tile, stride, 8, tta, amp, amp_dtype, False, d4_tta
        )
        prediction = (probability > threshold).astype(np.uint8)
        axes[row, 0].imshow(denorm(image_a))
        axes[row, 1].imshow(denorm(image_b))
        axes[row, 2].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        for col, header in enumerate(("pre", "post", "ground truth", "prediction")):
            axes[row, col].set_title(f"{name}: {header}" if col == 0 else header, fontsize=8)
            axes[row, col].axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    return figure
