from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _tensor_to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().float()
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    if image.ndim == 2:
        image = image.unsqueeze(-1)
    image = image.clamp(0.0, 1.0).numpy()
    image = (image * 255.0).astype(np.uint8)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image


def _mask_to_rgb(mask: np.ndarray, color: Optional[tuple] = (255, 0, 0)) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[0]
    palette = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    palette[mask > 0] = color
    return palette


def save_prediction_grid(
    images: torch.Tensor,
    masks: torch.Tensor,
    logits: torch.Tensor,
    output_path: Path,
    num_samples: int = 4,
    threshold: float = 0.5,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = images[:num_samples]
    masks = masks[:num_samples]
    logits = logits[:num_samples]

    n_samples = min(num_samples, images.shape[0])
    n_cols = 4
    n_rows = n_samples
    figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs > threshold).astype(np.uint8)

    for idx in range(n_rows):
        image = _tensor_to_rgb_image(images[idx])
        mask = masks[idx].detach().cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        pred = preds[idx][0]

        overlay = image.copy()
        overlay[pred > 0] = (overlay[pred > 0] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)

        rows = [image, _mask_to_rgb(mask, color=(0, 255, 0)), _mask_to_rgb(pred, color=(255, 0, 0)), overlay]
        titles = ["Image", "Ground Truth", "Prediction", "Overlay"]

        for col in range(n_cols):
            ax = axes[idx, col]
            ax.imshow(rows[col])
            ax.set_title(titles[col])
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
