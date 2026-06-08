import torch


def _binarize(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    preds = _binarize(logits, threshold)
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    preds = _binarize(logits, threshold)
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.mean().item()


def pixel_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    preds = _binarize(logits, threshold)
    targets = targets.float()
    correct = (preds == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()
