"""Loss modules for segmentation tasks."""

from .segmentation_losses import DiceLoss, BCEDiceLoss

__all__ = ["DiceLoss", "BCEDiceLoss"]
