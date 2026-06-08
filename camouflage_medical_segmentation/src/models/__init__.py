"""Model registry for Stage 1 camouflage segmentation."""

from .unet import UNet
from .attention_unet import AttentionUNet

__all__ = ["UNet", "AttentionUNet"]
