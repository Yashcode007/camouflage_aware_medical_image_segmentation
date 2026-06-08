"""Dataset classes and utilities for COD10K camouflage segmentation."""

from .cod10k_dataset import COD10KSegmentationDataset, create_dataloaders

__all__ = ["COD10KSegmentationDataset", "create_dataloaders"]
