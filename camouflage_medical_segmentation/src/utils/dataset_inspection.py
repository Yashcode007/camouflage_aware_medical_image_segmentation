"""
Dataset Inspection and Visualization Utilities.

This module provides comprehensive tools to:
1. Scan and analyze the COD10K dataset structure
2. Build metadata tables with image and mask properties
3. Detect issues (missing masks, corrupted files, shape mismatches)
4. Visualize samples (raw and transformed)
5. Export statistics and reports
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch
from PIL import Image
from tqdm import tqdm
import cv2

# Suppress PIL warnings
warnings.filterwarnings("ignore", category=UserWarning)


def inspect_dataset(
    root_dir: Union[str, Path],
    output_csv: Union[str, Path],
    image_dir_name: Optional[str] = None,
    mask_dir_name: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Comprehensively inspect a COD10K-like dataset.
    
    Scans all images and masks, pairs them by filename, and builds a detailed
    metadata table with information about:
    - File locations and extensions
    - Image and mask dimensions
    - Data types and modes
    - Value ranges and statistics
    - Missing/corrupted files
    - Shape mismatches
    - Empty masks
    
    Args:
        root_dir: Path to dataset root (containing Train/Test subdirs)
        output_csv: Path where to save metadata CSV
        image_dir_name: Optional name of image subdir. If None, auto-detect.
        mask_dir_name: Optional name of mask subdir. If None, auto-detect.
        max_samples: If set, only process first N samples per split.
    
    Returns:
        Tuple of:
        - pandas.DataFrame: Metadata table
        - dict: Summary statistics
    
    Example:
        >>> metadata_df, summary = inspect_dataset(
        ...     root_dir="data/COD10K",
        ...     output_csv="outputs/cod10k_metadata.csv"
        ... )
        >>> print(f"Total samples: {len(metadata_df)}")
        >>> print(f"Train samples: {summary['train_count']}")
    """
    
    root_dir = Path(root_dir)
    output_csv = Path(output_csv)
    
    metadata_rows = []
    summary = {
        "total_samples": 0,
        "train_count": 0,
        "test_count": 0,
        "val_count": 0,
        "missing_masks": 0,
        "corrupted_images": 0,
        "corrupted_masks": 0,
        "shape_mismatches": 0,
        "empty_masks": 0,
        "image_size_min": None,
        "image_size_max": None,
        "mask_value_ranges": {},
    }
    
    # Process each split
    for split_dir in sorted(root_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        
        split_name = split_dir.name.lower()
        if split_name not in ["train", "test", "val", "validation"]:
            continue
        
        # Map split names
        split_key = "train" if "train" in split_name else \
                   ("test" if "test" in split_name else "val")
        
        print(f"\n[Inspection] Processing {split_name} split...")
        
        # Find image and mask directories
        image_dir = _find_subdir(split_dir, image_dir_name,
                                 ["image", "images", "img", "image_data"])
        mask_dir = _find_subdir(split_dir, mask_dir_name,
                               ["gt_object", "mask", "masks", "gt", "annotation", "label"])
        
        if not image_dir.exists() or not mask_dir.exists():
            print(f"  Skipping {split_name}: missing image or mask directory")
            continue
        
        # Get all image files
        image_files = {}
        for img_file in image_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                stem = img_file.stem
                image_files[stem] = img_file
        
        # Get all mask files
        mask_files = {}
        for mask_file in mask_dir.iterdir():
            if mask_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                stem = mask_file.stem
                mask_files[stem] = mask_file
        
        # Process each image-mask pair
        sample_count = 0
        for img_stem in sorted(image_files.keys()):
            if max_samples and sample_count >= max_samples:
                break
            
            img_path = image_files[img_stem]
            mask_path = mask_files.get(img_stem)
            
            row = _inspect_sample(img_path, mask_path, split_key)
            metadata_rows.append(row)
            
            # Update summary
            summary["total_samples"] += 1
            summary[f"{split_key}_count"] += 1
            
            if mask_path is None:
                summary["missing_masks"] += 1
            
            if row.get("corrupted_image"):
                summary["corrupted_images"] += 1
            
            if row.get("corrupted_mask"):
                summary["corrupted_masks"] += 1
            
            if row.get("shape_mismatch"):
                summary["shape_mismatches"] += 1
            
            if row.get("mask_empty"):
                summary["empty_masks"] += 1
            
            sample_count += 1
        
        print(f"  Processed {sample_count} samples from {split_name}")
    
    # Create DataFrame
    metadata_df = pd.DataFrame(metadata_rows)
    
    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(output_csv, index=False)
    print(f"\n[Inspection] Saved metadata to {output_csv}")
    
    # Print summary
    _print_summary(metadata_df, summary)
    
    return metadata_df, summary


def _find_subdir(parent_dir: Path, name: Optional[str], 
                fallback_names: List[str]) -> Path:
    """Find a subdirectory by name with fallbacks."""
    if name is not None and (parent_dir / name).exists():
        return parent_dir / name
    
    for fallback in fallback_names:
        for item in parent_dir.iterdir():
            if not item.is_dir():
                continue
            if item.name.lower() == fallback.lower():
                return item
    
    # Return first candidate (will fail with clear error later)
    return parent_dir / (name or fallback_names[0])


def _inspect_sample(img_path: Path, mask_path: Optional[Path], 
                   split: str) -> Dict:
    """Inspect a single image-mask pair."""
    row = {
        "split": split,
        "filename": img_path.stem,
        "image_path": str(img_path),
        "mask_path": str(mask_path) if mask_path else None,
        "image_extension": img_path.suffix,
        "mask_extension": mask_path.suffix if mask_path else None,
        "image_exists": img_path.exists(),
        "mask_exists": mask_path.exists() if mask_path else False,
    }
    
    # Check image
    corrupted_image = False
    try:
        img = Image.open(img_path)
        row["image_width"] = img.width
        row["image_height"] = img.height
        row["image_mode"] = img.mode
        row["image_dtype"] = np.uint8  # PIL always reads as uint8
        img.close()
    except Exception as e:
        corrupted_image = True
        row["image_width"] = None
        row["image_height"] = None
        row["image_mode"] = None
        row["image_dtype"] = None
        row["corrupted_image"] = True
    
    # Check mask
    corrupted_mask = False
    if mask_path and mask_path.exists():
        try:
            mask = Image.open(mask_path)
            row["mask_width"] = mask.width
            row["mask_height"] = mask.height
            row["mask_mode"] = mask.mode
            
            mask_array = np.array(mask)
            row["mask_dtype"] = str(mask_array.dtype)
            row["mask_min"] = int(mask_array.min())
            row["mask_max"] = int(mask_array.max())
            row["mask_unique_values"] = len(np.unique(mask_array))
            row["mask_empty"] = (mask_array.max() == 0)
            
            mask.close()
        except Exception as e:
            corrupted_mask = True
            row["mask_width"] = None
            row["mask_height"] = None
            row["mask_mode"] = None
            row["mask_dtype"] = None
            row["mask_min"] = None
            row["mask_max"] = None
            row["mask_unique_values"] = None
            row["corrupted_mask"] = True
    else:
        row["mask_width"] = None
        row["mask_height"] = None
        row["mask_mode"] = None
        row["mask_dtype"] = None
        row["mask_min"] = None
        row["mask_max"] = None
        row["mask_unique_values"] = None
    
    # Check shape mismatch
    if not corrupted_image and not corrupted_mask and mask_path and mask_path.exists():
        if row["image_width"] != row["mask_width"] or \
           row["image_height"] != row["mask_height"]:
            row["shape_mismatch"] = True
    else:
        row["shape_mismatch"] = False
    
    # Set default values for missing columns
    for col in ["corrupted_image", "corrupted_mask", "mask_empty"]:
        if col not in row:
            row[col] = False
    
    return row


def _print_summary(df: pd.DataFrame, summary: Dict) -> None:
    """Print comprehensive dataset summary."""
    print("\n" + "=" * 80)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Total Samples: {summary['total_samples']}")
    print(f"   ├─ Train: {summary['train_count']}")
    print(f"   ├─ Val:   {summary['val_count']}")
    print(f"   └─ Test:  {summary['test_count']}")
    
    print(f"\n⚠️  Data Quality Issues:")
    print(f"   ├─ Missing masks: {summary['missing_masks']}")
    print(f"   ├─ Corrupted images: {summary['corrupted_images']}")
    print(f"   ├─ Corrupted masks: {summary['corrupted_masks']}")
    print(f"   ├─ Shape mismatches: {summary['shape_mismatches']}")
    print(f"   └─ Empty masks: {summary['empty_masks']}")
    
    print(f"\n📐 Image Dimensions:")
    if "image_width" in df.columns:
        valid_images = df[~df["corrupted_image"]]
        if len(valid_images) > 0:
            width_range = (valid_images["image_width"].min(), 
                          valid_images["image_width"].max())
            height_range = (valid_images["image_height"].min(),
                           valid_images["image_height"].max())
            print(f"   ├─ Width:  {width_range[0]}-{width_range[1]} px")
            print(f"   └─ Height: {height_range[0]}-{height_range[1]} px")
    
    print(f"\n📏 Mask Value Ranges:")
    if "mask_min" in df.columns:
        valid_masks = df[df["mask_exists"] & ~df["corrupted_mask"]]
        if len(valid_masks) > 0:
            min_val = valid_masks["mask_min"].min()
            max_val = valid_masks["mask_max"].max()
            print(f"   ├─ Min value: {min_val}")
            print(f"   └─ Max value: {max_val}")
    
    print(f"\n✅ Dataset is ready for processing!" if summary["missing_masks"] == 0 and \
                                                    summary["corrupted_images"] == 0 and \
                                                    summary["corrupted_masks"] == 0 \
          else "\n⚠️  Please review data quality issues above.")
    print("=" * 80 + "\n")


def visualize_sample(
    image: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    title: str = "Sample",
    show: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Visualize an image-mask pair with overlay.
    
    Args:
        image: Tensor [3, H, W] or numpy [H, W, 3] in [0, 1] or [0, 255]
        mask: Tensor [1, H, W] or numpy [H, W, 1] in {0, 1}
        title: Title for the visualization
        show: Whether to display the plot
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    
    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Ensure proper shapes
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    
    # Normalize to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    if mask.max() > 1.0:
        mask = mask / 255.0
    
    # Create overlay
    overlay = image.copy()
    overlay[mask > 0.5] = overlay[mask > 0.5] * 0.5 + np.array([1, 0, 0]) * 0.5
    
    # Create figure
    fig = plt.figure(figsize=(15, 4))
    gs = GridSpec(1, 3, figure=fig)
    
    # Plot image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title("Raw Image")
    ax1.axis("off")
    
    # Plot mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask, cmap="gray")
    ax2.set_title("Ground Truth Mask")
    ax2.axis("off")
    
    # Plot overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay)
    ax3.set_title("Image + Mask Overlay")
    ax3.axis("off")
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def save_batch_visualizations(
    loader: torch.utils.data.DataLoader,
    output_dir: Union[str, Path],
    num_samples: int = 5,
    show_transformed: bool = True,
) -> None:
    """
    Save random batch visualizations from a DataLoader.
    
    Args:
        loader: PyTorch DataLoader yielding batches
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        show_transformed: Whether to show already-transformed tensors
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a batch
    batch = next(iter(loader))
    
    images = batch["image"]  # [B, 3, H, W]
    masks = batch["mask"]    # [B, 1, H, W]
    
    print(f"Batch shapes:")
    print(f"  Image: {images.shape}")
    print(f"  Mask:  {masks.shape}")
    print(f"  Image dtype: {images.dtype}, value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Mask dtype:  {masks.dtype}, unique values: {torch.unique(masks)}")
    
    # Visualize first num_samples from batch
    for i in range(min(num_samples, len(images))):
        image = images[i]
        mask = masks[i]
        filename = batch.get("filename", [f"sample_{i}"])[i] if isinstance(batch.get("filename"), list) else f"sample_{i}"
        
        save_path = output_dir / f"batch_sample_{i:02d}_{filename}.png"
        visualize_sample(
            image,
            mask,
            title=f"Batch Sample {i}: {filename}",
            show=False,
            save_path=save_path,
        )
        print(f"Saved: {save_path}")
    
    print(f"\n✅ Saved {min(num_samples, len(images))} visualizations to {output_dir}")
