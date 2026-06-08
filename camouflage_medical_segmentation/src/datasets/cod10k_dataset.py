"""
COD10K (Camouflaged Object Detection) Dataset class for medical segmentation pre-training.

This module provides a flexible PyTorch Dataset class that:
1. Handles multiple COD10K folder structure variants
2. Automatically pairs images with their ground truth masks
3. Supports multiple image formats (.jpg, .png, .bmp, etc.)
4. Applies transformations (resize, normalize, augment)
5. Returns data in a structured dictionary format

Stage 1 Context:
- We use COD10K to pre-train camouflage expert models (CU1 and CUA1)
- Lesions in medical images behave like camouflaged objects (low contrast, subtle)
- By pre-training on camouflage data, models learn to detect subtle, hard-to-see objects
- Later in Stage 2, these experts guide medical segmentation
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from torchvision import transforms


class COD10KSegmentationDataset(Dataset):
    """
    PyTorch Dataset for COD10K camouflage segmentation.
    
    Flexible dataset class that automatically detects and pairs images with masks.
    
    Args:
        root_dir (str or Path): Path to COD10K dataset root
        split (str): "train", "val", or "test" - which split to load
        image_size (int): Target image size (H=W). Default: 256
        image_dir_name (str, optional): Name of image subdirectory. 
                                       If None, auto-detect ("Image", "images", "img", etc.)
        mask_dir_name (str, optional): Name of mask subdirectory.
                                      If None, auto-detect ("GT_Object", "masks", "mask", "gt", etc.)
        normalize (bool): Whether to normalize image to [0, 1]. Default: False
                         (We recommend NOT normalizing for medical data, but keeping raw values)
        augment (bool): Whether to apply geometric augmentations. Default: False
                       (Will only augment training split)
        extensions_images (tuple): Supported image extensions. 
                                 Default: ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        extensions_masks (tuple): Supported mask extensions.
                                Default: ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        return_paths (bool): Whether to return full paths in the output dict. Default: True
    
    Returns (from __getitem__):
        dict: {
            "image": torch.Tensor of shape [3, H, W] with values in [0, 1] if normalized
            "mask": torch.Tensor of shape [1, H, W] with values in {0.0, 1.0}
            "image_path": str, full path to image file
            "mask_path": str, full path to mask file
            "filename": str, image filename stem (without extension)
        }
    
    Example:
        >>> dataset = COD10KSegmentationDataset(
        ...     root_dir="data/COD10K",
        ...     split="train",
        ...     image_size=256,
        ...     normalize=False,
        ...     augment=True
        ... )
        >>> sample = dataset[0]
        >>> print(sample["image"].shape)  # torch.Size([3, 256, 256])
        >>> print(sample["mask"].shape)   # torch.Size([1, 256, 256])
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        image_size: int = 256,
        image_dir_name: Optional[str] = None,
        mask_dir_name: Optional[str] = None,
        normalize: bool = False,
        augment: bool = False,
        use_non_empty_masks_only: bool = False,
        metadata_csv_path: Optional[Union[str, Path]] = None,
        extensions_images: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'),
        extensions_masks: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'),
        return_paths: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment and (split.lower() == "train")  # Only augment training data
        self.extensions_images = tuple(ext.lower() for ext in extensions_images)
        self.extensions_masks = tuple(ext.lower() for ext in extensions_masks)
        self.return_paths = return_paths
        self.use_non_empty_masks_only = use_non_empty_masks_only
        self.metadata_csv_path = Path(metadata_csv_path) if metadata_csv_path is not None else None
        
        # Validate split
        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
        
        # Find split folder (e.g., "Train", "train", "training", etc.)
        self.split_dir = self._find_split_dir(split)
        
        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}\n"
                f"Available subdirs: {list(self.root_dir.iterdir())}"
            )
        
        # Find image and mask directories
        self.image_dir = self._find_subdir(self.split_dir, image_dir_name, 
                                          ["image", "images", "img", "image_data"])
        self.mask_dir = self._find_subdir(self.split_dir, mask_dir_name,
                                         ["gt_object", "mask", "masks", "gt", "annotation", "label"])
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        
        # Build list of image-mask pairs
        self.samples = self._build_sample_list()
        
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid image-mask pairs found in {self.split_dir}\n"
                f"Image dir: {self.image_dir}\n"
                f"Mask dir: {self.mask_dir}\n"
                f"Supported image extensions: {self.extensions_images}\n"
                f"Supported mask extensions: {self.extensions_masks}"
            )
        
        print(f"[COD10KDataset] Loaded {len(self.samples)} samples from {split} split")
    
    def _find_split_dir(self, split: str) -> Path:
        """Find the split directory, handling various naming conventions."""
        split_lower = split.lower()
        
        # Try exact match first
        for item in self.root_dir.iterdir():
            if not item.is_dir():
                continue
            if item.name.lower() == split_lower:
                return item
        
        # Try common alternatives
        split_variants = {
            "train": ["train", "training", "train_data", "trainset"],
            "val": ["val", "valid", "validation", "val_data"],
            "test": ["test", "testing", "test_data", "testset"],
        }
        
        for variant in split_variants.get(split_lower, []):
            for item in self.root_dir.iterdir():
                if not item.is_dir():
                    continue
                if item.name.lower() == variant:
                    return item
        
        # If not found, raise informative error
        available = [item.name for item in self.root_dir.iterdir() if item.is_dir()]
        raise FileNotFoundError(
            f"Could not find '{split}' split directory.\n"
            f"Available directories: {available}"
        )
    
    def _find_subdir(self, parent_dir: Path, name: Optional[str], 
                     fallback_names: List[str]) -> Path:
        """Find a subdirectory by name, with fallback options."""
        if name is not None:
            # Try exact name provided
            candidate = parent_dir / name
            if candidate.exists() and candidate.is_dir():
                return candidate
        
        # Try fallback names
        for fallback in fallback_names:
            for item in parent_dir.iterdir():
                if not item.is_dir():
                    continue
                if item.name.lower() == fallback.lower():
                    return item
        
        # If still not found, return first candidate (will fail later with better error)
        if name is not None:
            return parent_dir / name
        return parent_dir / fallback_names[0]
    
    def _build_sample_list(self) -> List[Dict[str, Path]]:
        """Build list of (image_path, mask_path) pairs by filename matching."""
        samples = []

        # Get all images
        image_files = {}
        for ext in self.extensions_images:
            for img_file in self.image_dir.glob(f"*{ext}"):
                if img_file.is_file():
                    stem = img_file.stem
                    image_files[stem] = img_file

        # Get all masks
        mask_files = {}
        for ext in self.extensions_masks:
            for mask_file in self.mask_dir.glob(f"*{ext}"):
                if mask_file.is_file():
                    stem = mask_file.stem
                    mask_files[stem] = mask_file

        # Pair images with masks
        missing_masks = []
        for stem, img_path in sorted(image_files.items()):
            if stem in mask_files:
                samples.append({
                    "image_path": img_path,
                    "mask_path": mask_files[stem],
                    "filename": stem,
                })
            else:
                missing_masks.append(stem)

        if missing_masks:
            warnings.warn(
                f"Found {len(missing_masks)} images without corresponding masks. "
                f"Examples: {missing_masks[:5]}"
            )

        if self.use_non_empty_masks_only:
            non_empty_samples = []
            if self.metadata_csv_path and self.metadata_csv_path.exists():
                metadata = pd.read_csv(self.metadata_csv_path)
                valid_split_names = [self.split]
                if self.split == "val":
                    valid_split_names.append("validation")

                non_empty_names = set(
                    metadata.loc[
                        metadata["split"].isin(valid_split_names) & ~metadata["mask_empty"],
                        "filename",
                    ].astype(str)
                )

                for sample in samples:
                    if sample["filename"] in non_empty_names:
                        non_empty_samples.append(sample)
            else:
                for sample in samples:
                    mask_array = self._load_mask(sample["mask_path"])
                    if mask_array.sum() > 0:
                        non_empty_samples.append(sample)

            warning_msg = (
                f"Filtering to non-empty masks only for split '{self.split}'. "
                f"{len(non_empty_samples)} / {len(samples)} samples remain."
            )
            warnings.warn(warning_msg)
            samples = non_empty_samples

        return samples
    
    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load and return a single image-mask pair with transformations.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Sample with image tensor, mask tensor, and metadata
        """
        sample = self.samples[idx]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        filename = sample["filename"]
        
        # Load image
        image = self._load_image(image_path)
        
        # Load mask
        mask = self._load_mask(mask_path)
        
        # Ensure image and mask have same spatial dimensions
        if image.shape[:2] != mask.shape[:2]:
            # Resize mask to match image
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Apply augmentations (only for training)
        if self.augment:
            image, mask = self._apply_augmentations(image, mask)
        
        # Resize to target size
        image = cv2.resize(image, (self.image_size, self.image_size), 
                          interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensors
        image_tensor = self._to_tensor(image, is_mask=False)
        mask_tensor = self._to_tensor(mask, is_mask=True)
        
        # Build output dictionary
        output = {
            "image": image_tensor,
            "mask": mask_tensor,
            "filename": filename,
        }
        
        if self.return_paths:
            output["image_path"] = str(image_path)
            output["mask_path"] = str(mask_path)
        
        return output
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image as RGB numpy array (H, W, 3)."""
        img = Image.open(path)
        
        # Ensure RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        return np.array(img, dtype=np.uint8)
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load mask as grayscale numpy array (H, W, 1) with binary values."""
        mask = Image.open(path)
        
        # Convert to grayscale
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        mask_array = np.array(mask, dtype=np.uint8)
        
        # Ensure binary (0 or 255, will normalize later)
        # If mask has other values, threshold at 128
        if mask_array.min() > 0 or mask_array.max() < 255:
            mask_array = (mask_array > 127).astype(np.uint8) * 255
        
        # Add channel dimension if needed
        if mask_array.ndim == 2:
            mask_array = mask_array[:, :, np.newaxis]
        
        return mask_array
    
    def _apply_augmentations(self, image: np.ndarray, 
                            mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply geometric augmentations identically to image and mask."""
        from config import AUGMENTATION_CONFIG
        
        # Horizontal flip
        if AUGMENTATION_CONFIG["horizontal_flip"] and \
           np.random.rand() < AUGMENTATION_CONFIG["horizontal_flip_prob"]:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # Vertical flip
        if AUGMENTATION_CONFIG["vertical_flip"] and \
           np.random.rand() < AUGMENTATION_CONFIG["vertical_flip_prob"]:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        # Rotation
        if AUGMENTATION_CONFIG["rotation"] and \
           np.random.rand() < AUGMENTATION_CONFIG["rotation_prob"]:
            angle = np.random.uniform(
                -AUGMENTATION_CONFIG["rotation_degrees"],
                AUGMENTATION_CONFIG["rotation_degrees"]
            )
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT,
                                flags=cv2.INTER_NEAREST)
        
        return image, mask
    
    def _to_tensor(self, array: np.ndarray, is_mask: bool = False) -> torch.Tensor:
        """
        Convert numpy array to torch tensor.
        
        For images: [0, 255] uint8 -> [0, 1] float32, channel-first [C, H, W]
        For masks: [0, 255] uint8 -> {0, 1} float32, channel-first [1, H, W]
        """
        # Normalize to [0, 1]
        tensor = torch.from_numpy(array).float() / 255.0
        
        # Ensure 3D (C, H, W)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
        
        # Convert from (H, W, C) to (C, H, W) if needed
        if tensor.shape[0] != 1 and tensor.shape[0] != 3:
            # Likely (H, W, C) format
            tensor = tensor.permute(2, 0, 1)
        
        # For mask, ensure binary values {0, 1}
        if is_mask:
            tensor = (tensor > 0.5).float()
        
        return tensor
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get detailed information about a sample (for inspection)."""
        sample = self.samples[idx]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        
        # Load without processing
        img = Image.open(image_path)
        msk = Image.open(mask_path)
        
        return {
            "filename": sample["filename"],
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_size": img.size,  # (W, H)
            "image_mode": img.mode,
            "image_format": img.format,
            "mask_size": msk.size,  # (W, H)
            "mask_mode": msk.mode,
            "mask_format": msk.format,
        }


def create_dataloaders(
    root_dir: Union[str, Path],
    batch_size: int = 16,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    image_size: int = 256,
    normalize: bool = False,
    augment: bool = True,
    use_non_empty_masks_only: bool = False,
    metadata_csv_path: Optional[Union[str, Path]] = None,
    image_dir_name: Optional[str] = None,
    mask_dir_name: Optional[str] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, 
           torch.utils.data.DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test splits.
    
    Splits the training data into train and validation sets based on val_ratio.
    
    Args:
        root_dir: Path to COD10K dataset root
        batch_size: Batch size for dataloaders. Default: 16
        val_ratio: Fraction of training data to use for validation. Default: 0.1
        num_workers: Number of worker processes for data loading. Default: 0
                    (Set to 4 or higher for faster loading on multi-core systems)
        seed: Random seed for reproducible train/val split. Default: 42
        image_size: Target image size (H=W). Default: 256
        normalize: Whether to normalize images. Default: False
        augment: Whether to apply augmentations. Default: True (training only)
        image_dir_name: Name of image directory. If None, auto-detect.
        mask_dir_name: Name of mask directory. If None, auto-detect.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     root_dir="data/COD10K",
        ...     batch_size=16,
        ...     val_ratio=0.1,
        ...     num_workers=4,
        ...     image_size=256
        ... )
        >>> batch = next(iter(train_loader))
        >>> print(batch["image"].shape)  # torch.Size([16, 3, 256, 256])
    """
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create train dataset
    train_dataset = COD10KSegmentationDataset(
        root_dir=root_dir,
        split="train",
        image_size=image_size,
        image_dir_name=image_dir_name,
        mask_dir_name=mask_dir_name,
        normalize=normalize,
        augment=augment,
        use_non_empty_masks_only=use_non_empty_masks_only,
        metadata_csv_path=metadata_csv_path,
    )
    
    # Split train into train/val
    n_train = int(len(train_dataset) * (1 - val_ratio))
    n_val = len(train_dataset) - n_train
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"[DataLoader] Train split: {n_train} samples")
    print(f"[DataLoader] Val split: {n_val} samples")
    
    # Create test dataset
    test_dataset = COD10KSegmentationDataset(
        root_dir=root_dir,
        split="test",
        image_size=image_size,
        image_dir_name=image_dir_name,
        mask_dir_name=mask_dir_name,
        normalize=normalize,
        augment=False,  # Never augment test data
        use_non_empty_masks_only=use_non_empty_masks_only,
        metadata_csv_path=metadata_csv_path,
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, val_loader, test_loader
