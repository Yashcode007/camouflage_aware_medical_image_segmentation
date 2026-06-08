"""
Configuration settings for Stage 1: Camouflage Dataset Pre-training.
Defines paths, hyperparameters, and dataset configurations.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
# Get the project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
COD10K_ROOT = DATA_DIR / "COD10K" / "archive" / "COD10K-v3"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALIZATION_DIR = OUTPUT_DIR / "dataset_visualizations"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

# Create necessary directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, VISUALIZATION_DIR, CHECKPOINT_DIR, PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Expected COD10K dataset structure (adjust these if your folder names differ):
# Option A (recommended):
#   COD10K/
#     ├── Train/
#     │   ├── Image/
#     │   └── GT_Object/
#     └── Test/
#         ├── Image/
#         └── GT_Object/
#
# Option B:
#   COD10K/
#     ├── train/
#     │   ├── images/
#     │   └── masks/
#     └── test/
#         ├── images/
#         └── masks/

# Dataset split configuration
# You will specify these in create_dataloaders() function
COD10K_CONFIG = {
    "train": {
        "split_name": "Train",  # or "train" depending on your folder structure
        "image_dir": "Image",   # or "images"
        "mask_dir": "GT_Object",  # or "masks"
    },
    "test": {
        "split_name": "Test",   # or "test"
        "image_dir": "Image",   # or "images"
        "mask_dir": "GT_Object",  # or "masks"
    },
}

# ============================================================================
# DATALOADER CONFIGURATION
# ============================================================================
DEFAULT_IMAGE_SIZE = 256  # Target image size for U-Net (must be divisible by 32)
DEFAULT_BATCH_SIZE = 16
DEFAULT_VAL_RATIO = 0.1  # 10% of training data for validation
DEFAULT_NUM_WORKERS = 0  # Set to number of CPU cores for faster loading (0 for debugging)
DEFAULT_SEED = 42  # Fixed seed for reproducibility
USE_NON_EMPTY_MASKS_ONLY = True
VALIDATE_NON_EMPTY_MASKS_ONLY = True
TEST_NON_EMPTY_MASKS_ONLY = False

# ============================================================================
# IMAGE PROCESSING
# ============================================================================
# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
SUPPORTED_MASK_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# Image normalization (optional, can be applied or skipped)
# These are standard ImageNet normalization values
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ============================================================================
# AUGMENTATION CONFIGURATION
# ============================================================================
AUGMENTATION_CONFIG = {
    "horizontal_flip": True,
    "horizontal_flip_prob": 0.5,
    "vertical_flip": True,
    "vertical_flip_prob": 0.5,
    "rotation": True,
    "rotation_degrees": 15,
    "rotation_prob": 0.3,
}

# ============================================================================
# LOGGING AND DEBUGGING
# ============================================================================
VERBOSE = True  # Print dataset info during initialization
SAVE_VISUALIZATIONS = True  # Save random samples to disk
NUM_VISUALIZATION_SAMPLES = 5  # Number of samples to visualize

# Metadata output file
METADATA_CSV_PATH = OUTPUT_DIR / "cod10k_metadata.csv"

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_AVAILABLE = torch.cuda.is_available()

if VERBOSE:
    print(f"[CONFIG] Device: {DEVICE}")
    if CUDA_AVAILABLE:
        print(f"[CONFIG] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[CONFIG] CUDA Version: {torch.version.cuda}")
