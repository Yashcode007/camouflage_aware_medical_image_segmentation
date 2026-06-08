# Camouflage-Aware Medical Image Segmentation
## Stage 1: Pre-training on COD10K Dataset

This is a research-grade PyTorch implementation for **Stage 1** of the Camouflage-Aware Medical Image Segmentation paper.

---

## 📋 Overview

### What is Stage 1?

**Stage 1** focuses on pre-training two "camouflage expert" models on the **COD10K** (Camouflage Object Detection) dataset:

1. **CU1**: Standard U-Net trained on COD10K
2. **CUA1**: Attention U-Net trained on COD10K

### Why COD10K?

Medical image lesions behave like **camouflaged objects**:
- **Low contrast** with surrounding tissue
- **Subtle and hard to detect**
- **Blended with background**

By pre-training on camouflage data (objects that mimic their surroundings), the models learn to detect hard-to-see patterns. These pre-trained models then guide medical segmentation in Stage 2.

### What This Module Does

This repository implements **only the dataset and dataloader pipeline** for Stage 1. We handle:

✅ **Flexible Dataset Class** (`COD10KSegmentationDataset`):
- Auto-detects folder structure (handles multiple variants: Option A, B, or custom)
- Pairs images with masks by filename stem
- Supports multiple image formats (.jpg, .png, .bmp, .tif, etc.)
- Applies safe geometric augmentations
- Returns standardized tensor format

✅ **DataLoader Creation** (`create_dataloaders`):
- Automatically splits training data into train/val/test
- Configurable batch size, number of workers, image size
- Fixed seed for reproducibility

✅ **Dataset Inspection** (`inspect_dataset`):
- Scans folder structure
- Builds comprehensive metadata table
- Detects missing/corrupted files
- Identifies shape mismatches and empty masks
- Exports statistics to CSV

✅ **Visualization Tools**:
- Raw image + mask overlays
- Transformed batch samples
- Saves visualizations for verification

---

## 📦 Installation

### 1. Clone/Navigate to Project

```bash
cd camouflage_medical_segmentation
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: PyTorch installation may vary by system. See [pytorch.org](https://pytorch.org) for GPU-specific instructions.

---

## 📥 Dataset Setup

### Expected Folder Structure

The COD10K dataset should be structured as **one of these options**:

#### Option A (Recommended - COD10K from Kaggle)
```
data/COD10K/
├── Train/
│   ├── Image/          (all training images)
│   └── GT_Object/      (all training masks)
└── Test/
    ├── Image/         (all test images)
    └── GT_Object/     (all test masks)
```

#### Option B (Alternative)
```
data/COD10K/
├── train/
│   ├── images/        (all training images)
│   └── masks/         (all training masks)
└── test/
    ├── images/       (all test images)
    └── masks/        (all test masks)
```

#### Option C (Custom)
The code auto-detects common folder names:
- Image dirs: `image`, `images`, `img`, `image_data`
- Mask dirs: `gt_object`, `mask`, `masks`, `gt`, `annotation`, `label`

### Download Dataset

**COD10K** is publicly available:
- [Kaggle Dataset](https://www.kaggle.com/datasets/aryehgod/camouflage-object-detection-cod10k)
- [Official GitHub](https://github.com/DengPingFan/SINet-V2)

**Downloaded dataset should be placed at**:
```
camouflage_medical_segmentation/data/COD10K/
```

---

## 🚀 Quick Start

### 1. Inspect the Dataset

Run comprehensive dataset inspection:

```bash
python inspect_dataset.py
```

**Output**:
- `outputs/cod10k_metadata.csv` - Metadata table with image/mask properties
- `outputs/dataset_visualizations/` - Sample visualizations
- Console output with summary statistics

**Flags**:
```bash
# Specify custom dataset path
python inspect_dataset.py --dataset-root /path/to/COD10K

# Adjust visualization count
python inspect_dataset.py --num-viz 10

# Skip DataLoader verification (only inspect files)
python inspect_dataset.py --skip-dataloader
```

### 2. Run Interactive Notebook (Optional)

```bash
jupyter notebook notebooks/01_dataset_inspection.ipynb
```

### 3. Verify DataLoader Output

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src")))

from config import COD10K_ROOT
from datasets import create_dataloaders

# Create DataLoaders
train_loader, val_loader, test_loader = create_dataloaders(
    root_dir="data/COD10K",
    batch_size=16,
    image_size=256,
    num_workers=0,
)

# Get one batch
batch = next(iter(train_loader))

print(f"Image shape: {batch['image'].shape}")      # Expected: [16, 3, 256, 256]
print(f"Mask shape:  {batch['mask'].shape}")       # Expected: [16, 1, 256, 256]
print(f"Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
print(f"Mask values: {batch['mask'].unique()}")    # Expected: tensor([0., 1.])
```

---

## 📊 Expected Outputs

### Dataset Inspection Summary

```
================================================================================
DATASET INSPECTION SUMMARY
================================================================================

📊 Total Samples: 10,040
   ├─ Train: 8,040
   ├─ Val:   806
   └─ Test:  1,194

✅ Data Quality Issues: 0
   ├─ Missing masks: 0
   ├─ Corrupted images: 0
   ├─ Corrupted masks: 0
   ├─ Shape mismatches: 0
   └─ Empty masks: 0

📐 Image Dimensions:
   ├─ Width:  640-1920 px
   └─ Height: 480-1440 px

📏 Mask Value Ranges:
   ├─ Min value: 0
   └─ Max value: 255

✅ Dataset is ready for processing!
================================================================================
```

### DataLoader Batch Format

```
Batch shapes:
  Image: torch.Size([16, 3, 256, 256])
  Mask:  torch.Size([16, 1, 256, 256])
  Image dtype: torch.float32, value range: [0.0000, 1.0000]
  Mask dtype:  torch.float32, unique values: tensor([0., 1.])
```

### Visualizations

Sample visualizations are saved to `outputs/dataset_visualizations/`:
- `batch_sample_00_*.png` - Raw image, ground truth mask, and overlay
- Shows transformation quality and data distribution

---

## 🔧 Module Reference

### `src/config.py`

Central configuration file defining:
- **Paths**: `COD10K_ROOT`, `OUTPUT_DIR`, `CHECKPOINT_DIR`
- **Dataset splits**: Image/mask folder names
- **Hyperparameters**: `DEFAULT_IMAGE_SIZE`, `DEFAULT_BATCH_SIZE`
- **Augmentation settings**: Flip probability, rotation degrees

**Key variables you may need to adjust**:
```python
COD10K_ROOT = Path("data/COD10K")  # Dataset location
DEFAULT_IMAGE_SIZE = 256            # Must be divisible by 32 for U-Net
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0             # Increase for faster loading
DEFAULT_SEED = 42                   # For reproducibility
```

### `src/datasets/cod10k_dataset.py`

Main dataset and dataloader utilities:

#### `COD10KSegmentationDataset`
```python
from datasets import COD10KSegmentationDataset

dataset = COD10KSegmentationDataset(
    root_dir="data/COD10K",
    split="train",
    image_size=256,
    normalize=False,        # Keep raw values
    augment=True,           # Geometric augmentations for training
)

sample = dataset[0]
# Returns:
# {
#     "image": torch.Tensor [3, 256, 256] in [0, 1]
#     "mask": torch.Tensor [1, 256, 256] in {0, 1}
#     "image_path": str
#     "mask_path": str
#     "filename": str
# }
```

#### `create_dataloaders`
```python
from datasets import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    root_dir="data/COD10K",
    batch_size=16,
    val_ratio=0.1,          # 10% of training for validation
    num_workers=4,          # Parallel data loading
    image_size=256,
    normalize=False,
    augment=True,
)

for batch in train_loader:
    images = batch["image"]  # [B, 3, 256, 256]
    masks = batch["mask"]    # [B, 1, 256, 256]
    # ... training code
```

### `src/utils/dataset_inspection.py`

Inspection and visualization tools:

#### `inspect_dataset`
```python
from utils import inspect_dataset

metadata_df, summary = inspect_dataset(
    root_dir="data/COD10K",
    output_csv="outputs/cod10k_metadata.csv",
)

# metadata_df contains: columns like image_path, mask_path, image_width,
# image_height, image_mode, mask_min, mask_max, corrupted_image, etc.

# summary dict contains: total_samples, train_count, missing_masks, etc.
```

#### `visualize_sample`
```python
from utils import visualize_sample

fig = visualize_sample(
    image=batch["image"][0],      # [3, H, W]
    mask=batch["mask"][0],        # [1, H, W]
    title="Sample 0",
    save_path="output.png",
    show=True,
)
```

#### `save_batch_visualizations`
```python
from utils import save_batch_visualizations

save_batch_visualizations(
    loader=train_loader,
    output_dir="outputs/dataset_visualizations",
    num_samples=5,
)
```

---

## 📝 Project Structure

```
camouflage_medical_segmentation/
├── data/
│   └── COD10K/              ← Place downloaded dataset here
│       ├── Train/
│       └── Test/
├── src/
│   ├── __init__.py
│   ├── config.py            ← Central configuration
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── cod10k_dataset.py    ← Main dataset & dataloader
│   └── utils/
│       ├── __init__.py
│       └── dataset_inspection.py ← Inspection & visualization
├── notebooks/
│   └── 01_dataset_inspection.ipynb  ← Interactive inspection
├── checkpoints/             ← Will store CU1.pth, CUA1.pth (Stage 2)
├── outputs/
│   ├── cod10k_metadata.csv
│   └── dataset_visualizations/
├── inspect_dataset.py       ← Main inspection script
├── requirements.txt
└── README.md
```

---

## 🎯 Expected Tensor Shapes & Values

### Batch from DataLoader

```python
batch = next(iter(train_loader))

# Image tensor
batch["image"]
  Shape: torch.Size([B, 3, 256, 256])
  Dtype: torch.float32
  Range: [0.0, 1.0]  (normalized from [0, 255])
  
# Mask tensor
batch["mask"]
  Shape: torch.Size([B, 1, 256, 256])
  Dtype: torch.float32
  Values: {0.0, 1.0} (binary)

# Metadata
batch["image_path"]  → List[str]
batch["mask_path"]   → List[str]
batch["filename"]    → List[str]
```

### Augmentations (Training Only)

Applied with specific probabilities to maintain identical transforms:
- Horizontal flip: 50%
- Vertical flip: 50%
- Rotation: 30% (±15°)

*Augmentations are NOT applied to validation/test splits.*

---

## ⚠️ Common Issues & Troubleshooting

### Issue: "FileNotFoundError: Could not find 'train' split directory"

**Solution**: Adjust folder names in `src/config.py` or use auto-detection:
```python
# The code auto-tries: "train", "Train", "training", "trainset"
# If your folder is named differently, create a symbolic link:
mklink /D data/COD10K/Train data/COD10K/your_train_folder  # Windows
ln -s data/COD10K/your_train_folder data/COD10K/Train     # Linux
```

### Issue: "No valid image-mask pairs found"

**Solution**: Check that image and mask filenames match:
```python
# Images: image_001.jpg, image_002.jpg
# Masks:  image_001.jpg, image_002.jpg  ← Same stem!
# ✓ Will pair correctly

# Images: img_001.jpg
# Masks:  mask_001.jpg  ← Different stems!
# ✗ Will NOT pair (add filename mapping if needed)
```

### Issue: GPU Out of Memory

**Solution**: Reduce batch size or image size:
```python
create_dataloaders(
    batch_size=8,       # Reduce from 16
    image_size=192,     # Reduce from 256
)
```

### Issue: DataLoader very slow

**Solution**: Increase number of workers:
```python
create_dataloaders(
    num_workers=4,  # Increase from 0 (use CPU count)
)
```

---

## 🔄 Workflow: From Inspection to Training

### Current Status: ✅ Stage 1 Setup (COMPLETE)

This repository implements **Stage 1 Dataset Pipeline**:
- [x] Project structure created
- [x] Dataset class with flexible folder detection
- [x] DataLoader creation with train/val/test splits
- [x] Dataset inspection and metadata extraction
- [x] Visualization tools
- [x] Comprehensive documentation

### Stage 1 Status: Training Added

This repository now includes Stage 1 training for camouflage experts:
- [x] U-Net model architecture (CU1)
- [x] Attention U-Net model architecture (CUA1)
- [x] Dice + BCE loss function
- [x] Training loops with validation
- [x] Checkpoint saving: CU1_best.pth, CU1_final.pth, CUA1_best.pth, CUA1_final.pth
- [x] Prediction visualization for Stage 1 evaluation

### Stage 1 Usage

Train CU1:
```bash
python train_stage1.py \
  --model_type unet \
  --data_root data/COD10K/archive/COD10K-v3 \
  --epochs 50 \
  --batch_size 16 \
  --image_size 256 \
  --base_channels 32 \
  --use_non_empty_masks_only
```

Train CUA1:
```bash
python train_stage1.py \
  --model_type attention_unet \
  --data_root data/COD10K/archive/COD10K-v3 \
  --epochs 50 \
  --batch_size 16 \
  --image_size 256 \
  --base_channels 32 \
  --use_non_empty_masks_only
```

Evaluate CU1:
```bash
python evaluate_stage1.py \
  --model_type unet \
  --checkpoint checkpoints/CU1.pth \
  --data_root data/COD10K/archive/COD10K-v3 \
  --image_size 256
```

Evaluate CUA1:
```bash
python evaluate_stage1.py \
  --model_type attention_unet \
  --checkpoint checkpoints/CUA1.pth \
  --data_root data/COD10K/archive/COD10K-v3 \
  --image_size 256
```

### Stage 2 Medical Segmentation (Pending)
- [ ] Load pre-trained CU1 and CUA1 experts
- [ ] Implement medical segmentation guidance mechanism
- [ ] Adapt models to medical datasets (BUSI, Kvasir-SEG, etc.)
- [ ] Fine-tuning pipelines

---

## 🎓 Learning Resources

### Relevant Papers & Concepts

1. **Camouflaged Object Detection**
   - SINet-V2: https://github.com/DengPingFan/SINet-V2
   - Focuses on detecting objects that mimic their background
   
2. **U-Net & Attention U-Net**
   - U-Net: Convolutional Networks for Biomedical Image Segmentation
   - Attention U-Net: Learning Where to Look for the Pancreas

3. **Medical Image Segmentation**
   - Common datasets: BUSI (breast), Kvasir-SEG (gastrointestinal)
   - Loss functions: Dice Loss, Focal Loss, Tversky Loss

### PyTorch References

- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html)
- [Torchvision Transforms](https://pytorch.org/vision/stable/transforms.html)
- [Custom Datasets](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

---

## 📞 Support & Questions

For issues or questions:

1. **Check the metadata CSV** for data quality issues
2. **Inspect visualizations** to understand data format
3. **Review console output** from `inspect_dataset.py` for detailed diagnostics
4. **Verify dataset structure** matches expected format
5. **Try with smaller subset** if having memory issues

---

## 📜 Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{CamouflageAwareMedicalSegmentation,
  title={Camouflage-Aware Medical Image Segmentation},
  author={[Author Names]},
  year={2024},
  journal={[Journal Name]}
}
```

Also cite COD10K:
```bibtex
@inproceedings{COD10K,
  title={Camouflaged Object Detection with 10,000 Images},
  author={Fan, Deng-Ping and {others}},
  booktitle={ICCV},
  year={2021}
}
```

---

## 📄 License

This implementation is provided as-is for research purposes. Please respect the licenses of the original COD10K dataset and cited papers.

---

## ✨ Key Features Summary

| Feature | Status | Details |
|---------|--------|---------|
| Flexible folder detection | ✅ | Auto-detects Option A, B, or custom |
| Image-mask pairing | ✅ | By filename stem, robust |
| Multiple formats | ✅ | .jpg, .png, .bmp, .tif, .tiff |
| Augmentations | ✅ | Flip, rotation (training only) |
| DataLoader creation | ✅ | Auto train/val/test split |
| Metadata extraction | ✅ | CSV export with statistics |
| Visualizations | ✅ | Raw and transformed samples |
| GPU support | ✅ | Automatic device detection |
| Reproducibility | ✅ | Fixed seed support |

---

## 🎉 You're Ready!

Your dataset pipeline is now ready. To proceed:

1. **Run**: `python inspect_dataset.py`
2. **Verify**: Check metadata CSV and visualizations
3. **Confirm**: Share dataset summary and batch shapes
4. **Next**: We'll implement CU1 (U-Net) and CUA1 (Attention U-Net) models

**Happy researching!** 🚀

---

*Last updated: 2024*
*For the Camouflage-Aware Medical Image Segmentation project*
