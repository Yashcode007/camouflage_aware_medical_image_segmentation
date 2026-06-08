# 🚀 Quick Reference: Exact Commands to Run

## Step 1: Navigate to Project Directory

```bash
cd camouflage_medical_segmentation
```

## Step 2: Create Virtual Environment (Optional but Recommended)

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

*This will install:*
- torch, torchvision
- numpy, pandas
- pillow, opencv-python
- matplotlib, seaborn
- tqdm, scikit-learn
- jupyter, ipython

## Step 4: Download Dataset

1. Go to: https://www.kaggle.com/datasets/aryehgod/camouflage-object-detection-cod10k
2. Download the dataset
3. Extract to: `camouflage_medical_segmentation/data/COD10K/`

**Verify structure:**
```bash
# Windows
dir data\COD10K\Train\Image

# Linux
ls data/COD10K/Train/Image
```

Should show 4040+ .jpg files

## Step 5: Run Dataset Inspection

### Option A: Standalone Script (Recommended)
```bash
python inspect_dataset.py
```

**Output will include:**
- Console summary with dataset statistics
- CSV: `outputs/cod10k_metadata.csv`
- PNG visualizations: `outputs/dataset_visualizations/`

### Option B: Interactive Jupyter Notebook

```bash
jupyter notebook notebooks/01_dataset_inspection.ipynb
```

Then run cells sequentially.

## Step 6: Verify DataLoader Works

### Quick Python Test

```bash
python << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "src")

from config import COD10K_ROOT, DEFAULT_IMAGE_SIZE, DEFAULT_BATCH_SIZE
from datasets import create_dataloaders
import torch

print(f"Dataset root: {COD10K_ROOT}")
print(f"Dataset exists: {COD10K_ROOT.exists()}")

if COD10K_ROOT.exists():
    print("\n[Creating DataLoaders...]")
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=COD10K_ROOT,
        batch_size=DEFAULT_BATCH_SIZE,
        image_size=DEFAULT_IMAGE_SIZE,
        num_workers=0,
    )
    
    print(f"\n[Getting batch...]")
    batch = next(iter(train_loader))
    
    print(f"\n✅ BATCH VERIFICATION")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Expected:    [16, 3, 256, 256]")
    print(f"Match: {batch['image'].shape == torch.Size([16, 3, 256, 256])}")
    
    print(f"\nMask shape:  {batch['mask'].shape}")
    print(f"Expected:    [16, 1, 256, 256]")
    print(f"Match: {batch['mask'].shape == torch.Size([16, 1, 256, 256])}")
    
    print(f"\nImage value range: [{batch['image'].min():.4f}, {batch['image'].max():.4f}]")
    print(f"Mask unique values: {torch.unique(batch['mask'])}")
    
    print("\n✅ DataLoader works correctly!")
else:
    print("❌ Dataset not found. Please download and extract COD10K.")
EOF
```

---

## 📋 Expected Output

### Dataset Inspection Summary (from `python inspect_dataset.py`)

```
================================================================================
COD10K DATASET INSPECTION PIPELINE
================================================================================

[Step 1/3] Scanning dataset structure and building metadata...
Dataset root: c:\...\camouflage_medical_segmentation\data\COD10K

[COD10KDataset] Loaded 4040 samples from train split
[COD10KDataset] Loaded 1000 samples from test split

================================================================================
DATASET INSPECTION SUMMARY
================================================================================

📊 Total Samples: 5040
   ├─ Train: 4040
   ├─ Val:   404
   └─ Test:  1000

⚠️  Data Quality Issues: 0
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

[Step 2/3] Creating DataLoaders and verifying batch format...

[DataLoader] Train split: 3636 samples
[DataLoader] Val split: 404 samples

✅ DataLoaders created successfully!
   Train batches: 227
   Val batches:   26
   Test batches:  63

[Step 2/3] Verifying first batch...

  Batch keys: ['image', 'mask', 'image_path', 'mask_path', 'filename']
  Image tensor shape: torch.Size([16, 3, 256, 256])
  Mask tensor shape:  torch.Size([16, 1, 256, 256])
  Image dtype: torch.float32
  Mask dtype:  torch.float32
  Image value range: [0.0000, 1.0000]
  Mask unique values: tensor([0., 1.])

✅ Batch format verified!
   Expected tensor shapes confirmed:
   - Image: [B, 3, 256, 256]
   - Mask:  [B, 1, 256, 256]

[Step 3/3] Saving batch visualizations...
✅ Visualizations saved to: c:\...\outputs\dataset_visualizations\

================================================================================
INSPECTION COMPLETE ✅
================================================================================

Metadata saved to:     c:\...\outputs\cod10k_metadata.csv
Visualizations saved to: c:\...\outputs\dataset_visualizations\

Next steps:
  1. Review the metadata CSV to understand dataset structure
  2. Check visualizations to verify data quality
  3. If everything looks good, proceed to Stage 1 model training

Dataset summary:
  - Total samples: 5040
  - Train/Val/Test split: 4040/404/1000
  - Data quality issues: 0

🎉 Dataset is clean and ready for training!
================================================================================
```

---

## 🔧 Advanced Commands

### Run with Custom Dataset Path
```bash
python inspect_dataset.py --dataset-root /path/to/COD10K
```

### Increase Number of Visualizations
```bash
python inspect_dataset.py --num-viz 10
```

### Skip DataLoader Verification (Faster)
```bash
python inspect_dataset.py --skip-dataloader
```

### Custom Batch Size
```bash
python inspect_dataset.py --batch-size 8
```

### Custom Image Size
```bash
python inspect_dataset.py --image-size 512
```

---

## ✅ Checklist: What to Verify

After running `python inspect_dataset.py`, verify:

- [ ] **Console output** shows dataset summary with correct split counts
- [ ] **No errors** related to missing folders or corrupted files
- [ ] **Batch verification** shows:
  ```
  Image tensor shape: torch.Size([16, 3, 256, 256])
  Mask tensor shape:  torch.Size([16, 1, 256, 256])
  Mask unique values: tensor([0., 1.])
  ```
- [ ] **CSV file** created at `outputs/cod10k_metadata.csv`
- [ ] **PNG visualizations** created in `outputs/dataset_visualizations/`
- [ ] **No data quality issues** (0 missing masks, corrupted files, etc.)

---

## 📁 Expected File Structure After Setup

```
camouflage_medical_segmentation/
├── src/
│   ├── config.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── cod10k_dataset.py
│   └── utils/
│       ├── __init__.py
│       └── dataset_inspection.py
├── notebooks/
│   └── 01_dataset_inspection.ipynb
├── data/
│   └── COD10K/              ← Extract dataset here
│       ├── Train/
│       │   ├── Image/       (4040 .jpg files)
│       │   └── GT_Object/   (4040 .jpg files)
│       └── Test/
│           ├── Image/       (1000 .jpg files)
│           └── GT_Object/   (1000 .jpg files)
├── checkpoints/             ← Will store CU1.pth, CUA1.pth
├── outputs/
│   ├── cod10k_metadata.csv  ← Generated
│   └── dataset_visualizations/  ← Generated
├── inspect_dataset.py
├── requirements.txt
├── README.md
├── IMPLEMENTATION_SUMMARY.md
└── quickstart.sh
```

---

## 🚨 Troubleshooting Commands

### Check if Dataset Exists
```bash
# Windows
dir data\COD10K\Train\Image | find ".jpg"

# Linux
ls data/COD10K/Train/Image | grep -c ".jpg"
```

### Test Python Import
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Check Virtual Environment
```bash
# Windows
venv\Scripts\python --version

# Linux
venv/bin/python --version
```

### Install Specific Package
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📝 What to Report Back

After running the inspection, please report:

1. **Folder structure** - Copy output of:
   ```bash
   python inspect_dataset.py | head -20
   ```

2. **Dataset summary** - Copy output of:
   ```bash
   Dataset summary:
     - Total samples: [YOUR NUMBER]
     - Train/Val/Test split: [YOUR SPLIT]
     - Data quality issues: [YOUR COUNT]
   ```

3. **Batch verification** - Copy output of:
   ```bash
   Image tensor shape: [YOUR SHAPE]
   Mask tensor shape: [YOUR SHAPE]
   Mask unique values: [YOUR VALUES]
   ```

4. **Any errors** - Copy full error message if any

---

## 💡 Pro Tips

### Faster DataLoader (Production)
```python
# In your training script
train_loader, val_loader, test_loader = create_dataloaders(
    root_dir="data/COD10K",
    num_workers=4,  # Increase based on CPU cores
    batch_size=32,
    # ...
)
```

### Debug Mode
```python
# Check a few samples before training
for i in range(5):
    sample = dataset[i]
    print(f"Sample {i}: {sample['filename']}")
```

### Memory Optimization
```python
# Reduce batch size if GPU runs out of memory
create_dataloaders(
    batch_size=8,       # Reduce from 16
    image_size=192,     # Reduce from 256
)
```

---

## 📞 Getting Help

If you encounter issues:

1. **Check README.md** - Full documentation
2. **Review IMPLEMENTATION_SUMMARY.md** - Design decisions
3. **Check console error** - Detailed error message
4. **Inspect outputs/** - Check CSV and visualizations
5. **Run with verbose mode** - See detailed logs

---

**Next Step**: Run `python inspect_dataset.py` and report results! 🚀
