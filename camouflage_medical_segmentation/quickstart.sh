#!/bin/bash
# Quick Start Script for Camouflage-Aware Medical Image Segmentation
# Stage 1: COD10K Dataset & DataLoader Setup

echo "=================================================="
echo "Stage 1 Quick Start - COD10K Dataset Pipeline"
echo "=================================================="

# Check Python version
echo ""
echo "[1/4] Checking Python installation..."
python --version

# Create virtual environment (optional)
echo ""
echo "[2/4] Install dependencies..."
echo "Run: pip install -r requirements.txt"

# Run inspection
echo ""
echo "[3/4] Run dataset inspection..."
echo "Run: python inspect_dataset.py"

# Open notebook
echo ""
echo "[4/4] Run Jupyter notebook (optional)..."
echo "Run: jupyter notebook notebooks/01_dataset_inspection.ipynb"

echo ""
echo "=================================================="
echo "Quick verification commands:"
echo "=================================================="
echo ""
echo "# Test dataset loading"
python << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src")))

try:
    from config import COD10K_ROOT
    from datasets import COD10KSegmentationDataset
    
    print(f"✅ Imports successful")
    print(f"   Dataset root configured at: {COD10K_ROOT}")
    
    if COD10K_ROOT.exists():
        print(f"✅ Dataset directory exists")
        
        # Try loading dataset
        try:
            dataset = COD10KSegmentationDataset(
                root_dir=COD10K_ROOT,
                split='train',
                image_size=256,
            )
            print(f"✅ Dataset loaded: {len(dataset)} samples")
            
            # Get one sample
            sample = dataset[0]
            print(f"✅ Sample loaded")
            print(f"   Image shape: {sample['image'].shape}")
            print(f"   Mask shape: {sample['mask'].shape}")
        except Exception as e:
            print(f"⚠️  Error loading dataset: {e}")
    else:
        print(f"⚠️  Dataset directory not found: {COD10K_ROOT}")
        print(f"   Download from: https://www.kaggle.com/datasets/aryehgod/camouflage-object-detection-cod10k")
        
except Exception as e:
    print(f"❌ Import error: {e}")
EOF

echo ""
echo "=================================================="
echo "Setup complete! Next steps:"
echo "=================================================="
echo ""
echo "1. Download COD10K dataset:"
echo "   https://www.kaggle.com/datasets/aryehgod/camouflage-object-detection-cod10k"
echo ""
echo "2. Extract to: data/COD10K/"
echo ""
echo "3. Run inspection: python inspect_dataset.py"
echo ""
echo "4. Verify outputs in: outputs/"
echo ""
echo "=================================================="
