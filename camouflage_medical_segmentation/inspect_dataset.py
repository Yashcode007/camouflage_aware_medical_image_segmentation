#!/usr/bin/env python3
"""
Dataset Inspection Script for COD10K.

This script performs comprehensive inspection of the COD10K dataset:
1. Scans image and mask folders
2. Builds a metadata table
3. Checks for data quality issues
4. Saves results to CSV
5. Generates sample visualizations
6. Verifies DataLoader functionality

Usage:
    python inspect_dataset.py --dataset-root data/COD10K

To run with defaults (assumes data/COD10K exists):
    python inspect_dataset.py
"""

import argparse
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import pandas as pd

# Import our modules
from config import (
    COD10K_ROOT, OUTPUT_DIR, VISUALIZATION_DIR, METADATA_CSV_PATH,
    DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE, DEFAULT_NUM_WORKERS,
    DEFAULT_SEED, VERBOSE
)
from datasets import COD10KSegmentationDataset, create_dataloaders
from utils import inspect_dataset, visualize_sample, save_batch_visualizations


def main():
    """Main inspection pipeline."""
    
    parser = argparse.ArgumentParser(
        description="Inspect COD10K dataset for pre-training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(COD10K_ROOT),
        help="Path to COD10K dataset root directory"
    )
    
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(METADATA_CSV_PATH),
        help="Path to save metadata CSV"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(VISUALIZATION_DIR),
        help="Directory to save visualizations"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for DataLoader verification"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Target image size (H=W)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of DataLoader workers"
    )
    
    parser.add_argument(
        "--num-viz",
        type=int,
        default=5,
        help="Number of samples to visualize"
    )
    
    parser.add_argument(
        "--skip-dataloader",
        action="store_true",
        help="Skip DataLoader verification (only run file inspection)"
    )
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_csv = Path(args.output_csv)
    output_dir = Path(args.output_dir)
    
    print("\n" + "="*80)
    print("COD10K DATASET INSPECTION PIPELINE")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Inspect dataset files
    # ========================================================================
    print("\n[Step 1/3] Scanning dataset structure and building metadata...")
    print(f"Dataset root: {dataset_root}")
    
    if not dataset_root.exists():
        print(f"❌ Error: Dataset root does not exist: {dataset_root}")
        print("\nPlease download COD10K dataset or adjust --dataset-root path.")
        print("Download link: https://www.kaggle.com/datasets/aryehgod/camouflage-object-detection-cod10k")
        sys.exit(1)
    
    try:
        metadata_df, summary = inspect_dataset(
            root_dir=dataset_root,
            output_csv=output_csv,
        )
    except Exception as e:
        print(f"❌ Error during inspection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # STEP 2: Create DataLoaders
    # ========================================================================
    if not args.skip_dataloader:
        print("\n[Step 2/3] Creating DataLoaders and verifying batch format...")
        
        try:
            train_loader, val_loader, test_loader = create_dataloaders(
                root_dir=dataset_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                image_size=args.image_size,
                normalize=False,
                augment=True,
            )
            
            print(f"\n✅ DataLoaders created successfully!")
            print(f"   Train batches: {len(train_loader)}")
            print(f"   Val batches:   {len(val_loader)}")
            print(f"   Test batches:  {len(test_loader)}")
            
            # Get one batch and verify
            print("\n[Step 2/3] Verifying first batch...")
            batch = next(iter(train_loader))
            
            print(f"\n  Batch keys: {list(batch.keys())}")
            print(f"  Image tensor shape: {batch['image'].shape}")
            print(f"  Mask tensor shape:  {batch['mask'].shape}")
            print(f"  Image dtype: {batch['image'].dtype}")
            print(f"  Mask dtype:  {batch['mask'].dtype}")
            print(f"  Image value range: [{batch['image'].min():.4f}, {batch['image'].max():.4f}]")
            print(f"  Mask unique values: {torch.unique(batch['mask'])}")
            
            # Check consistency
            assert batch['image'].shape[0] == args.batch_size, "Batch size mismatch"
            assert batch['image'].shape[1] == 3, "Image should have 3 channels"
            assert batch['image'].shape[2] == args.image_size, f"Image height should be {args.image_size}"
            assert batch['image'].shape[3] == args.image_size, f"Image width should be {args.image_size}"
            assert batch['mask'].shape[0] == args.batch_size, "Batch size mismatch"
            assert batch['mask'].shape[1] == 1, "Mask should have 1 channel"
            assert batch['mask'].shape[2] == args.image_size, f"Mask height should be {args.image_size}"
            assert batch['mask'].shape[3] == args.image_size, f"Mask width should be {args.image_size}"
            
            print("\n✅ Batch format verified!")
            print("   Expected tensor shapes confirmed:")
            print(f"   - Image: [B, 3, {args.image_size}, {args.image_size}]")
            print(f"   - Mask:  [B, 1, {args.image_size}, {args.image_size}]")
            
        except Exception as e:
            print(f"\n❌ Error during DataLoader creation: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # ====================================================================
        # STEP 3: Save visualizations
        # ====================================================================
        print(f"\n[Step 3/3] Saving batch visualizations...")
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_batch_visualizations(
                loader=train_loader,
                output_dir=output_dir,
                num_samples=args.num_viz,
                show_transformed=True,
            )
            print(f"\n✅ Visualizations saved to: {output_dir}")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not save visualizations: {e}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("INSPECTION COMPLETE ✅")
    print("="*80)
    print(f"\nMetadata saved to:     {output_csv}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review the metadata CSV to understand dataset structure")
    print(f"  2. Check visualizations to verify data quality")
    print(f"  3. If everything looks good, proceed to Stage 1 model training")
    print(f"\nDataset summary:")
    print(f"  - Total samples: {summary['total_samples']}")
    print(f"  - Train/Val/Test split: {summary['train_count']}/{summary['val_count']}/{summary['test_count']}")
    print(f"  - Data quality issues: {summary['missing_masks'] + summary['corrupted_images'] + summary['corrupted_masks']}")
    
    if summary['missing_masks'] > 0 or summary['corrupted_images'] > 0 or summary['corrupted_masks'] > 0:
        print(f"\n⚠️  Please address data quality issues before proceeding to training!")
    else:
        print(f"\n🎉 Dataset is clean and ready for training!")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
