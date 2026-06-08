#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from torch import optim

# Add src to path so imports resolve correctly
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import (
    COD10K_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    METADATA_CSV_PATH,
    OUTPUT_DIR,
)
from datasets import create_dataloaders
from losses import BCEDiceLoss
from metrics import dice_score, iou_score
from models import AttentionUNet, UNet
from utils.checkpointing import load_checkpoint
from utils.seed import set_seed
from utils.visualization import save_prediction_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Stage 1 camouflage expert model.")

    parser.add_argument("--model_type", choices=["unet", "attention_unet"], default="unet")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=str(COD10K_ROOT))
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--metadata_csv_path", type=str, default=str(METADATA_CSV_PATH))
    parser.add_argument("--test_non_empty_only", action="store_true")
    parser.add_argument("--use_all_test_masks", dest="test_non_empty_only", action="store_false")
    parser.set_defaults(test_non_empty_only=False)

    return parser.parse_args()


def build_model(model_type: str, base_channels: int, device: torch.device) -> torch.nn.Module:
    if model_type == "unet":
        model = UNet(in_channels=3, out_channels=1, base_channels=base_channels)
    else:
        model = AttentionUNet(in_channels=3, out_channels=1, base_channels=base_channels)
    return model.to(device)


def get_prediction_dir(output_dir: Path, model_type: str) -> Path:
    model_name = "CU1" if model_type == "unet" else "CUA1"
    return output_dir / "predictions" / model_name / "evaluation"


def evaluate(model, loader, criterion, device):
    model.eval()
    test_loss = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    total_samples = 0
    batch_count = 0

    with torch.no_grad():
        for batch in loader:
            batch_count += 1
            print(f"  Processing batch {batch_count}...", end="\r", flush=True)
            
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, masks)

            batch_size = images.size(0)
            test_loss += loss.item() * batch_size
            dice_sum += dice_score(logits, masks) * batch_size
            iou_sum += iou_score(logits, masks) * batch_size
            total_samples += batch_size
    
    print()  # New line after progress
    return test_loss / total_samples, dice_sum / total_samples, iou_sum / total_samples


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    prediction_dir = get_prediction_dir(output_dir, args.model_type)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n[Stage 1 Evaluation] model={args.model_type}, checkpoint={checkpoint_path}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Test non-empty only: {args.test_non_empty_only}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})
    base_channels = config.get("base_channels", args.base_channels)

    model = build_model(args.model_type, base_channels, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, test_loader = create_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        val_ratio=0.0,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        normalize=False,
        augment=False,
        use_non_empty_masks_only=args.test_non_empty_only,
        metadata_csv_path=args.metadata_csv_path,
    )

    criterion = BCEDiceLoss()
    print("[Evaluating...] Computing metrics on test set...")
    test_loss, test_dice, test_iou = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"Test IoU:  {test_iou:.4f}")

    print("[Visualizing...] Generating prediction grid...")
    val_batch = next(iter(test_loader))
    images = val_batch["image"].to(device)
    masks = val_batch["mask"].to(device)
    logits = model(images)
    viz_path = prediction_dir / f"evaluation_predictions.png"
    save_prediction_grid(
        images=images,
        masks=masks,
        logits=logits,
        output_path=viz_path,
        num_samples=min(4, images.shape[0]),
    )
    print(f"Saved evaluation predictions to: {viz_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
