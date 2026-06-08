#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

# Add src to path so imports resolve correctly
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import (
    CHECKPOINT_DIR,
    COD10K_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    DEFAULT_VAL_RATIO,
    METADATA_CSV_PATH,
    OUTPUT_DIR,
    PREDICTIONS_DIR,
    USE_NON_EMPTY_MASKS_ONLY,
    VALIDATE_NON_EMPTY_MASKS_ONLY,
)
from datasets import create_dataloaders
from losses import BCEDiceLoss
from metrics import dice_score, iou_score
from models import AttentionUNet, UNet
from utils.checkpointing import save_checkpoint
from utils.seed import set_seed
from utils.visualization import save_prediction_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1 camouflage expert models.")

    parser.add_argument("--model_type", choices=["unet", "attention_unet"], default="unet")
    parser.add_argument("--data_root", type=str, default=str(COD10K_ROOT))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--save_viz_every", type=int, default=5)
    parser.add_argument("--metadata_csv_path", type=str, default=str(METADATA_CSV_PATH))

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_non_empty_masks_only", dest="use_non_empty_masks_only", action="store_true")
    group.add_argument("--use_all_masks", dest="use_non_empty_masks_only", action="store_false")
    parser.set_defaults(use_non_empty_masks_only=USE_NON_EMPTY_MASKS_ONLY)

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("--validate_non_empty", dest="validate_non_empty", action="store_true")
    group2.add_argument("--validate_with_all", dest="validate_non_empty", action="store_false")
    parser.set_defaults(validate_non_empty=VALIDATE_NON_EMPTY_MASKS_ONLY)

    return parser.parse_args()


def build_model(model_type: str, base_channels: int, device: torch.device) -> torch.nn.Module:
    if model_type == "unet":
        model = UNet(in_channels=3, out_channels=1, base_channels=base_channels)
    else:
        model = AttentionUNet(in_channels=3, out_channels=1, base_channels=base_channels)
    return model.to(device)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)

    return epoch_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validate", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)

            loss = criterion(logits, masks)
            batch_size = images.size(0)
            val_loss += loss.item() * batch_size
            dice_sum += dice_score(logits, masks) * batch_size
            iou_sum += iou_score(logits, masks) * batch_size
            total_samples += batch_size

    return (
        val_loss / total_samples,
        dice_sum / total_samples,
        iou_sum / total_samples,
    )


def get_prediction_dir(output_dir: Path, model_type: str) -> Path:
    model_name = "CU1" if model_type == "unet" else "CUA1"
    return output_dir / "predictions" / model_name


def get_checkpoint_prefix(model_type: str) -> str:
    return "CU1" if model_type == "unet" else "CUA1"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    prediction_dir = get_prediction_dir(output_dir, args.model_type)
    checkpoint_prefix = get_checkpoint_prefix(args.model_type)

    print(f"\n[Stage 1 Training] model={args.model_type}, device={device}, seed={args.seed}")
    print(f"Data root: {args.data_root}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Use non-empty masks only: {args.use_non_empty_masks_only}")
    print(f"Validate non-empty masks only: {args.validate_non_empty}")

    train_loader, val_loader, _ = create_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        val_ratio=DEFAULT_VAL_RATIO,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        normalize=False,
        augment=True,
        use_non_empty_masks_only=args.use_non_empty_masks_only,
        metadata_csv_path=args.metadata_csv_path,
    )

    model = build_model(args.model_type, args.base_channels, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
    criterion = BCEDiceLoss()

    best_val_dice = 0.0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = validate_epoch(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Dice:   {val_dice:.4f}")
        print(f"  Val IoU:    {val_iou:.4f}")

        scheduler.step(val_dice)

        checkpoint_config = {
            "model_type": args.model_type,
            "data_root": args.data_root,
            "image_size": args.image_size,
            "base_channels": args.base_channels,
            "batch_size": args.batch_size,
            "use_non_empty_masks_only": args.use_non_empty_masks_only,
            "validate_non_empty": args.validate_non_empty,
            "seed": args.seed,
            "lr": args.lr,
        }

        final_path = checkpoint_dir / f"{checkpoint_prefix}_final.pth"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_dice=best_val_dice,
            config=checkpoint_config,
            filepath=final_path,
        )

        if val_dice >= best_val_dice:
            best_val_dice = val_dice
            best_path = checkpoint_dir / f"{checkpoint_prefix}_best.pth"
            alias_path = checkpoint_dir / f"{checkpoint_prefix}.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_dice=best_val_dice,
                config=checkpoint_config,
                filepath=best_path,
                alias_path=alias_path,
            )
            print(f"  Saved best checkpoint: {best_path}")

        if epoch % args.save_viz_every == 0 or epoch == 1:
            val_batch = next(iter(val_loader))
            val_images = val_batch["image"].to(device)
            val_masks = val_batch["mask"].to(device)
            val_logits = model(val_images)
            viz_path = prediction_dir / f"epoch_{epoch:03d}_predictions.png"
            save_prediction_grid(
                images=val_images,
                masks=val_masks,
                logits=val_logits,
                output_path=viz_path,
                num_samples=min(4, val_images.shape[0]),
            )
            print(f"  Saved prediction visualization: {viz_path}")

    print("\nStage 1 training complete.")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Saved checkpoints to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
