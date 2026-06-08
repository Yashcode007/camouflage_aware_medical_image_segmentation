import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_dice: float,
    config: Dict[str, Any],
    filepath: Path,
    alias_path: Optional[Path] = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_dice": best_val_dice,
        "config": config,
    }
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)
    if alias_path is not None:
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filepath, alias_path)


def load_checkpoint(filepath: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint
