import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch


@dataclass
class Checkpoint:
    step: int
    model: Dict[str, Any]
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    scaler: Optional[Dict[str, Any]] = None


CKPT_PATH = "checkpoint.pth"

def save_checkpoint(
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler=None,
        path: str = CKPT_PATH) -> None:

    state = Checkpoint(
        step=step,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        scaler=None if scaler is None else scaler.state_dict(),
    )

    torch.save(state.__dict__, path)

def load_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler=None,
        path: str = CKPT_PATH,
        device: str = "cpu") -> int:

    if not os.path.exists(path):
        return 0  # Training from scratch

    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt["step"]