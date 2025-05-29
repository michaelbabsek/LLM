import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class Checkpoint:
    step: int
    loss: float
    model: Dict[str, Any]
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    scaler: Optional[Dict[str, Any]] = None


CKPT_PATH = "checkpoint.pth"

def save_checkpoint(
        step: int,
        loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        scaler: GradScaler,
        path: str = CKPT_PATH) -> None:

    state = Checkpoint(
        step=step,
        loss=loss,
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
        device: str = "cpu") -> Dict[str, Any]:

    if not os.path.exists(path):
        return {'step': 0, 'loss': float('inf')} # Training from scratch

    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return {'step': ckpt["step"], 'loss': ckpt["loss"]}