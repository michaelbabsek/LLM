import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class Checkpoint:
    step: int  # number of completed optimizer steps, training resumes at this step
    best_val_loss: float
    model: Dict[str, Any]
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    scaler: Optional[Dict[str, Any]] = None
    train_rng: Optional[torch.Tensor] = None  # state of the training batch generator
    wandb_run_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None  # full Config as dict, lets sample.py rebuild the model


def best_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "best.pt")


def last_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "last.pt")


def save_checkpoint(
        path: str,
        step: int,
        best_val_loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        scaler: Optional[GradScaler] = None,
        train_rng: Optional[torch.Generator] = None,
        wandb_run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None) -> None:

    state = Checkpoint(
        step=step,
        best_val_loss=best_val_loss,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        scaler=None if scaler is None else scaler.state_dict(),
        train_rng=None if train_rng is None else train_rng.get_state(),
        wandb_run_id=wandb_run_id,
        config=config,
    )

    # write to a temporary file first, so an interrupted save never corrupts the existing checkpoint
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state.__dict__, tmp)
    os.replace(tmp, path)


def latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Returns whichever of last.pt and best.pt has the higher step, or None."""
    candidates = [p for p in (last_path(ckpt_dir), best_path(ckpt_dir)) if os.path.exists(p)]
    if not candidates:
        return None
    # mmap avoids reading the full file just to look at the step
    return max(candidates, key=lambda p: torch.load(p, map_location="cpu", mmap=True)["step"])


def load_checkpoint(
        path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        scaler: Optional[GradScaler] = None,
        train_rng: Optional[torch.Generator] = None) -> Dict[str, Any]:

    # load to cpu, load_state_dict copies into the parameters on their device (the rng state must stay on cpu)
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if train_rng is not None and ckpt.get("train_rng") is not None:
        train_rng.set_state(ckpt["train_rng"])

    return {'step': ckpt["step"], 'best_val_loss': ckpt["best_val_loss"], 'wandb_run_id': ckpt.get("wandb_run_id")}
