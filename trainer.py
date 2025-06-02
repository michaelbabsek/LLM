import contextlib
import itertools
from typing import Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from checkpoint import save_checkpoint
from config import Config


class Trainer:
    def __init__(
            self,
            cfg: Config,
            model: torch.nn.Module,
            optimizer: Optimizer,
            scheduler: LambdaLR,
            train_dl: DataLoader,
            val_dl: DataLoader,
            scaler: Optional[GradScaler] = None,
            ctx: Optional[autocast] = contextlib.nullcontext(),
            device: Optional[str] = "cpu",
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.ctx = ctx
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device

        self.model.to(self.device)
        self.train_iter = itertools.cycle(self.train_dl)
        self.val_iter = itertools.cycle(self.val_dl)

    # ─────────────────── helper functions to clean up code
    def _move(self, batch):
        x, y = batch
        return x.to(self.device), y.to(self.device)

    def _forward(self, x, y):
        with self.ctx:
            loss, _ = self.model(x, y)
        return loss

    def _backward(self, loss):
        loss = loss
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _opt_step(self):
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        grad_norm = clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return grad_norm

    def _evaluate(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(self.cfg.training.eval_iters):
                loss = self._forward(*self._move(next(self.val_iter)))
                losses.append(loss.detach())
        self.model.train()
        return torch.mean(torch.stack(losses)).item()

    # ─────────────────── training loop
    def train(self, start_step: int = 0, last_loss: float = float("inf")):
        torch.set_float32_matmul_precision("high")
        pbar = tqdm(range(start_step, self.cfg.training.train_iters), desc="Training")

        for step_idx in pbar:
            step_loss = 0.0
            for micro_step in range(self.cfg.training.grad_accum_steps):
                loss = self._forward(*self._move(next(self.train_iter))) / self.cfg.training.grad_accum_steps
                self._backward(loss)
                step_loss += loss.item()

            grad_norm = self._opt_step()

            if step_idx % self.cfg.run.log_interval == 0:
                wandb.log(
                    {
                        "train/loss": step_loss,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/grad_norm": grad_norm,
                    },
                    step=step_idx,
                )

            pbar.set_postfix(loss=f"{step_loss:.4f}")

            if step_idx % self.cfg.training.eval_interval == 0: # evaluate the model (also does it at the beginning to not have missing data in visualization)
                val_loss = self._evaluate()
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/perplexity": torch.exp(torch.tensor(val_loss)).item(),
                    },
                    step=step_idx,
                )

                if val_loss < last_loss: # save model if improved
                    save_checkpoint(
                        step=step_idx,
                        loss=val_loss,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        path=self.cfg.training.ckpt_path
                    )

                    last_loss = val_loss

        pbar.close()