from typing import Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Lambda
from tqdm import tqdm
import wandb

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
            ctx: Optional[autocast] = None,
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

    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        pbar = tqdm(total=self.cfg.training.eval_iters, desc="Evaluating")
        total_loss = 0.0
        val_iter = iter(self.val_dl)
        for step_idx in range(self.cfg.training.eval_iters):
            x, y = next(val_iter)

            x = x.to(self.device)
            y = y.to(self.device)

            with self.ctx:
                loss, _ = self.model(x, y)

            total_loss += loss.item()
            pbar.update(1)

        pbar.close()
        self.model.train()
        return total_loss / self.cfg.training.eval_iters  # avg val loss

    def train(self, start_step):
        torch.set_float32_matmul_precision('high')
        self.model.train()

        pbar = tqdm(total=self.cfg.training.train_iters, desc="Training")
        total_loss = 0.0
        global_step = 0

        train_iter = iter(self.train_dl)
        for step_idx in range(start_step, self.cfg.training.train_iters):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dl)
                x, y = next(train_iter)

            x = x.to(self.device)
            y = y.to(self.device)

            with self.ctx:
                loss, _ = self.model(x, y)

                step_loss = loss.item()
                total_loss += step_loss

                loss = loss / self.cfg.training.grad_accum_steps  # loss for optimizer based on accumulation steps

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step_idx + 1) % self.cfg.training.grad_accum_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        norm = clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        norm = clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    wandb.log({
                        "train/loss": step_loss,
                        "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                        "train/norm": norm
                    }, step=global_step)

                    global_step += 1

            pbar.update(1)
            pbar.set_postfix(step_loss=f"{step_loss:.4f}")

            if (step_idx + 1) % self.cfg.training.eval_interval == 0:
                val_loss = self.estimate_loss()
                val_perplexity = torch.exp(torch.tensor(val_loss)).item()
                wandb.log({
                    "val/loss": val_loss,
                    "val/perplexity": val_perplexity
                }, step=global_step)

        pbar.close()