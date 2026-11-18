import contextlib
import time
from dataclasses import asdict
from typing import Optional

import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import wandb
from checkpoint import save_checkpoint, best_path, last_path
from config import Config
from dataset import BinDataset


class Trainer:
    def __init__(
            self,
            cfg: Config,
            model: torch.nn.Module,
            optimizer: Optimizer,
            scheduler: LambdaLR,
            train_data: BinDataset,
            val_data: BinDataset,
            scaler: Optional[GradScaler] = None,
            ctx: Optional[autocast] = contextlib.nullcontext(),
            device: Optional[str] = "cpu",
            train_gen: Optional[torch.Generator] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.ctx = ctx
        self.train_data = train_data
        self.val_data = val_data
        self.device = device

        self.model.to(self.device)
        # continuous generator for training batches, seeded from the config (its state is part of the checkpoint)
        self.train_gen = train_gen if train_gen is not None else torch.Generator().manual_seed(cfg.run.seed)
        self.tokens_per_step = cfg.training.batch_size * cfg.training.grad_accum_steps * cfg.model.max_seq_len

    # ─────────────────── helper functions to clean up code
    def _train_batch(self):
        return self.train_data.get_batch(self.cfg.training.batch_size, self.train_gen)

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
        # fresh generator with a fixed seed: every evaluation uses the same validation windows
        val_gen = torch.Generator().manual_seed(self.cfg.training.eval_seed)
        self.model.eval()
        losses = []
        with torch.no_grad():  # autocast is entered in _forward only, entering the same ctx twice leaks its state
            for _ in range(self.cfg.training.eval_iters):
                x, y = self.val_data.get_batch(self.cfg.training.eval_batch_size, val_gen)
                losses.append(self._forward(x, y).detach())
        self.model.train()
        return torch.mean(torch.stack(losses)).item()

    def _memory_gb(self) -> float:
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / 1e9
        if self.device == "mps":
            return torch.mps.driver_allocated_memory() / 1e9
        return 0.0

    def _save(self, path: str, step: int, best_val_loss: float):
        save_checkpoint(
            path=path,
            step=step,
            best_val_loss=best_val_loss,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            train_rng=self.train_gen,
            wandb_run_id=wandb.run.id if wandb.run else None,
            config=asdict(self.cfg),
        )

    # ─────────────────── training loop
    def train(self, start_step: int = 0, best_val_loss: float = float("inf")):
        torch.set_float32_matmul_precision("high")
        train_iters = self.cfg.training.train_iters
        ckpt_dir = self.cfg.training.ckpt_dir
        pbar = tqdm(range(start_step, train_iters), desc="Training", initial=start_step, total=train_iters)
        t_last, steps_since_log = time.perf_counter(), 0
        completed = start_step  # number of finished optimizer steps

        try:
            for step_idx in pbar:
                step_loss = 0.0
                for micro_step in range(self.cfg.training.grad_accum_steps):
                    loss = self._forward(*self._train_batch()) / self.cfg.training.grad_accum_steps
                    self._backward(loss)
                    step_loss += loss.item()

                lr = self.optimizer.param_groups[0]["lr"]  # lr used for this step, read before the scheduler advances
                grad_norm = self._opt_step()
                steps_since_log += 1
                completed = step_idx + 1

                if step_idx % self.cfg.run.log_interval == 0:
                    t_now = time.perf_counter()  # loss.item() above already synchronized with the device
                    tokens_per_sec = steps_since_log * self.tokens_per_step / (t_now - t_last)
                    t_last, steps_since_log = t_now, 0
                    mem_gb = self._memory_gb()
                    wandb.log(
                        {
                            "train/loss": step_loss,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "train/tokens_seen": completed * self.tokens_per_step,
                            "perf/tokens_per_sec": tokens_per_sec,
                            "perf/memory_gb": mem_gb,
                        },
                        step=step_idx,
                    )
                    pbar.write(f"step {step_idx:6d} | loss {step_loss:.4f} | lr {lr:.3e} | "
                               f"{tokens_per_sec:,.0f} tok/s | mem {mem_gb:.2f} GB")

                pbar.set_postfix(loss=f"{step_loss:.4f}")

                # evaluate at the first step (no gap in the curves), every eval_interval and at the very end
                if step_idx % self.cfg.training.eval_interval == 0 or completed == train_iters:
                    val_loss = self._evaluate()
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/perplexity": torch.exp(torch.tensor(val_loss)).item(),
                        },
                        step=step_idx,
                    )
                    pbar.write(f"step {step_idx:6d} | val loss {val_loss:.4f}")

                    if val_loss < best_val_loss:  # save model if improved
                        best_val_loss = val_loss
                        self._save(best_path(ckpt_dir), completed, best_val_loss)
                    t_last, steps_since_log = time.perf_counter(), 0  # keep evaluation time out of the throughput

                if completed % self.cfg.training.ckpt_interval == 0 or completed == train_iters:
                    self._save(last_path(ckpt_dir), completed, best_val_loss)
                    pbar.write(f"saved {last_path(ckpt_dir)} at step {completed}")
                    t_last, steps_since_log = time.perf_counter(), 0

        except KeyboardInterrupt:
            self._save(last_path(ckpt_dir), completed, best_val_loss)
            pbar.write(f"interrupted, saved {last_path(ckpt_dir)} at step {completed}")
        finally:
            pbar.close()
