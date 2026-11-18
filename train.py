
import argparse
import contextlib
import math
import random
import time
from dataclasses import asdict

import torch

import wandb
from checkpoint import load_checkpoint, latest_checkpoint
from config import load_cfg
from dataset import BinDataset
from model import Transformer
from tokenizer import Tokenizer
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yml', help='yaml config, e.g. config.smoke.yml or config.gpu.yml')
parser.add_argument('--resume', default=None,
                    help='checkpoint to resume from, e.g. ./checkpoints/gpu/step_005000.pt; starts a new wandb run. '
                         'Default: the newer of last.pt and best.pt in ckpt_dir')
parser.add_argument('--lr-scale', type=float, default=None,
                    help='multiplies the lr schedule (max_lr to min_lr) for the rest of the run, e.g. 0.5, without '
                         'resetting the step. Relative to the config, stored in the checkpoint, so it doesn\'t '
                         'compound over restarts')
args = parser.parse_args()
cfg = load_cfg(args.config)
print(f"Using config: {args.config}")

# ─────────────────── seeds
torch.manual_seed(cfg.run.seed); random.seed(cfg.run.seed)
run_name = cfg.run.name or time.strftime('%Y%m%d_%H%M%S')

# ─────────────────── device
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print("Using device: ", device)
amp_dtype      = torch.bfloat16
autocast_ctx   = torch.amp.autocast('cuda', dtype=amp_dtype) if device=='cuda' else contextlib.nullcontext()
# loss scaling is only needed for float16 (narrow exponent range), bfloat16 has the same range as float32
scaler         = torch.amp.GradScaler() if device=='cuda' and amp_dtype==torch.float16 else None

# ─────────────────── tokenizer
tokenizer = Tokenizer()

# ─────────────────── model
cfg.model.vocab_size = tokenizer.padded_vocab_size()  # gpt2 + special tokens (50267), padded to 50304
model = Transformer(cfg.model).to(device)
if torch.__version__ >= '2' and device=='cuda': model.compile()

# ─────────────────── optimizer + scheduler
optimizer = torch.optim.AdamW(
    model.get_optimizer_grouped_parameters(cfg.optim.weight_decay),
    lr=cfg.optim.max_lr, betas=(cfg.optim.beta1, cfg.optim.beta2), eps=cfg.optim.eps)

def lr_lambda(step_idx):
    if step_idx < cfg.training.warmup_iters:
        return step_idx / cfg.training.warmup_iters
    progress = (step_idx - cfg.training.warmup_iters) / (cfg.training.train_iters - cfg.training.warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return (cfg.optim.min_lr / cfg.optim.max_lr) + (1 - cfg.optim.min_lr / cfg.optim.max_lr) * cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ─────────────────── resume?
start_step = 0
best_val_loss = float('inf')
wandb_run_id = None
train_gen = torch.Generator().manual_seed(cfg.run.seed)
ckpt_path = args.resume or (latest_checkpoint(cfg.training.ckpt_dir) if cfg.training.use_checkpoint else None)
if ckpt_path is not None:
    ckpt = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, train_gen)
    start_step, best_val_loss, wandb_run_id = ckpt['step'], ckpt['best_val_loss'], ckpt['wandb_run_id']
    print(f"resuming from {ckpt_path} at step {start_step} (best val loss {best_val_loss:.4f})")
    if args.resume:  # going back in steps, wandb ignores steps lower than the ones already logged in a run
        wandb_run_id, run_name = None, f"{run_name}-from-{start_step}"

# the scale lives in scheduler.base_lrs, which is part of the checkpoint
if args.lr_scale is not None:
    scheduler.base_lrs = [cfg.optim.max_lr * args.lr_scale for _ in scheduler.base_lrs]
    for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
        group['lr'] = base_lr * lr_lambda(scheduler.last_epoch)  # lr of the next step
print(f"lr scale {scheduler.base_lrs[0] / cfg.optim.max_lr:g} (next lr {optimizer.param_groups[0]['lr']:.3e})")

# ─────────────────── wandb (continues the run stored in the checkpoint)
wandb.init(project=cfg.run.project, name=run_name, config=asdict(cfg), id=wandb_run_id,
           resume='allow' if wandb_run_id else None)

# ─────────────────── data
train_data = BinDataset('train', cfg.model.max_seq_len, cfg.training.data_dir, device)
val_data   = BinDataset('val',   cfg.model.max_seq_len, cfg.training.data_dir, device)

if __name__ == "__main__":
    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ctx=autocast_ctx,
        device=device,
        train_data=train_data,
        val_data=val_data,
        train_gen=train_gen,
    )

    total_params = sum(p.numel() for p in model.parameters())  # tied weights are counted once
    print(f"total params: {total_params}")
    wandb.config.update({"n_params": total_params})
    wandb.run.summary["n_params"] = total_params

    trainer.train(start_step=start_step, best_val_loss=best_val_loss)