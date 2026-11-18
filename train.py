
import contextlib
import math
import random
import time
from dataclasses import asdict

import torch

import wandb
from checkpoint import load_checkpoint
from config import load_cfg
from dataset import BinDataset
from model import Transformer
from tokenizer import Tokenizer
from trainer import Trainer

cfg = load_cfg()

# ─────────────────── seeds + wandb
torch.manual_seed(cfg.run.seed); random.seed(cfg.run.seed)
run_name = cfg.run.name or time.strftime('%Y%m%d_%H%M%S')

wandb.init( project=cfg.run.project, name=run_name, config=asdict(cfg))

# ─────────────────── device
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print("Using device: ", device)
autocast_ctx   = torch.amp.autocast('cuda', dtype=torch.bfloat16) if device=='cuda' else contextlib.nullcontext()
scaler         = torch.amp.GradScaler() if device=='cuda' else None

# ─────────────────── tokenizer
tokenizer = Tokenizer()

# ─────────────────── model
cfg.model.vocab_size = 50304  # gpt2 + special tokens (50267), padded to a multiple of 64 for efficiency
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
last_loss = float('inf')
if cfg.training.use_checkpoint:
    ckpt = load_checkpoint(model, optimizer, scheduler, scaler, cfg.training.ckpt_path, device)
    start_step = ckpt['step']
    last_loss = ckpt['loss']

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

    )

    total_params = sum(p.numel() for p in model.parameters())  # tied weights are counted once
    print(f"total params: {total_params}")
    wandb.config.update({"n_params": total_params})
    wandb.run.summary["n_params"] = total_params

    trainer.train(start_step=start_step, last_loss=last_loss)