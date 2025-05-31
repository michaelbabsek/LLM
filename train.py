
import contextlib
import math
import random
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

import wandb
from checkpoint import load_checkpoint
from config import load_cfg, ModelCfg
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
autocast_ctx   = torch.amp.autocast(device_type=device) if device=='cuda' else contextlib.nullcontext()
scaler         = torch.amp.GradScaler() if device=='cuda' else None

# ─────────────────── tokenizer
tokenizer = Tokenizer()

# ─────────────────── model
model = Transformer(ModelCfg( vocab_size=len(tokenizer))).to(device)
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
train_ds = BinDataset(chunk_size=cfg.model.max_seq_len, split='train', device=device)
val_ds   = BinDataset(chunk_size=cfg.model.max_seq_len, split='val',   device=device)
train_dl = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=False, pin_memory=device=='cuda')
val_dl   = DataLoader(val_ds,  batch_size=cfg.training.batch_size, shuffle=False, pin_memory=device=='cuda')


if __name__ == "__main__":
    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ctx=autocast_ctx,
        device=device,
        train_dl=train_dl,
        val_dl=val_dl,

    )

    trainer.train(start_step=start_step, last_loss=last_loss)