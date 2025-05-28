
import contextlib, random, time, yaml, torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import wandb
from checkpoint import load_checkpoint
from dataset   import BinDataset
from model     import Transformer
from config import load_cfg, ModelCfg

from tokenizer import Tokenizer
from trainer import Trainer

from dataclasses import asdict

cfg = load_cfg()

# ─────────────────── seeds + wandb
torch.manual_seed(cfg.run.seed); random.seed(cfg.run.seed)
run_name = cfg.run.name or time.strftime('%Y%m%d_%H%M%S')

wandb.init( project=cfg.run.project, name=run_name, config=asdict(cfg))

# ─────────────────── device
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
autocast_ctx   = torch.cuda.amp.autocast() if device=='cuda' else contextlib.nullcontext()
scaler         = torch.cuda.amp.GradScaler() if device=='cuda' else None

# ─────────────────── tokenizer
tokenizer = Tokenizer()

# ─────────────────── model
model = Transformer(ModelCfg( vocab_size=len(tokenizer))).to(device)
if torch.__version__ >= '2' and device=='cuda': model.compile()

# ─────────────────── optimizer + scheduler
optimizer = torch.optim.AdamW(
    model.get_optimizer_grouped_parameters(cfg.optim.weight_decay),
    lr=cfg.optim.max_lr, betas=(cfg.optim.beta1, cfg.optim.beta2), eps=cfg.optim.eps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer, cfg.training.train_iters*cfg.training.warmup_frac,
    cfg.training.train_iters//cfg.training.grad_accum_steps, num_cycles=0.5)

# ─────────────────── resume?
start_step = 0
if cfg.training.use_checkpoint:
    start_step = load_checkpoint(model, optimizer, scheduler, scaler, cfg.training.ckpt_path, device)

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

    trainer.train(start_step=start_step)