import contextlib
import os

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import wandb
from dataset import BinDataset
from model import Transformer, ModelArgs
from tokenizer import Tokenizer

#model args
n_dim: int = 768
n_blocks: int = 16
n_heads: int = 8
max_seq_len: int = 1024

# training
train_iters: int = 600_000
eval_iters: int = 10
eval_interval: int = 100
warmup_frac: float = 0.1
batch_size: int = 1
grad_clip: float = 1.0

# optimizer
max_lr: float = 6e-4
weight_decay: float = 1e-1
beta1: float = 0.9
beta2: float = 0.95
eps: float =1e-8


wandb.login()

run = wandb.init(project="LLM")

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_device()

cuda = device == "cuda"
ctx = torch.cuda.amp.autocast() if cuda else contextlib.nullcontext()
scaler = torch.cuda.amp.GradScaler() if cuda else None

tokenizer = Tokenizer()

model = Transformer(
    args=ModelArgs(
    n_dim=n_dim,
    n_blocks=n_blocks,
    n_heads=n_heads,
    max_seq_len=max_seq_len,
    vocab_size=len(tokenizer))).to(device)

print(model.args.vocab_size)

if torch.__version__ >= "2.0" and device == "cuda":
    model.compile()

if os.path.exists('./model.pt'): # load model if existing
    model.load_state_dict(torch.load("model.pt", map_location=device))

print(f"Model parameters: {(sum(param.numel() for param in model.parameters())):,}")

optimizer = torch.optim.AdamW(
    params=model.get_optimizer_grouped_parameters(weight_decay=weight_decay),
    betas=(beta1, beta2),
    eps=eps,
    lr=max_lr
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=train_iters * warmup_frac,
    num_training_steps=train_iters, num_cycles=0.5
)

train_data = BinDataset(chunk_size=max_seq_len, split="train", device=device)
val_data = BinDataset(chunk_size=max_seq_len, split="val", device=device)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, pin_memory=cuda) #shuffling would take ages
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, pin_memory=cuda)


@torch.no_grad()
def estimate_loss():
    model.eval()
    pbar = tqdm(total=eval_iters, desc="Evaluating")
    total_loss = 0.0
    val_iter = iter(val_loader)
    for step_idx in range(eval_iters):
        x, y = next(val_iter)

        x = x.to(device)
        y = y.to(device)

        with ctx:
            loss, _ = model(x, y)

        total_loss += loss.item()
        pbar.update(1)

    pbar.close()
    model.train()
    return total_loss / eval_iters # avg val loss


def train():
    torch.set_float32_matmul_precision('high')
    model.train()
    pbar = tqdm(total=train_iters, desc="Training")
    total_loss = 0.0

    train_iter = iter(train_loader)
    for step_idx in range(train_iters):
        x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        with ctx:
            loss, _ = model(x, y)

            optimizer.zero_grad()

            if cuda:
                scaler.scale(loss).backward()
                norm = clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                norm = clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        scheduler.step()

        step_loss = loss.item()
        total_loss += step_loss
        avg_loss = total_loss / (step_idx + 1)

        pbar.update(1)
        pbar.set_postfix(step_loss=f"{step_loss:.4f}", avg_loss=f"{avg_loss:.4f}")

        wandb.log({
            "train/loss": step_loss,
            "train/learning_rate": optimizer.param_groups[0]['lr'],
            "train/norm": norm
        })

        if (step_idx + 1) % eval_interval == 0:
            val_loss = estimate_loss()
            val_perplexity = torch.exp(torch.tensor(val_loss)).item()
            wandb.log({
                "val/loss": val_loss,
                "val/perplexity": val_perplexity
            })

    pbar.close()

if __name__ == "__main__":
    train()