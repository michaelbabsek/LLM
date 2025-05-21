import contextlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Transformer, ModelArgs
from tokenizer import Tokenizer

#model args
n_dim: int = 768
n_blocks: int = 16
n_heads: int = 8
max_seq_len: int = 1024

# training
total_iters = 60_000
saving_interval = 1_000
batch_size: int = 1
max_lr = 6e-4
min_lr = 1e-6
warmup_steps_percentage = 0.1

#generation
max_token_len = max_seq_len

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_device()

tokenizer = Tokenizer()

args = ModelArgs(n_dim=n_dim, n_blocks=n_blocks, n_heads=n_heads, max_seq_len=max_seq_len, vocab_size=len(tokenizer))

def get_batch(split): # source: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint32, mode='r')
    else:
        data = np.memmap('val.bin', dtype=np.uint32, mode='r')

    ix = torch.randint(len(data) - max_seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+max_seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+max_seq_len]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def train():
    torch.set_float32_matmul_precision('high')

    model = Transformer(args=args).to(device)

    if os.path.exists('./model.pt'):
        model.load_state_dict(torch.load("model.pt", map_location=device))

    if device != 'mps':
        model.compile()

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

    warmup_steps = int(total_iters * warmup_steps_percentage)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_iters - warmup_steps,
        eta_min=min_lr,
    )

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    model.train()

    pbar = tqdm(total=total_iters, desc=f"Training Step")

    for step_idx in range(1, total_iters):
        loss_sum = 0.0
        input_ids, target_ids = get_batch('train')

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        else:
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

        optimizer.zero_grad()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        step_loss = loss.item()
        loss_sum += step_loss
        avg_loss = loss_sum / step_idx

        pbar.update(1)

        pbar.set_postfix({
            "step_loss": f"{step_loss:.4f}",
            "avg_loss": f"{avg_loss:.4f}"
        })

        if step_idx % saving_interval == 0:
            model.eval()
            input = tokenizer.encode("Hallo", add_bos=True)
            output = model.generate(input, device=device, max_token_length=max_token_len)
            print(tokenizer.decode(output))
            model.train()

            torch.save(model.state_dict(), f"model.pt")  # saving the model N steps

if __name__ == "__main__":
    train()