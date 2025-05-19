import contextlib
import os

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Transformer, ModelArgs
from tokenizer import Tokenizer
from wiki_dataset import WikiDataset

#model args
n_dim: int = 768
n_blocks: int = 4
n_heads: int = 4
max_seq_len: int = 1024

# training
n_epochs = 5
batch_size: int = 4
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

wiki_dataset = WikiDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len)

data_loader = DataLoader(
    wiki_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=(device.startswith("cuda")))

def train():
    torch.set_float32_matmul_precision('high')

    model = Transformer(args=args).to(device)

    if os.path.exists('./model.pt'):
        model.load_state_dict(torch.load("model.pt", map_location=device))

    if device != 'mps':
        model.compile()

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

    total_steps = n_epochs * len(data_loader)
    warmup_steps = int(total_steps * warmup_steps_percentage)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=min_lr,
    )

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{n_epochs}")
        for idx, batch in enumerate(pbar):
            batch = batch.to(device)
            input_ids  = batch[:, :-1]
            target_ids = batch[:, 1:]

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

            epoch_loss += step_loss
            avg_loss = epoch_loss / (idx + 1)

            pbar.set_postfix({
                "step_loss": f"{step_loss:.4f}",
                "avg_loss":  f"{avg_loss:.4f}",
            })

        torch.save(model.state_dict(), f"model.pt") # saving the model after each epoch

        # generate sample
        model.eval()
        input = tokenizer.encode("Hallo", add_bos=True)
        output = model.generate(input, device=device, max_token_length=max_token_len)
        print({f"Epoch {epoch+1}": tokenizer.decode(output)})

if __name__ == "__main__":
    train()