import os

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Transformer, ModelArgs
from tokenizer import Tokenizer
from wiki_dataset import WikiDataset


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_device()

tokenizer = Tokenizer()

args = ModelArgs(n_dim=768, n_blocks=4, n_heads=4, max_seq_len=1024, vocab_size=len(tokenizer))

wiki_dataset = WikiDataset(tokenizer=tokenizer, max_seq_len=args.max_seq_len)

data_loader = DataLoader(
    wiki_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=(device.startswith("cuda")))

def train(num_epochs: int, lr: float):
    torch.set_float32_matmul_precision('high')

    model = Transformer(args=args).to(device)

    if os.path.exists('./model.pt'):
        model.load_state_dict(torch.load("model.pt", map_location=device))


    model.compile()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = num_epochs * len(data_loader)
    warmup_steps = int(total_steps * 0.1)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler(device)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for idx, batch in enumerate(pbar):
            batch = batch.to(device)
            input_ids  = batch[:, :-1]
            target_ids = batch[:, 1:]

            with torch.amp.autocast('cuda'):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
        input = tokenizer.encode("Hallo", add_bos=True)
        output = model.generate(input, device=device, max_token_length=100)
        print({f"Epoch {epoch+1}": tokenizer.decode(output)})

if __name__ == "__main__":
    steps_per_epoch = len(data_loader)
    train(num_epochs=1, lr=6e-4)