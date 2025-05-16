import logging
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from pyarrow import Table
from tqdm import tqdm
from model import Transformer, ModelArgs
from tokenizer import Tokenizer
import logging

CACHE_PATH = "./wiki_chunks.pt"

tokenizer = Tokenizer()

args = ModelArgs(
    n_dim=768,
    n_blocks=4,
    n_heads=8,
    max_seq_len=256,
    vocab_size=len(tokenizer),
)

# Define custom dataset for DataLoader
class WikiDataset(Dataset):
    def __init__(self, examples):
        super().__init__()
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def build_or_load_dataset(texts, tokenizer, max_seq_len):
    if os.path.exists(CACHE_PATH):
        examples = torch.load(CACHE_PATH)
    else:
        flat_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_eos=True)
            flat_tokens.extend(tokens)

        examples = []
        for i in range(0, len(flat_tokens), max_seq_len):
            chunk = flat_tokens[i : i + max_seq_len]
            if len(chunk) == max_seq_len:
                examples.append(torch.tensor(chunk, dtype=torch.long))

        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        torch.save(examples, CACHE_PATH)

    return WikiDataset(examples)

# Create dataset and DataLoader
raw = load_dataset("wikipedia", "20220301.de", split="train[:1%]") # only use 1% of the dataset for testing purposes
wiki_texts = [item["text"] for item in raw]

wiki_dataset = build_or_load_dataset(
    wiki_texts,
    tokenizer,
    args.max_seq_len
)

data_loader = DataLoader(
    wiki_dataset,
    batch_size=8,
    shuffle=True,
    drop_last=True
)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def train(num_epochs: int, lr: float):
    device = get_device()
    model = Transformer(args=args).to(device)
    if os.path.exists('./model.pt'):
        model.load_state_dict(torch.load("model.pt", map_location=device))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for idx, batch in enumerate(pbar):
            batch = batch.to(device)
            input_ids  = batch[:, :-1]
            target_ids = batch[:, 1:]

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            epoch_loss += step_loss
            avg_loss = epoch_loss / (idx + 1)

            pbar.set_postfix({
                "step_loss": f"{step_loss:.4f}",
                "avg_loss":  f"{avg_loss:.4f}"
            })

            if idx % 100 == 0:
                torch.save(model.state_dict(), f"model.pt")

if __name__ == "__main__":
    train(num_epochs=1, lr=3e-4)