import os
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from tokenizer import Tokenizer

num_proc: int = 10
dataset_name: str = "Skylion007/openwebtext"
tokenizer = Tokenizer()

def _process(example):
        ids = tokenizer.encode(example["text"], add_bos=True, add_eos=True)
        out = {'ids': ids, 'len': len(ids)}
        return out

def _prepare():  # source: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
    dataset = load_dataset(dataset_name, split="train[:50%]", num_proc=num_proc)

    split_dataset = dataset.train_test_split(test_size=0.0005, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val

    tokenized = split_dataset.map(
        _process,
        remove_columns=["text"],
        desc="Tokenizing text",
        num_proc=num_proc
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        out_dir = "./dataset"
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"{split}.bin")
        arr = np.memmap(filename, dtype=np.uint16, mode='w+',
                        shape=(arr_len,))
        total_shards = 1024

        idx = 0
        for shard_idx in tqdm(range(total_shards), desc=f'writing {filename}'):
            # Batch together samples for faster write
            shard = dset.shard(num_shards=total_shards, index=shard_idx, contiguous=True).with_format('numpy')
            arr_shard = np.concatenate(shard['ids'])
            # Write into mmap
            arr[idx: idx + len(arr_shard)] = arr_shard
            idx += len(arr_shard)
        arr.flush()

class BinDataset:
    """Draws random windows directly from the memory-mapped token file (no index permutation, no caching)."""

    def __init__(self, split: str, seq_len: int, data_dir: str = "./dataset", device: str = "cpu"):
        self.path = os.path.join(data_dir, f"{split}.bin")
        self.seq_len = seq_len
        self.device = device
        self.n_tokens = len(np.memmap(self.path, dtype=np.uint16, mode="r"))

    def get_batch(self, batch_size: int, generator: torch.Generator):
        # recreate the memmap every batch to avoid a memory leak, see nanoGPT train.py
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        ix = torch.randint(self.n_tokens - self.seq_len, (batch_size,), generator=generator).tolist()
        x = torch.from_numpy(np.stack([data[i: i + self.seq_len] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1: i + 1 + self.seq_len] for i in ix]).astype(np.int64))  # shifted by one token
        if self.device == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)

if __name__ == "__main__":
    _prepare()
