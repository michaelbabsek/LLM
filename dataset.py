import os
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from tokenizer import Tokenizer

num_proc: int = 10
dataset_name: str = "openwebtext"
tokenizer = Tokenizer()

def _process(example):
        ids = tokenizer.encode(example["text"], add_bos=True, add_eos=True)
        out = {'ids': ids, 'len': len(ids)}
        return out

def _prepare():  # source: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
    dataset = load_dataset(dataset_name, num_proc=num_proc)

    split_dataset = dataset["train"].train_test_split(test_size=0.0005, shuffle=True)
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
        arr = np.memmap(filename, dtype=np.uint32, mode='w+',
                        shape=(arr_len,))  # int16 won't work since 2^16 < 100277 (vocab size)
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

class BinDataset(Dataset):
    def __init__(self, chunk_size: int, split: str, device = "cpu"):
        super().__init__()
        self.data = np.memmap(filename=f"./dataset/{split}.bin", dtype=np.uint32, mode="r")
        self.chunk_size = chunk_size
        self.device = device

    def __len__(self):
        return max(0, len(self.data) - self.chunk_size) -1

    def __getitem__(self, idx):
        x = self.data[idx      : idx + self.chunk_size    ].astype(np.int64, copy=False)
        y = self.data[idx + 1  : idx + self.chunk_size + 1].astype(np.int64, copy=False)
        y[-1] = tokenizer.pad_token_id # mask last token
        return torch.from_numpy(x), torch.from_numpy(y)

if __name__ == "__main__":
    _prepare()
