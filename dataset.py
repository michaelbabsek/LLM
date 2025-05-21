import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from tokenizer import Tokenizer

num_proc: int = 8
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
        filename = os.path.join(os.path.dirname("./dataset"), f'{split}.bin')
        arr = np.memmap(filename, dtype=np.int32, mode='w+',
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


if __name__ == "__main__":
    _prepare()
