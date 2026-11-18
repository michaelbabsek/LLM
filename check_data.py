"""Sanity check for the tokenized dataset files: size, token range, special tokens and decoded samples."""
import argparse
import os

import numpy as np

from tokenizer import Tokenizer

CHUNK = 100_000_000  # tokens per chunk, keeps memory use at ~200 MB while scanning train.bin


def check_split(path: str, tokenizer: Tokenizer, vocab_size: int, rng: np.random.Generator, n_samples: int, sample_len: int):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(data)
    max_id, n_bos, n_eos = 0, 0, 0
    for i in range(0, n, CHUNK):
        chunk = np.asarray(data[i: i + CHUNK])
        max_id = max(max_id, int(chunk.max()))
        n_bos += int(np.count_nonzero(chunk == tokenizer.bos_token_id))
        n_eos += int(np.count_nonzero(chunk == tokenizer.eos_token_id))

    print(f"=== {path}")
    print(f"tokens:      {n:,}")
    print(f"max token id: {max_id} (tokenizer: {len(tokenizer)}, model vocab_size: {vocab_size}) "
          f"-> {'OK' if max_id < len(tokenizer) else 'ERROR: id out of range'}")
    print(f"BOS tokens:  {n_bos:,} ({n_bos / n:.4%})")
    print(f"EOS tokens:  {n_eos:,} ({n_eos / n:.4%})")
    print(f"avg document length: {n / max(n_bos, 1):.0f} tokens")

    for offset in rng.integers(0, n - sample_len, size=n_samples):
        text = tokenizer.decode(data[offset: offset + sample_len].tolist())
        print(f"\n--- offset {offset:,}\n{text}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./dataset")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--sample-len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = Tokenizer()
    rng = np.random.default_rng(args.seed)
    for split in ("train", "val"):
        check_split(os.path.join(args.data_dir, f"{split}.bin"), tokenizer, tokenizer.padded_vocab_size(),
                    rng, args.samples, args.sample_len)
