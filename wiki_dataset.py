import os
from itertools import chain

import datasets
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

from tokenizer import Tokenizer


def _chunk_full(flat_examples, seq_len):
    end = len(flat_examples) // seq_len * seq_len
    return [flat_examples[i: i + seq_len] for i in range(0, end, seq_len)]


class WikiDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, max_seq_len: int, cache_path: str = "dataset_cache"):
        super().__init__()
        self.examples = None
        self.tokenizer = tokenizer
        self.cache_path = cache_path
        self.max_seq_len = max_seq_len

        if os.path.exists(cache_path):
            # load dataset from cache
            loaded = load_from_disk(cache_path)['text']
            self._process_examples(loaded)

        else:
            # load dataset and process it
            raw_dataset = load_dataset(
                path="wikipedia",
                name="20220301.de",
                split="train[:10%]" #loading 10% of the dataset
            ).remove_columns(["id", "url", "title"]) # only use contents

            text_encoded = raw_dataset.map(self._encode_text)
            text_encoded.save_to_disk(self.cache_path)
            self._process_examples(text_encoded['text'])


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def _encode_text(self, example):
        return {'text': self.tokenizer.encode(example["text"], add_bos=True, add_eos=True)}

    def _process_examples(self, examples):
        flat = list(chain.from_iterable(examples))
        self.examples = torch.tensor(_chunk_full(flat, self.max_seq_len))


