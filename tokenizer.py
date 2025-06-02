import tiktoken
from typing import List, Sequence

class Tokenizer:
    def __init__(self, model: str = "gpt2"):
        self._base_name = model
        basemodel = tiktoken.get_encoding(model)

        extra_special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|reserved_special_token_4|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|pad_id|>"
        ]

        self.special_tokens = {
            tok: basemodel.n_vocab + i for i, tok in enumerate(extra_special_tokens)
        }

        self.bos_token_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_token_id = self.special_tokens["<|end_of_text|>"]
        self.pad_token_id = self.special_tokens["<|pad_id|>"]

        self.model = tiktoken.Encoding(
            name=f"{model}_custom",
            pat_str=basemodel._pat_str,
            mergeable_ranks=basemodel._mergeable_ranks,
            special_tokens=self.special_tokens,
        )

    def __len__(self) -> int:
        return self.model.n_vocab

    def encode(self, s: str, *, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_token_id)
        tokens.extend(self.model.encode(s))
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: Sequence[int], *, skip_special: bool = False) -> str:
        if skip_special:
            tokens = [t for t in tokens if t not in self.special_tokens.values()]
        return self.model.decode(tokens)

    def __getstate__(self):
        return {"model": self._base_name, "special_tokens": self.special_tokens}

    def __setstate__(self, state):
        self.__init__(state["model"])