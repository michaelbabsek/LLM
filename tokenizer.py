from typing import List, Sequence

import tiktoken

class Tokenizer:
    def __init__(self, model: str = "cl100k_base"): # use gpt-4 base model
        basemodel = tiktoken.get_encoding(model)

        # llama3 pat_str
        self.pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

        extra_special_tokens = [
            "<|begin_of_text|>",     # BOS
            "<|end_of_text|>",       # EOS
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|reserved_special_token_4|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>"            # end of turn
        ]

        self.special_tokens = {
            tok: basemodel.n_vocab + i for i, tok in enumerate(extra_special_tokens)
        }

        self.bos_token_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_token_id = self.special_tokens["<|end_of_text|>"]

        self.model = tiktoken.Encoding(
            name=f"{model}_custom",
            pat_str=self.pat_str,
            mergeable_ranks=basemodel._mergeable_ranks, # calling private attribute, probably not optimal
            special_tokens=self.special_tokens,
        )

    def __len__(self) -> int:
        return self.model.n_vocab


    def encode(self, s: str, add_bos: bool = False,add_eos: bool = False,) -> List[int]:
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_token_id)
        tokens.extend(self.model.encode(s))
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: Sequence[int], skip_special: bool = False) -> str:
        if skip_special:
            tokens = [tok for tok in tokens if tok not in self.special_tokens.values()]
        return self.model.decode(tokens)
