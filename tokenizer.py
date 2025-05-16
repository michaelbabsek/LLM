import tiktoken
import torch
from typing import List, Sequence, Union, Optional


class Tokenizer:
    def __init__(self, pattern: Optional[str] = None, model: str = "cl100k_base"):
        base_enc = tiktoken.get_encoding(model)

        self.pat_str = pattern or base_enc._pat_str
        mergeable_ranks = base_enc._mergeable_ranks

        extra_special_tokens = [
            "<|begin_of_text|>",     # BOS
            "<|end_of_text|>",       # EOS
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",            # end of turn
            "<|pad|>",               # PAD  ← NEU
        ]

        self.special_tokens = {
            tok: base_enc.n_vocab + i for i, tok in enumerate(extra_special_tokens)
        }

        self.pad_token: str = "<|pad|>"
        self.pad_id: int = self.special_tokens[self.pad_token]
        self.bos_token = "<|begin_of_text|>"
        self.eos_token = "<|end_of_text|>"

        self._enc = tiktoken.Encoding(
            name=f"{model}_custom",
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

    def __len__(self) -> int:
        return self._enc.n_vocab

    def encode(
        self,
        text: str,
        *,
        allowed_special: Union[str, set[str]] = "all",
        add_bos: bool = False,
        add_eos: bool = False,
        pad_to: Optional[int] = None,
        return_torch: bool = False,
        device: Optional[torch.device] = None,
    ) -> Union[List[int], torch.Tensor]:

        tokens: List[int] = []

        if add_bos:
            tokens.append(self.special_tokens[self.bos_token])

        tokens.extend(self._enc.encode(text, allowed_special=allowed_special))

        if add_eos:
            tokens.append(self.special_tokens[self.eos_token])

        if pad_to is not None:
            if len(tokens) > pad_to:
                tokens = tokens[:pad_to]          # hart abschneiden
            else:
                tokens.extend([self.pad_id] * (pad_to - len(tokens)))

        if return_torch:
            return torch.tensor(tokens, dtype=torch.long, device=device)

        return tokens

    def decode(
        self,
        tokens: Sequence[int],
        *,
        skip_special: bool = False,
        skip_pad: bool = True,
    ) -> str:
        if skip_special or skip_pad:
            filtered: List[int] = []
            for t in tokens:
                if skip_pad and t == self.pad_id:
                    continue
                if skip_special and t in self.special_tokens.values() and t != self.pad_id:
                    continue
                filtered.append(t)
            tokens = filtered

        return self._enc.decode(list(tokens))