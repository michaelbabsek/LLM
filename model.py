import math

import torch
from torch import nn
from dataclasses import dataclass
from tokenizer import Tokenizer
import torch.nn.functional as F

@dataclass
class ModelArgs:
    n_dim: int = 768
    n_blocks: int = 4
    n_heads: int = 4
    max_seq_len: int = 256
    vocab_size: int = -1 # later defined by tokenizer

class MultiheadAttention(nn.Module):
    def __init__(self, args: ModelArgs, use_mask: bool = True):
        super().__init__()
        self.args = args
        self.use_mask = use_mask
        self.qkv = nn.Linear(args.n_dim, args.n_dim * 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, S, _ = x.shape  # Batch, Sequence
        H = self.args.n_heads # number of heads
        D = self.args.n_dim  # model dim
        D_h = D // H  # head dim

        qkv = self.qkv(x)
        qkv = qkv.view(B, S, H, 3, D_h).permute(0, 2, 3, 1, 4) # (B, H, 3, S, D_h)

        q, k, v = qkv.unbind(dim=2)

        # Scaled-Dot-Product Attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D_h)  # (B, H, S, S)

        # mask
        if self.use_mask:
            causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        context = weights @ v  # (B, H, S, D_h)
        context = context.transpose(1, 2).contiguous().view(B, S, D)

        return context

class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(args.n_dim, args.n_dim * 4)
        self.fc2 = nn.Linear(args.n_dim * 4, args.n_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.attn = MultiheadAttention(args)
        self.mlp = MLP(args)
        self.norm1 = nn.LayerNorm(args.n_dim)
        self.norm2 = nn.LayerNorm(args.n_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok = nn.Embedding(args.vocab_size, args.n_dim)
        self.pos = nn.Embedding(args.max_seq_len, args.n_dim)
        self.norm = nn.LayerNorm(args.n_dim)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.n_blocks)])
        self.fc = nn.Linear(args.n_dim, args.vocab_size)

    def forward(self, x):
        B, S = x.shape
        x = self.tok(x)  # (B, S, n_dim)
        x = x + self.pos(torch.arange(S, device=x.device))
        x = self.norm(x)

        for block in self.blocks:
            x = block(x)

        x = self.fc(x)
        return x


