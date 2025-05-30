from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from config import ModelCfg


class MultiheadAttention(nn.Module):
    def __init__(self, args: ModelCfg):
        super().__init__()
        self.args = args
        self.qkv = nn.Linear(args.n_dim, args.n_dim * 3)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        B, S, _ = x.shape  # Batch, Sequence
        H = self.args.n_heads # number of heads
        D = self.args.n_dim  # model dim
        D_h = D // H  # head dim

        qkv = self.qkv(x)
        qkv = qkv.view(B, S, H, 3, D_h).permute(0, 2, 3, 1, 4) # (B, H, 3, S, D_h)

        q, k, v = qkv.unbind(dim=2)

        context = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=None, is_causal=True, dropout_p=self.args.dropout) # using flash attention
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        context = self.dropout(context)

        return context

class MLP(nn.Module):
    def __init__(self, args: ModelCfg):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(args.n_dim, args.n_dim * 4)
        self.fc2 = nn.Linear(args.n_dim * 4, args.n_dim)
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, args: ModelCfg):
        super().__init__()
        self.args = args
        self.attn = MultiheadAttention(args)
        self.mlp = MLP(args)
        self.attn_norm = nn.RMSNorm(args.n_dim, eps=args.norm_eps)
        self.mlp_norm = nn.RMSNorm(args.n_dim, eps=args.norm_eps)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelCfg):
        super().__init__()
        self.args = args
        self.tok = nn.Embedding(args.vocab_size, args.n_dim)
        self.pos = nn.Embedding(args.max_seq_len, args.n_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.norm = nn.RMSNorm(args.n_dim, eps=args.norm_eps)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.n_blocks)])
        self.fc = nn.Linear(args.n_dim, args.vocab_size, bias=False)

        self.fc.weight = self.tok.weight # weight tying significantly reduces number of parameters and improves performance
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        B, S = x.shape
        h = self.tok(x)  # (B, S, D)
        h = h + self.pos(torch.arange(S, device=x.device))
        h = self.dropout(h)

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        logits = self.fc(h)  # (B, S, V)

        if y is not None:
            logits = logits[:, :-1, :].contiguous()
            y = y[:, 1:].contiguous()

            loss = F.cross_entropy(
                input=logits.view(-1, logits.size(-1)),
                target=y.view(-1),
                label_smoothing=0.1,
            )
            return loss, logits

        return logits

    def get_optimizer_grouped_parameters(self, weight_decay=0.01):
        decay = []
        no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if any(nd in name.lower() for nd in ['bias', 'norm']):
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def generate(self, tokens: List[int], max_token_length: int, device):
        input_tensor = torch.tensor(tokens, device=device).unsqueeze(0)
        output = tokens.copy()

        for _ in range(max_token_length - len(tokens)):
            logits = self.forward(input_tensor)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.argmax(probs, dim=-1).item()
            output.append(next_token)

            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)

        return output




