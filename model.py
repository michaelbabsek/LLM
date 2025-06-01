from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
from torch import nn

from config import ModelCfg


class MultiheadAttention(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.n_dim, cfg.n_dim * 3, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, S, _ = x.shape  # Batch, Sequence
        H = self.cfg.n_heads # number of heads
        D = self.cfg.n_dim  # model dim
        D_h = D // H  # head dim

        qkv = self.qkv(x)
        qkv = qkv.view(B, S, H, 3, D_h).permute(0, 2, 3, 1, 4) # (B, H, 3, S, D_h)

        q, k, v = qkv.unbind(dim=2)

        context = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=None, is_causal=True, dropout_p=self.cfg.dropout) # using flash attention
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        context = self.dropout(context)

        return context

class MLP(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.n_dim, cfg.n_dim * 4, bias=cfg.bias)
        self.fc2 = nn.Linear(cfg.n_dim * 4, cfg.n_dim, bias=cfg.bias)
        self.dropout = nn.Dropout(self.cfg.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.attn = MultiheadAttention(cfg)
        self.mlp = MLP(cfg)
        self.attn_norm = nn.RMSNorm(cfg.n_dim, eps=cfg.norm_eps)
        self.mlp_norm = nn.RMSNorm(cfg.n_dim, eps=cfg.norm_eps)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_dim)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.n_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.RMSNorm(cfg.n_dim, eps=cfg.norm_eps)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_blocks)])
        self.fc = nn.Linear(cfg.n_dim, cfg.vocab_size, bias=cfg.bias)

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

    def get_optimizer_grouped_parameters(self, weight_decay: float = 0.01) -> List[Dict]:
        decay, no_decay, seen = [], [], set()
        skip_modules = (nn.LayerNorm, nn.RMSNorm, nn.Embedding)

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if id(param) in seen or not param.requires_grad:
                    continue
                seen.add(id(param))

                if isinstance(module, skip_modules) or param.dim() == 1:
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




