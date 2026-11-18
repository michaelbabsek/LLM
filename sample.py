"""Generates continuations for a fixed set of prompts from a checkpoint and writes them to a markdown file.

    python sample.py --ckpt ./checkpoints/gpu/best.pt
"""
import argparse
import time

import torch

from config import ModelCfg, load_cfg
from model import Transformer
from tokenizer import Tokenizer

PROMPTS = [
    "The history of the Roman Empire",
    "Scientists have discovered a new species of",
    "In a statement released on Monday, the company said",
    "The best way to learn a new programming language is",
    "Once upon a time, in a small village,",
]


def load_model(ckpt_path: str, config_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if ckpt.get("config") is not None:
        model_cfg = ModelCfg(**ckpt["config"]["model"])
    else:  # checkpoints saved before the config was stored
        model_cfg = load_cfg(config_path).model
        model_cfg.vocab_size = Tokenizer().padded_vocab_size()
    model = Transformer(model_cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="config.yml", help="only used if the checkpoint has no stored config")
    parser.add_argument("--out", default="samples.md")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    tokenizer = Tokenizer()
    model, ckpt = load_model(args.ckpt, args.config, device)

    lines = [
        "# Samples",
        "",
        f"- checkpoint: `{args.ckpt}` (step {ckpt['step']}, best val loss {ckpt.get('best_val_loss', ckpt.get('loss')):.4f})",
        f"- temperature {args.temperature}, top-k {args.top_k}, max {args.max_new_tokens} new tokens, seed {args.seed}",
        f"- generated {time.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    for prompt in PROMPTS:
        tokens = tokenizer.encode(prompt, add_bos=True)
        out = model.generate(tokens, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                             eos_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(out[len(tokens):], skip_special=True)
        print(f"=== {prompt}{completion}\n")
        lines += [f"## {prompt}", "", f"**{prompt}**{completion}", ""]

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}")
