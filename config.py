from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class RunCfg:
    project: str = 'LLM'
    name: str | None = None
    seed: int = 42
    log_interval: int = 10


@dataclass
class ModelCfg:
    n_dim: int = 96
    n_blocks: int = 4
    n_heads: int = 4
    max_seq_len: int = 256
    vocab_size: int = -1  # later defined by tokenizer
    norm_eps: float = 1e-5
    dropout: float = 0.1
    bias: bool = False


@dataclass
class TrainCfg:
    train_iters: int = 10_000
    eval_iters: int = 200
    eval_batch_size: int = 16
    eval_seed: int = 1337  # fixed, so every evaluation sees the same validation windows
    eval_interval: int = 1000
    warmup_iters: int = 1000
    batch_size: int = 16
    grad_clip: float = 1.0
    grad_accum_steps: int = 2
    use_checkpoint: bool = True
    ckpt_path: str = 'ckpt.pt'
    data_dir: str = './dataset'


@dataclass
class OptimCfg:
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass
class Config:
    run: RunCfg
    model: ModelCfg
    training: TrainCfg
    optim: OptimCfg


def load_cfg(path: str | Path = 'config.yml') -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(
        run=RunCfg(**raw['run']),
        model=ModelCfg(**raw['model']),
        training=TrainCfg(**raw['training']),
        optim=OptimCfg(**raw['optim']),
    )
