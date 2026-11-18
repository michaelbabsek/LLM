import math
from unittest import TestCase

import torch

from config import ModelCfg
from model import Transformer


class ModelTests(TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = ModelCfg(n_dim=64, n_blocks=8, n_heads=4, max_seq_len=32, vocab_size=512, dropout=0.0)
        self.model = Transformer(self.cfg)

    def test_weight_tying_survives_init(self):
        self.assertIs(self.model.fc.weight, self.model.tok.weight)

    def test_residual_projection_init(self):
        residual_std = 0.02 / math.sqrt(2 * self.cfg.n_blocks)
        for block in self.model.blocks:
            self.assertAlmostEqual(block.attn.proj.weight.std().item(), residual_std, delta=residual_std * 0.1)
            self.assertAlmostEqual(block.mlp.fc2.weight.std().item(), residual_std, delta=residual_std * 0.1)
            self.assertAlmostEqual(block.mlp.fc1.weight.std().item(), 0.02, delta=0.002)
        self.assertAlmostEqual(self.model.tok.weight.std().item(), 0.02, delta=0.002)

    def test_forward_shapes(self):
        x = torch.randint(0, self.cfg.vocab_size, (2, 16))
        loss, logits = self.model(x, x)
        self.assertEqual(logits.shape, (2, 16, self.cfg.vocab_size))
        self.assertTrue(torch.isfinite(loss))

    def test_weight_decay_groups(self):
        decay, no_decay = self.model.get_optimizer_grouped_parameters(0.1)
        decay_ids = [id(p) for p in decay["params"]]
        no_decay_ids = [id(p) for p in no_decay["params"]]
        all_ids = decay_ids + no_decay_ids

        self.assertEqual(len(all_ids), len(set(all_ids)))  # every tensor once, the tied matrix included
        self.assertEqual(set(all_ids), {id(p) for p in self.model.parameters()})
        self.assertIn(id(self.model.tok.weight), decay_ids)
        self.assertIn(id(self.model.pos.weight), decay_ids)
        self.assertIn(id(self.model.norm.weight), no_decay_ids)
        self.assertTrue(all(p.dim() == 1 for p in no_decay["params"]))
