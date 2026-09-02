"""Shared test fixtures: tiny deterministic real torch models (no downloads)."""

import os
import sys

import torch
import torch.nn as nn

SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


class MiniBlock(nn.Module):
    """Tiny deterministic block keyed on real HF module names the profiler uses:
    self_attn.q_proj etc. + mlp + norm, so categorisation and leaf-timing are
    exercised exactly as on a real model (self_attn container excludes its own
    span; the projections are the timed leaves)."""

    def __init__(self, dim=8, heads=2):
        super().__init__()
        class SelfAttn(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.q_proj = nn.Linear(dim, dim)
                self.k_proj = nn.Linear(dim, dim)
                self.v_proj = nn.Linear(dim, dim)
                self.o_proj = nn.Linear(dim, dim)
        self.self_attn = SelfAttn(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.register_buffer("base", torch.arange(dim).float() + 1.0)

    def forward(self, x):
        # a small deterministic attention-ish pass over every projection
        q = self.self_attn.q_proj(x)
        k = self.self_attn.k_proj(x)
        v = self.self_attn.v_proj(x)
        attn = (q + k + v) * (1.0 / 3.0)
        x = self.self_attn.o_proj(attn) + self.base
        x = self.norm1(x) * 2.0
        x = self.mlp(x)
        x = self.norm2(x)
        return x


class MiniLM(nn.Module):
    """Mini causal-ish LM: embed + blocks + lm_head. Deterministic."""
    def __init__(self, vocab=32, dim=8, heads=2, n_blocks=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([MiniBlock(dim, heads) for _ in range(n_blocks)])
        self.lm_head = nn.Linear(dim, vocab)

    def forward(self, ids):
        x = self.embed_tokens(ids)
        for b in self.blocks:
            x = b(x)
        logits = self.lm_head(x)
        return type("Out", (), {"logits": logits})()


def make_mini_lm(vocab=32, dim=8, heads=2, n_blocks=2):
    return MiniLM(vocab, dim, heads, n_blocks)


def make_dummy_logits(n_tokens, vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((n_tokens, vocab), generator=g)