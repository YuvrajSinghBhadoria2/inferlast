"""Tests for the KV-cache decode module (decode.py) and its auto-detection."""

import torch
import torch.nn as nn

from conftest import make_mini_lm
from decode import (
    kv_supported,
    new_cache,
    generate_with_kv,
    measure_decode,
)
from quantize import _collect_logits
from auto_optimizer import run_auto_optimizer


class KVCompatible(nn.Module):
    """A causal LM whose forward accepts KV-cache kwargs (so the decode loop /
    auto-detection paths are exercised) but recomputes each step internally --
    here the correctness of the loop plumbing is what matters."""

    def __init__(self, vocab=32):
        super().__init__()
        self.inner = make_mini_lm(vocab=vocab)

    @property
    def embed_tokens(self):
        return self.inner.embed_tokens

    @property
    def blocks(self):
        return self.inner.blocks

    @property
    def lm_head(self):
        return self.inner.lm_head

    def forward(self, ids, attention_mask=None, past_key_values=None,
                use_cache=False, position_ids=None, **kwargs):
        x = self.embed_tokens(ids)
        for b in self.blocks:
            x = b(x)
        logits = self.lm_head(x)
        return type("Out", (), {"logits": logits})()


def _kv_model(vocab=32):
    return KVCompatible(vocab=vocab)


def test_kv_supported_detects_hf_style_forward():
    assert kv_supported(_kv_model()) is True
    assert kv_supported(make_mini_lm()) is False


def test_new_cache_returns_dynamic_cache():
    from transformers import DynamicCache
    assert isinstance(new_cache(), DynamicCache)


def test_generate_with_kv_kv_equals_recompute():
    m = _kv_model()
    ids = torch.tensor([[0, 1, 2, 3]])
    kv = generate_with_kv(m, ids, n_new=5, use_cache=True)
    nc = generate_with_kv(m, ids, n_new=5, use_cache=False)
    assert kv.shape == (1, 5)
    assert nc.shape == (1, 5)
    assert torch.equal(kv, nc), "KV and recompute paths must match"


def test_measure_decode_returns_positive_rates():
    m = _kv_model()
    ids = torch.tensor([[0, 1, 2, 3]])
    mpt_kv, tps_kv = measure_decode(m, ids, n_new=5, use_cache=True,
                                    repeats=2, warmup=1)
    mpt_nc, tps_nc = measure_decode(m, ids, n_new=5, use_cache=False,
                                    repeats=2, warmup=1)
    assert mpt_kv > 0 and tps_kv > 0
    assert mpt_nc > 0 and tps_nc > 0


def test_collect_logits_auto_detects_kv_support():
    tok = None

    class _T:
        def __call__(self, text, **kwargs):
            return type("_", (), {"input_ids": torch.tensor([[0, 1, 2]])})()

    # supported model -> use_cache auto-True, no crash, returns token count
    mpt1, logits1, _ = _collect_logits(_kv_model(), _T(), "x", 4)
    assert logits1.shape[0] == 4
    assert mpt1 > 0
    # unsupported model -> auto-False recompute path, no crash
    mpt2, logits2, _ = _collect_logits(make_mini_lm(), _T(), "x", 4)
    assert logits2.shape[0] == 4
    assert mpt2 > 0


class _FakeTok:
    def __call__(self, prompt, return_tensors="pt", **kwargs):
        return type("_", (), {"input_ids": torch.tensor([[0, 1, 2]])})()


def test_run_auto_optimizer_exposes_kv_technique_when_supported():
    from auto_optimizer import run_auto_optimizer
    out, report = run_auto_optimizer(
        _kv_model(), _FakeTok(), prompt="x", n_new=4, batches=(1, 2))
    kv = out["techniques"]["kv_cache"]
    assert kv["avail"] is True
    assert kv["kv_cache_ms_per_token"] > 0
    assert kv["no_cache_ms_per_token"] > 0
    assert kv["kv_speedup_x"] > 0

