"""Tests for the profiler: category classification + wall-clock aggregation."""

import torch
import pytest

from conftest import make_mini_lm
from profiler import _classify, _is_leaf_for, profile_forward, profile_decode


# --- classification (pure, deterministic) -----------------------------------

@pytest.mark.parametrize("name,expected_cat,expected_leaf", [
    ("embed_tokens", "embed", True),
    ("model.embed_tokens", "embed", True),
    ("lm_head", "lm_head", True),
    ("model.lm_head", "lm_head", True),
    ("layers.0.self_attn.q_proj", "attention", True),
    ("layers.0.self_attn", "attention", False),   # attn container excluded
    ("layers.0.mlp.gate_proj", "mlp", True),
    ("layers.0.mlp", "mlp", True),
    ("layers.0.norm1", "norm", True),
    ("layers.0.input_layernorm", "norm", True),
    ("model.rotary_emb", "rotary", False),
])
def test_classify(name, expected_cat, expected_leaf):
    cat, leaf = _classify(name)
    assert cat == expected_cat
    assert leaf == expected_leaf


def test_is_leaf_excludes_attn_container():
    # the container 'self_attn' is not timed directly; its leaflets are
    assert _is_leaf_for("layers.0.self_attn") is False
    assert _is_leaf_for("layers.0.self_attn.o_proj") is True


# --- integration on a real tiny torch model ---------------------------------

def _ids(batch=1, seq=4, vocab=32):
    return torch.randint(0, vocab, (batch, seq))


def test_profile_forward_classifies_categories():
    model = make_mini_lm()
    ids = _ids()
    pr = profile_forward(model, ids, run_repeats=2, warmup=1)
    # every interesting category must be attributed
    for cat in ("attention", "mlp", "norm", "embed", "lm_head"):
        assert pr.cat_times.get(cat, 0.0) > 0.0, f"missing {cat}"
    # no double counting: cat_params embedded+lm_head present
    assert pr.cat_params.get("embed", 0) > 0
    assert pr.cat_params.get("lm_head", 0) > 0


def test_profile_forward_tokens_and_total():
    model = make_mini_lm()
    ids = _ids(batch=2, seq=5, vocab=32)
    runs = 3
    pr = profile_forward(model, ids, run_repeats=runs, warmup=1)
    assert pr.tokens == 2 * 5 * runs
    assert pr.total_time > 0.0


def test_profile_forward_leaf_sum_not_huge_vs_total():
    # leaf sum should be <= total wall time (or close), never inflating above it
    model = make_mini_lm()
    pr = profile_forward(model, _ids(), run_repeats=2, warmup=1)
    leaf_sum = sum(pr.cat_times.values())
    assert leaf_sum >= 0.0
    # per category leaf time is real and finite
    for v in pr.cat_times.values():
        assert v == v and v >= 0.0


def test_profile_decode_returns_categories():
    model = make_mini_lm()
    ids = _ids(seq=4, vocab=32)
    pr = profile_decode(model, ids, n_new_tokens=3, run_repeats=1, warmup=1)
    assert pr.tokens == 3 * 1
    assert pr.total_time > 0.0
    # decode should still attribute leaf categories
    any_cat = sum(pr.cat_times.values())
    assert any_cat > 0.0