"""Tests for the auto-quantizer: robust quality metric + honest decision rule."""

import torch
import pytest

from conftest import make_mini_lm, make_dummy_logits
from quantize import _logit_quality, _decide, auto_quantize
import torch.nn as nn


# --- the robust quality metric (replaces brittle greedy-token identity) -----

def test_logit_quality_identical_logits_is_one():
    logits = make_dummy_logits(n_tokens=5, vocab=16, seed=3)
    cos, top5 = _logit_quality(logits, logits)
    assert cos == pytest.approx(1.0, abs=1e-4)
    assert top5 == pytest.approx(1.0)


def test_logit_quality_orthogonal_is_zero_cosine():
    a = make_dummy_logits(n_tokens=5, vocab=16, seed=0)
    b = make_dummy_logits(n_tokens=5, vocab=16, seed=1)
    # random independent logits -> near-zero mean cosine, small top-5 overlap
    cos, top5 = _logit_quality(a, b)
    assert cos < 0.5
    assert top5 < 1.0


def test_logit_quality_mild_perturbation_keeps_cosine_high_top5():
    # perturbing a fraction keeps signal: cosine high but ranking may shift
    a = make_dummy_logits(n_tokens=8, vocab=32, seed=7)
    b = a.clone()
    b[:, 0] += 5.0  # coat one slot
    cos, top5 = _logit_quality(a, b)
    assert cos > 0.7
    # top-5 overlap still high since only low-rank slots changed
    assert top5 > 0.6


# --- the honest decision rule (pure, no clock) ------------------------------

_QUAL = (0.70, 0.60)  # (quality_floor, top5_floor)
_SPEED = 1.05


@pytest.mark.parametrize("speedup,logit_cos,top5,expected", [
    # fast AND quality preserved -> recommend
    (1.30, 0.80, 0.90, True),
    # fast but quality destroyed -> refuse (the audit case)
    (1.30, 0.70, 0.10, False),
    (1.30, 0.30, 0.27, False),
    # quality preserved but NO real speedup -> refuse (overhead-bound)
    (1.01, 0.90, 0.95, False),
    (0.98, 0.80, 0.80, False),
    # quality barely above cos-floor but top5 below -> refuse
    (1.30, 0.75, 0.40, False),
])
def test_decide_honest_refusal(speedup, logit_cos, top5, expected):
    _, _, rec, _ = _decide(speedup, 0.8 * speedup, logit_cos, top5,
                            _SPEED, _QUAL[0], _QUAL[1])
    assert rec is expected, f"speedup={speedup} cos={logit_cos} top5={top5}"


def test_decide_recommends_only_when_both_floors_pass():
    ok_spd, ok_qual, rec, _ = _decide(1.30, 1.10, 0.80, 0.70, _SPEED, 0.65, 0.60)
    assert ok_spd and ok_qual and rec is True


# --- smoke: the full measured path runs and returns a well-formed verdict -----

class FakeTok:
    def __init__(self, vocab=32):
        self.vocab = vocab

    def __call__(self, prompt, return_tensors="pt"):
        # [1,3] fixed ids so a tiny forward works
        ids = torch.tensor([[0, 1, 2]])
        return type("Batch", (), {"input_ids": ids})()


def test_auto_quantize_runs_and_returns_verdict():
    model = make_mini_lm(vocab=32)
    tok = FakeTok(vocab=32)
    v = auto_quantize(model, tok, prompt="x", n_new=3, repeats=2)
    # structural invariants of the verdict
    assert v.recommended is True or v.recommended is False
    assert 0.0 <= v.logit_cosine <= 1.0
    assert 0.0 <= v.top5_overlap <= 1.0
    assert v.speedup > 0.0
    assert v.speedup_min <= v.speedup + 1e-6
    assert v.speedup_min > 0.0
    assert v.reason
    # the quality fields are always populated (never stale default)
    assert v.before["ms_spread"][1] >= v.before["ms_spread"][0]