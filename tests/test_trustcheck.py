"""Tests for trustcheck: does a measured 'win' actually hold up?"""

import os
import tempfile

import torch
import pytest

from conftest import make_dummy_logits
from trustcheck import (
    classify_speedup, speedup_ci, flag_brittle_metric, robust_logit_quality,
    flag_read_by_nothing, audit, TrustVerdict,
)


# --- single-run noise: the 3.0x-vs-0.65x flip is NOT a reliable win ---------

def test_the_real_flip_is_not_a_trustworthy_win():
    # Two STRICTLY REAL measurements of the SAME config (recorded in this repo's
    # evidence): standalone saw INT8 as 3.0x faster, endtoend saw 0.65x slower.
    controls = [1180.0, 1316.0]      # fp32 ms/token, two sessions
    treatments = [390.0, 2031.0]     # int8 ms/token, two sessions
    lo, hi = speedup_ci(controls, treatments)
    verdict, _ = classify_speedup(lo, hi)
    # With only 2 paired samples the CI can't clear 1.15x -> NOT a reliable win.
    assert verdict in ("MARGINAL", "FALSE")


def test_clearly_real_win_is_classified_real():
    controls = [1000.0, 1020.0, 990.0, 1010.0]
    treatments = [500.0, 510.0, 490.0, 505.0]  # ~2x faster, tight spread
    lo, hi = speedup_ci(controls, treatments)
    verdict, reason = classify_speedup(lo, hi)
    assert verdict == "REAL", reason
    assert lo >= 1.15


def test_win_strictly_below_one_is_false():
    # reliably SLOWER -> never a 'win'
    controls = [1000.0, 1000.0, 1000.0]
    treatments = [1500.0, 1500.0, 1500.0]
    lo, hi = speedup_ci(controls, treatments)
    assert classify_speedup(lo, hi)[0] == "FALSE"


def test_noise_ci_straddles_one():
    # high variance => interval around 1.0 => can't certify
    controls = [100.0, 400.0, 100.0]
    treatments = [110.0, 380.0, 120.0]
    lo, hi = speedup_ci(controls, treatments)
    assert lo <= 1.0 <= hi
    assert classify_speedup(lo, hi)[0] != "REAL"


# --- brittle vs robust metric ------------------------------------------------

def test_greedy_token_match_is_flagged_brittle():
    brittle, reason = flag_brittle_metric("token match", "token", uses_logits=False)
    assert brittle is True
    assert "can lie" in reason


def test_logit_based_is_not_brittle():
    brittle, _ = flag_brittle_metric("token match", "token", uses_logits=True)
    assert brittle is False


def test_robust_quality_identical_is_perfect():
    a = make_dummy_logits(6, 16, seed=1)
    cos, top5 = robust_logit_quality(a, a)
    assert cos == pytest.approx(1.0, abs=1e-4)
    assert top5 == pytest.approx(1.0)


def test_robust_quality_random_is_low():
    a = make_dummy_logits(6, 16, seed=0)
    b = make_dummy_logits(6, 16, seed=9)
    cos, top5 = robust_logit_quality(a, b)
    assert cos < 0.5
    assert top5 < 1.0


# --- config read by nothing ---------------------------------------------------

def test_documented_but_unread_key_flagged(tmp_path):
    # A config key that appears in comments/docstring but no code reads it.
    (tmp_path / "mod.py").write_text(
        '# --stream_layers\n"stream_layers is a knob"\n\nclass A:\n    pass\n')
    unread, reason = flag_read_by_nothing("stream_layers", str(tmp_path))
    assert unread is True, reason
    assert "read by NO" in reason


def test_read_key_is_not_flagged(tmp_path):
    (tmp_path / "mod.py").write_text(
        "class C:\n    def run(self, cfg):\n        return cfg.stream_layers\n")
    unread, reason = flag_read_by_nothing("stream_layers", str(tmp_path))
    assert unread is False, reason


# --- top-level audit ----------------------------------------------------------

def test_audit_treats_fast_but_brittle_metric_as_false():
    # A big real speedup, but the quality metric is brittle and no logits given
    # -> can't certify => overall FALSE.
    controls = [1000.0, 1010.0, 990.0]
    treatments = [500.0, 505.0, 495.0]
    v = audit(controls, treatments,
              metric_name="token match", comparison_level="token")
    assert v.speed_verdict == "REAL"
    assert v.metric_brittle is True
    assert v.verdict == "FALSE"


def test_audit_treats_real_speed_plus_robust_logits_as_real():
    controls = [1000.0, 1010.0, 990.0]
    treatments = [500.0, 505.0, 495.0]
    a = make_dummy_logits(6, 16, seed=2)
    v = audit(controls, treatments, metric_name="logit cosine",
              comparison_level="logit", fp_logits=a, iq_logits=a.clone())
    assert v.verdict == "REAL"
    assert v.metric_brittle is False
    assert v.robust_cos is not None


def test_audit_flags_unread_knob_as_false(tmp_path):
    (tmp_path / "mod.py").write_text('"uses --quant\n"\nclass X: pass\n')
    controls = [1000.0, 1010.0, 990.0]
    treatments = [500.0, 505.0, 495.0]
    a = make_dummy_logits(6, 16, seed=3)
    v = audit(controls, treatments, source_root=str(tmp_path),
              fp_logits=a, iq_logits=a.clone(),
              config_key="quant")
    assert v.verdict == "FALSE"
    assert "quant" in v.unread_keys