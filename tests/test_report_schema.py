"""Guard against the stale-evidence regression: the orchestrator output and the
report must always carry the NEW robust metrics, never the old brittle ones."""

import torch
import pytest

from conftest import make_mini_lm
from auto_optimizer import run_auto_optimizer, _bottom_line


class FakeTok:
    def __init__(self, vocab=32):
        self.vocab = vocab

    def __call__(self, prompt, return_tensors="pt"):
        ids = torch.tensor([[0, 1, 2]])
        return type("Batch", (), {"input_ids": ids})()


NEW_KEYS = ("speedup_worst_repeat", "logit_cosine", "top5_overlap")
FORBIDDEN = ("output_match_pct",)


def test_report_uses_new_metrics_and_bans_stale_field():
    model = make_mini_lm(vocab=32)
    tok = FakeTok(vocab=32)
    out, report = run_auto_optimizer(
        model, tok, prompt="x", n_new=3,
        batches=(1, 2),
    )
    q = out["quantization"]
    for k in NEW_KEYS:
        assert k in q, f"orchestrator lost required key {k}"
    for k in FORBIDDEN:
        assert k not in q, f"stale brittle field {k} leaked into output"
    # the markdown report renders the new metric, never the old one
    assert "logit-cos" in report or "logit_cosine" in report
    assert "worst repeat" in report
    assert "output_match_pct" not in report  # never reconstruct stale text
    # report always leads with the sharp bottom-line verdict (not a generic wall)
    assert "## Fastest CPU config (bottom line)" in report
    # all persisted sections present
    for section in ("prefill", "decode", "quantization", "batching", "elapsed_s"):
        assert section in out
    # the KV-cache technique slot always exists and reports availability
    assert "techniques" in out
    assert "kv_cache" in out["techniques"]
    assert out["techniques"]["kv_cache"]["avail"] is False  # MiniLM has no cache API
    # speculative slot always exists; off by default (no draft supplied)
    assert "speculative" in out["techniques"]
    assert out["techniques"]["speculative"]["avail"] is False


def _base():
    return {
        "quantization": {"recommended_int8": False,
                         "speedup_int8_over_fp32": 1.47,
                         "top5_overlap": 0.18},
        "batching": {"rows": [{"tok_per_s": 25.0}, {"tok_per_s": 83.8}],
                     "best_batch": 16, "best_tok_per_s": 83.8},
        "techniques": {"kv_cache": {"avail": True, "kv_speedup_x": 1.48},
                       "speculative": {"avail": False}},
    }


def test_bottom_line_reports_real_wins_and_rejects_fake_quant():
    lines = _bottom_line(_base())
    txt = "\n".join(lines)
    assert "batching B=16" in txt and "3.4x total throughput" in txt
    assert "KV cache -> ~1.5x decode" in txt
    # quantization is rejected despite being FASTER: the top-5 quality broke
    assert "KEEP fp32" in txt
    assert "1.47x faster" in txt and "quality top-5 0.18" in txt
    # honesty: it explicitly says the 1.47x is NOT a real win
    assert "not a real win" in txt


def test_bottom_line_ships_int8_when_quality_holds():
    o = _base()
    o["quantization"] = {"recommended_int8": True,
                         "speedup_int8_over_fp32": 1.8,
                         "top5_overlap": 0.9}
    txt = "\n".join(_bottom_line(o))
    assert "SHIP int8" in txt and "1.80x faster" in txt


def test_bottom_line_labels_slower_int8_honestly():
    o = _base()
    o["quantization"] = {"recommended_int8": False,
                         "speedup_int8_over_fp32": 0.86,
                         "top5_overlap": 0.65}
    txt = "\n".join(_bottom_line(o))
    assert "KEEP fp32" in txt
    assert "slower, not faster" in txt  # must not call 0.86x "faster"
    assert "0.86x (slower" in txt


def test_bottom_line_surfaces_measured_speculative_slower():
    o = _base()
    o["techniques"]["speculative"] = {
        "avail": True, "speedup_x": 0.55, "verdict": "SLOWER"}
    txt = "\n".join(_bottom_line(o))
    assert "0.55x -> SLOWER" in txt


def test_bottom_line_no_win_treated_as_baseline():
    o = _base()
    o["batching"] = {"rows": [{"tok_per_s": 25.0}], "best_batch": 1,
                     "best_tok_per_s": 25.0}
    o["techniques"]["kv_cache"] = {"avail": False}
    txt = "\n".join(_bottom_line(o))
    assert "No single lever produced a clean measured win" in txt
