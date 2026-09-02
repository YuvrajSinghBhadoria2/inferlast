"""Guard against the stale-evidence regression: the orchestrator output and the
report must always carry the NEW robust metrics, never the old brittle ones."""

import torch
import pytest

from conftest import make_mini_lm
from auto_optimizer import run_auto_optimizer


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
    # all persisted sections present
    for section in ("prefill", "decode", "quantization", "batching", "elapsed_s"):
        assert section in out