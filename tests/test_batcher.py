"""Tests for the auto-batcher: best_batch selection + sweep smoke."""

import torch
import pytest

from conftest import make_mini_lm
from batcher import BatchRow, best_batch


def _rows(tok_per_s):
    """rows with batch=1..len, deterministic throughputs (monotone up to a peak)."""
    return [BatchRow(batch=i + 1, wall_ms=100.0 / (t + 1e-9),
                     tok_per_s=t, ms_per_req=100.0, eff_tok_per_s_per_req=t / (i + 1))
            for i, t in enumerate(tok_per_s)]


def test_best_batch_throughput_picks_peak():
    rows = _rows([10.0, 25.0, 30.0, 22.0])  # peak at B=3
    best = best_batch(rows, goal="throughput")
    assert best.batch == 3


def test_best_batch_latency_picks_smallest_near_peak():
    rows = _rows([10.0, 48.0, 50.0, 50.0])  # peak=50; B=2 gives 48 = 96% >= 90%
    best = best_batch(rows, goal="latency")
    assert best.batch == 2


def test_best_batch_latency_falls_back_to_peak_if_none_close():
    rows = _rows([10.0, 20.0, 21.0])  # peak 21; B=1 (10=47%) not near, B=2 (20=95%)
    best = best_batch(rows, goal="latency")
    assert best.batch == 2


def test_best_batch_single_row():
    rows = _rows([5.0])
    assert best_batch(rows, goal="throughput").batch == 1
    assert best_batch(rows, goal="latency").batch == 1


# --- smoke: full sweep runs on the tiny model --------------------------------

class FakeTok:
    def __call__(self, prompt, return_tensors="pt"):
        ids = torch.tensor([[0, 1, 2, 3]])
        return type("Batch", (), {"input_ids": ids})()


def test_bench_batch_runs_and_scales_with_batch():
    from batcher import bench_batch
    model = make_mini_lm(vocab=32)
    tok = FakeTok()
    rows = bench_batch(model, tok, "x", batches=(1, 2, 4), seq_single=4,
                       n_steps=1, warmup=1)
    assert len(rows) == 3
    for r in rows:
        assert r.wall_ms > 0.0
        assert r.tok_per_s > 0.0
        assert r.eff_tok_per_s_per_req == pytest.approx(r.tok_per_s / r.batch)
        assert r.batch in (1, 2, 4)