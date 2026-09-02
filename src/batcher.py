"""Auto-batching selector: latency vs throughput tradeoff sweep.

Sweeps batch size B over a fixed prompt, running B identical (short) requests in a
single tensor of shape [B, seq]. For each B we measure:

  wall_time   : total time to do one decode step for the whole batch
  tok_per_s   : total tokens/s across the batch   (= B * seq' / time)
  ms_per_req  : effective latency per request      (= wall_time milliseconds,
                 since all requests complete together)

Key inference-engineer metric: does batching raise total throughput enough to
outweigh the per-request latency cost (and constant-memory limits)? We report
both so the user can pick B for their goal: max throughput vs min latency.

CPU-only. The measurement is one decode step averaged over n_steps, so it
captures the real throughput/latency of serving a fixed request mix.
"""

from __future__ import annotations
import time
from dataclasses import dataclass

import torch


@dataclass
class BatchRow:
    batch: int
    wall_ms: float        # ms for one decode step across the batch
    tok_per_s: float      # total tokens/s across the batch
    ms_per_req: float     # effective per-request latency (ms)
    eff_tok_per_s_per_req: float  # (per-request share) = tok_per_s / batch


def _one_decode_step(model, ids) -> None:
    out = model(ids)
    return out


def bench_batch(
    model,
    tok,
    prompt: str,
    batches=(1, 2, 4, 8, 16),
    seq_single: int = 8,
    n_steps: int = 2,
    warmup: int = 1,
) -> list[BatchRow]:
    """Measure one decode step at each batch size on CPU.
    prompt is tokenized once then replicated B times."""
    ids0 = tok([prompt], return_tensors="pt").input_ids  # [1, seq]
    seq = ids0.shape[1]
    rows = []
    model.eval()

    with torch.no_grad(), torch.inference_mode():
        for B in batches:
            ids = ids0.expand(B, -1).clone()  # [B, seq] identical requests
            for _ in range(warmup):
                _one_decode_step(model, ids)
            t0 = time.perf_counter()
            for _ in range(n_steps):
                _one_decode_step(model, ids)
            dt = time.perf_counter() - t0
            wall_ms = dt / n_steps * 1000.0
            # each step processes B sequences x seq tokens
            tok_per_s = (B * seq * n_steps) / dt
            ms_per_req = wall_ms  # requests finish together
            rows.append(BatchRow(
                batch=B, wall_ms=wall_ms, tok_per_s=tok_per_s,
                ms_per_req=ms_per_req, eff_tok_per_s_per_req=tok_per_s / B,
            ))
    return rows


def best_batch(rows: list[BatchRow], goal: str = "throughput"):
    """Pick the batch that maximizes the goal. 'throughput' -> max total tok/s
    (with a per-request latency guard), 'latency' -> smallest batch that still
    gets near-peak throughput."""
    if goal == "latency":
        # smallest B where throughput >= 90% of peak
        peak = max(r.tok_per_s for r in rows)
        for r in sorted(rows, key=lambda x: x.batch):
            if r.tok_per_s >= 0.90 * peak:
                return r
        return max(rows, key=lambda r: r.tok_per_s)
    return max(rows, key=lambda r: r.tok_per_s)


def summarize_batch(rows: list[BatchRow]) -> str:
    lines = [
        f"{'B':>4}{'wall(ms)':>12}{'tok/s':>12}{'ms/req':>12}{'tok/s/req':>12}",
    ]
    for r in rows:
        lines.append(
            f"{r.batch:>4}{r.wall_ms:>12.2f}{r.tok_per_s:>12.1f}"
            f"{r.ms_per_req:>12.2f}{r.eff_tok_per_s_per_req:>12.2f}"
        )
    return "\n".join(lines)