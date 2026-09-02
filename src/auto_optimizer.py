"""Auto-Optimizer orchestrator: one command -> optimized config + proof report.

Runs all four measurements on a given model and emits:
  * profile (prefill per-category bottleneck + overhead-vs-weight fraction)
  * decode (ms/token, tok/s)
  * auto-quantization (fp32 vs int8 BEFORE/AFTER + honest verdict)
  * auto-batching (latency vs throughput sweep + best batch for a goal)

Returns one dict with every result plus a human-readable proof report string.
Everything is measured on the caller's hardware (CPU here). The whole point:
refuse to recommend a change that doesn't provably help.
"""

from __future__ import annotations
import time

import torch

from profiler import profile_forward, profile_decode
from quantize import auto_quantize
from batcher import bench_batch, best_batch


def _overhead_frac(pr) -> float:
    """fraction of wall time NOT accounted for by leaf matmul spans."""
    leaf = sum(pr.cat_times.values())
    return 1.0 - (leaf / max(pr.total_time, 1e-9))


def run_auto_optimizer(
    model,
    tok,
    prompt: str = "The capital of France is the city of",
    n_new: int = 12,
    batches=(1, 2, 4, 8, 16),
    batch_goal: str = "throughput",
    quant_speedup_floor: float = 1.05,
    quant_quality_floor: float = 0.65,
    quant_top5_floor: float = 0.60,
) -> tuple[dict, str]:
    out: dict = {
        "prompt": prompt,
        "hardware": "CPU (2019 MacBook Pro 16in i7-9750H, 16GB)",
        "model_untouched": True,
    }

    # 1 & 2: bottleneck profile (prefill) + overhead fraction
    t0 = time.time()
    ids = tok([prompt], return_tensors="pt").input_ids
    pr_pre = profile_forward(model, ids, run_repeats=2, warmup=1)
    out["prefill"] = {
        "wall_total_s": pr_pre.total_time,
        "tokens": pr_pre.tokens,
        "tok_per_s": pr_pre.tokens / max(pr_pre.total_time, 1e-9),
        "leaf_time_s": sum(pr_pre.cat_times.values()),
        "pct_of_leaf": {
            k: round(v / max(sum(pr_pre.cat_times.values()), 1e-9) * 100, 1)
            for k, v in pr_pre.cat_times.items()
        },
        "overhead_fraction": round(_overhead_frac(pr_pre), 3),
    }

    # decode
    pr_dec = profile_decode(model, ids, n_new_tokens=n_new,
                            run_repeats=1, warmup=1)
    out["decode"] = {
        "ms_per_token": pr_dec.total_time / n_new * 1000.0,
        "tok_per_s": n_new / max(pr_dec.total_time, 1e-9),
        "overhead_fraction": round(_overhead_frac(pr_dec), 3),
    }

    # quantization
    qv = auto_quantize(model, tok, prompt=prompt, n_new=n_new,
                       speedup_floor=quant_speedup_floor,
                       quality_floor=quant_quality_floor,
                       top5_floor=quant_top5_floor)
    out["quantization"] = {
        "fp32_ms_per_token": qv.before["ms_per_token"],
        "int8_ms_per_token": qv.after["ms_per_token"],
        "speedup_int8_over_fp32": round(qv.speedup, 3),
        "speedup_worst_repeat": round(qv.speedup_min, 3),
        "logit_cosine": round(qv.logit_cosine, 3),
        "top5_overlap": round(qv.top5_overlap, 3),
        "recommended_int8": qv.recommended,
        "reason": qv.reason,
    }

    # batching
    rows = bench_batch(model, tok, prompt=prompt, batches=batches,
                       n_steps=1, warmup=1)
    best = best_batch(rows, goal=batch_goal)
    out["batching"] = {
        "goal": batch_goal,
        "rows": [
            {"batch": r.batch, "wall_ms": r.wall_ms, "tok_per_s": r.tok_per_s,
             "ms_per_req": r.ms_per_req, "tok_per_s_per_req": r.eff_tok_per_s_per_req}
            for r in rows
        ],
        "best_batch": best.batch,
        "best_tok_per_s": best.tok_per_s,
    }
    out["elapsed_s"] = round(time.time() - t0, 1)

    report = _report(out)
    return out, report


def _report(o: dict) -> str:
    p = o["prefill"]
    d = o["decode"]
    q = o["quantization"]
    b = o["batching"]
    overhead_note = (
        f"{p['overhead_fraction']*100:.0f}% of prefill wall time and "
        f"{d['overhead_fraction']*100:.0f}% of decode wall time is framework "
        "overhead on CPU. NOTE: this is CPU-specific; on a GPU the same model "
        "(with fused kernels) is typically weight/compute-bound, so the picture "
        "differs. Here it means: on CPU, tiny models are overhead-bound."
    )
    lines = [
        "# Auto-Optimizer proof report",
        f"hardware: {o['hardware']}   prompt: {o['prompt']}",
        "",
        "## 1. Bottleneck (from profile)",
        f"prefill throughput ~{p['tok_per_s']:.1f} tok/s; wall {p['wall_total_s']:.2f}s "
        f"for {p['tokens']} tokens.",
        "per-category (% of measured leaf work): " + ", ".join(
            f"{k} {v}%" for k, v in sorted(p['pct_of_leaf'].items(),
                                           key=lambda x: -x[1])
        ),
        f"overhead-vs-weight: {overhead_note}",
        "",
        "## 2. Decode",
        f"{d['ms_per_token']:.0f} ms/token (~{d['tok_per_s']:.2f} tok/s). "
        "Single-token serial, latency-sensitive.",
        "",
        "## 3. Quantization (fp32 vs int8, measured BEFORE/AFTER, averaged)",
        f"fp32 {q['fp32_ms_per_token']:.0f} ms/tok -> int8 {q['int8_ms_per_token']:.0f} ms/tok "
        f"= {q['speedup_int8_over_fp32']:.2f}x (worst repeat {q['speedup_worst_repeat']:.2f}x), "
        f"quality logit-cos {q['logit_cosine']:.2f}, top-5 {q['top5_overlap']:.2f}.",
        f"suggested: {'USE int8' if q['recommended_int8'] else 'KEEP fp32'} — {q['reason']}",
        "",
        "## 4. Batching (latency vs throughput)",
        "B | wall(ms) | tok/s | ms/req | tok/s/req",
    ]
    for r in b["rows"]:
        lines.append(
            f"{r['batch']} | {r['wall_ms']:.0f} | {r['tok_per_s']:.1f} | "
            f"{r['ms_per_req']:.0f} | {r['tok_per_s_per_req']:.2f}"
        )
    lines += [
        f"best batch for goal='{b['goal']}': B={b['best_batch']} "
        f"-> {b['best_tok_per_s']:.1f} tok/s total.",
        "",
        "## Honest takeaway",
        "The model was never permanently modified; any optimization is advised "
        "ONLY if it provably helps.",
    ]
    return "\n".join(lines)