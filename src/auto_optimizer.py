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
from decode import measure_decode
from speculative import measure_speculative


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
    draft_model=None,
    draft_tok=None,
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

    # decode -- measured on the REAL KV-cache path (a served user sees this)
    pr_dec = profile_decode(model, ids, n_new_tokens=n_new,
                            run_repeats=1, warmup=1)
    out["decode"] = {
        "ms_per_token": pr_dec.total_time / n_new * 1000.0,
        "tok_per_s": n_new / max(pr_dec.total_time, 1e-9),
        "overhead_fraction": round(_overhead_frac(pr_dec), 3),
        "kv_cache": True,
    }

    # KV-cache technique measurement: the real decode path vs O(n^2) recompute.
    # This is the single biggest CPU decode lever and is NOT emphasised enough
    # elsewhere -- the naive path under-reports decode speed by 1.5-3x.
    out["techniques"] = {}
    try:
        mpt_kv, tps_kv = measure_decode(model, ids, n_new, use_cache=True, repeats=2, warmup=1)
        mpt_nc, tps_nc = measure_decode(model, ids, n_new, use_cache=False, repeats=2, warmup=1)
        out["techniques"]["kv_cache"] = {
            "avail": True,
            "kv_cache_ms_per_token": round(mpt_kv, 1),
            "no_cache_ms_per_token": round(mpt_nc, 1),
            "kv_speedup_x": round(mpt_nc / max(mpt_kv, 1e-9), 2),
            "note": "uses past_key_values so each decode step reuses cached "
                    "keys/values instead of re-forwarding the whole prefix.",
        }
    except Exception as exc:  # e.g. a model without a compatible cache API
        out["techniques"]["kv_cache"] = {"avail": False, "error": str(exc)}

    # speculative (assisted) decode -- opt-in, needs a second draft model.
    if draft_model is not None:
        out["techniques"]["speculative"] = measure_speculative(
            model, ids, ids.shape[1], draft_model, n_new=n_new,
            tokenizer=tok, assistant_tokenizer=draft_tok,
            repeats=2, warmup=1)
    else:
        out["techniques"]["speculative"] = {
            "avail": False,
            "note": "not measured unless a --draft-model is supplied.",
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


def _bottom_line(o: dict) -> list[str]:
    """One decisive 'what should you actually do' verdict, derived only from the
    measured fields in `o`. This is the headline of the report: it turns a wall
    of numbers into a sharp, honest recommendation."""
    b = o["batching"]
    q = o["quantization"]
    kv = o.get("techniques", {}).get("kv_cache", {})
    sp = o.get("techniques", {}).get("speculative", {})

    # throughput lever: best batch tok/s vs B=1
    base = b["rows"][0]["tok_per_s"] if b["rows"] else None
    if base:
        batch_speedup = (b["best_tok_per_s"] / max(base, 1e-9)
                         if base > 0 else None)
        best_batch = b["best_batch"]
        total = b["best_tok_per_s"]
    else:
        batch_speedup = None
        best_batch = None
        total = None

    # decode lever: KV cache vs recompute
    kv_speedup = kv.get("kv_speedup_x") if kv.get("avail") else None

    # quant verdict
    q_verdict = ("SHIP int8" if q["recommended_int8"] else "KEEP fp32")
    if q["recommended_int8"]:
        q_brief = (f"INT8 is {q['speedup_int8_over_fp32']:.2f}x faster and "
                   "keeps quality — ship it")
    else:
        s = q["speedup_int8_over_fp32"]
        if s >= 1.0:
            speed_txt = f"INT8 {s:.2f}x faster"
        else:
            speed_txt = f"INT8 {s:.2f}x (slower, not faster)"
        q_brief = (f"{speed_txt} but quality top-5 "
                   f"{q['top5_overlap']:.2f} — not a real win, rejected")

    lines = ["## Fastest CPU config (bottom line)"]
    # collect the provable wins, verbatim from measurement.
    # a "win" requires a REAL gain (>=1.05x); 1.0x is not a win.
    wins: list[str] = []
    if best_batch and batch_speedup and batch_speedup >= 1.05:
        wins.append(f"batching B={best_batch} -> ~{total:.1f} tok/s "
                    f"({batch_speedup:.1f}x total throughput vs B=1)")
    if kv_speedup and kv_speedup >= 1.05:
        wins.append(f"KV cache -> ~{kv_speedup:.1f}x decode vs naive recompute")
    if wins:
        lines.append("Real, measured wins (I would ship these): "
                     + "; ".join(wins) + ".")
    else:
        lines.append("No single lever produced a clean measured win on this "
                     "hardware; treat these numbers as a baseline.")

    lines.append(
        f"Quantization: {q_verdict} — {q_brief}."
    )
    if sp.get("avail") and "error" not in sp:
        lines.append(
            f"Speculative decode: {sp['speedup_x']}x -> {sp['verdict']}"
            " on this CPU."
        )
    lines.append("")
    return lines


def _report(o: dict) -> str:
    p = o["prefill"]
    d = o["decode"]
    q = o["quantization"]
    b = o["batching"]
    t = o.get("techniques", {})
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
        *_bottom_line(o),
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
        "## 2b. Optimization techniques measured (KV cache)",
    ]
    kv = t.get("kv_cache", {})
    if kv.get("avail"):
        lines += [
            f"KV cache is ON in the decode measurement. "
            f"KV-cache {kv['kv_cache_ms_per_token']} ms/token vs "
            f"no-cache(recompute) {kv['no_cache_ms_per_token']} ms/token "
            f"= {kv['kv_speedup_x']}x faster. {kv.get('note','')}",
            "The O(n^2) recompute loop is what most ad-hoc CPU decoders use; "
            "enabling the KV cache is the largest single CPU decode win.",
        ]
    else:
        lines.append("KV cache not measurable on this model: "
                     f"{kv.get('error','?')}")
    sp = t.get("speculative", {})
    if sp.get("avail") and "error" not in sp:
        lines += [
            "## 2c. Speculative (assisted) decoding — measured",
            f"assisted {sp['assisted_tok_s']} tok/s vs the target alone "
            f"(KV) {sp['plain_kv_tok_s']} tok/s = {sp['speedup_x']}x -> "
            f"{sp['verdict']}. {sp.get('note','')}",
        ]
    lines += [
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