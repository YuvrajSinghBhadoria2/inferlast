"""Measure baseline profiles for a set of small LLMs on CPU.

Usage:
  ./.venv/bin/python scripts/bench.py --model HuggingFaceTB/SmolLM2-135M-Instruct
  ./.venv/bin/python scripts/bench.py --model HuggingFaceTB/SmolLM2-135M-Instruct --decode --n-new 16

Saves a JSON record under benchmarks/.
"""

from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from profiler import profile_forward, profile_decode, summarize


def save(record, args, tags=""):
    if args.out:
        outpath = os.path.join(os.path.dirname(__file__), "..", "benchmarks", args.out)
    else:
        short = args.model.split("/")[-1]
        outpath = os.path.join(
            os.path.dirname(__file__), "..", "benchmarks",
            f"{tags}profile_{short}.json".lstrip("_"),
        )
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nsaved -> {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--prompt", default="The capital of France is the city of")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--decode", action="store_true",
                    help="profile decode (autoregressive) phase instead of prefill")
    ap.add_argument("--quant", action="store_true",
                    help="run auto-quantization BEFORE/AFTER comparison (Milestone 3)")
    ap.add_argument("--batch", action="store_true",
                    help="run auto-batching latency/throughput sweep (Milestone 4)")
    ap.add_argument("--n-new", type=int, default=16,
                    help="new tokens generated per decode repeat")
    ap.add_argument("--out", default=None, help="output JSON path under benchmarks/")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    ids = tok([args.prompt], return_tensors="pt")["input_ids"]

    base = {
        "model": args.model,
        "prompt": args.prompt,
        "input_shape": list(ids.shape),
        "device": "cpu",
        "hardware": "2019 MacBook Pro 16in i7-9750H CPU-only 16GB",
        "run_repeats": args.repeats,
    }

    t0 = time.time()
    if args.quant:
        from quantize import auto_quantize, summarize_verdict
        print("== auto-quantization BEFORE/AFTER ==")
        v = auto_quantize(model, tok, prompt=args.prompt, n_new=args.n_new)
        print(summarize_verdict(v))
        record = {**base, "phase": "quantize_int8_vs_fp32",
                  "n_new_tokens": args.n_new,
                  "before_ms_per_token": v.before["ms_per_token"],
                  "before_tok_per_s": v.before["tok_per_s"],
                  "after_ms_per_token": v.after["ms_per_token"],
                  "after_tok_per_s": v.after["tok_per_s"],
                  "speedup_after_over_before": v.speedup,
                  "speedup_worst_repeat": v.speedup_min,
                  "logit_cosine": v.logit_cosine,
                  "top5_overlap": v.top5_overlap,
                  "repeats": v.after["repeats"],
                  "recommended_int8": v.recommended,
                  "reason": v.reason,
                  }
        save(record, args, tags="quant_")
    elif args.batch:
        from batcher import bench_batch, best_batch, summarize_batch
        print("== auto-batching sweep (B identical short requests, one step) ==")
        rows = bench_batch(model, tok, prompt=args.prompt,
                           batches=(1, 2, 4, 8, 16), n_steps=1, warmup=1)
        print(summarize_batch(rows))
        best = best_batch(rows, goal="throughput")
        print(f"\nbest batch for max total throughput: B={best.batch} "
              f"-> {best.tok_per_s:.1f} tok/s total")
        record = {**base, "phase": "batch_sweep",
                  "rows": [{"batch": r.batch, "wall_ms": r.wall_ms,
                            "tok_per_s": r.tok_per_s, "ms_per_req": r.ms_per_req,
                            "tok_per_s_per_req": r.eff_tok_per_s_per_req}
                           for r in rows],
                  "best_batch": best.batch,
                  "best_tok_per_s": best.tok_per_s,
                  }
        save(record, args, tags="batch_")
    elif args.decode:
        pr = profile_decode(model, ids, n_new_tokens=args.n_new,
                            run_repeats=args.repeats, warmup=args.warmup)
        print(summarize(pr))
        per_token = pr.total_time / max(args.n_new * args.repeats, 1e-9)
        print(f"\nde-code: {args.n_new} new tokens x {args.repeats} reps; "
              f"~{per_token*1000:.1f} ms/token (TPOT on this CPU)")
        ts = pr.total_time / max(args.n_new * args.repeats, 1e-9)
        record = {**base, "phase": "decode",
                  "n_new_tokens": args.n_new,
                  "total_time_s": pr.total_time,
                  "tokens": pr.tokens,
                  "throughput_tok_per_s": pr.tokens / max(pr.total_time, 1e-9),
                  "ms_per_token": ts * 1000,
                  "leaf_time_sum_s": sum(pr.cat_times.values()),
                  "cat_times_s": dict(pr.cat_times),
                  "cat_calls": dict(pr.cat_calls),
                  "cat_params": dict(pr.cat_params),
                  "pct_of_leaf_sum": {
                      k: round(v / sum(pr.cat_times.values()) * 100, 1)
                      for k, v in pr.cat_times.items()
                  },
                  }
        save(record, args, tags="decode_")
    else:
        pr = profile_forward(model, ids, run_repeats=args.repeats, warmup=args.warmup)
        print(summarize(pr))
        print(f"\n(profile step took {time.time()-t0:.1f}s)")
        record = {**base, "phase": "prefill",
                  "total_time_s": pr.total_time,
                  "tokens": pr.tokens,
                  "throughput_tok_per_s": pr.tokens / max(pr.total_time, 1e-9),
                  "leaf_time_sum_s": sum(pr.cat_times.values()),
                  "cat_times_s": dict(pr.cat_times),
                  "cat_calls": dict(pr.cat_calls),
                  "cat_params": dict(pr.cat_params),
                  "pct_of_leaf_sum": {
                      k: round(v / sum(pr.cat_times.values()) * 100, 1)
                      for k, v in pr.cat_times.items()
                  }}
        save(record, args)


if __name__ == "__main__":
    main()