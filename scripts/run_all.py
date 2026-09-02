"""One-command entry point for the Auto-Optimizer.

Usage:
  ./.venv/bin/python scripts/run_all.py --model Qwen/Qwen2.5-0.5B-Instruct
  ./.venv/bin/python scripts/run_all.py --model HuggingFaceTB/SmolLM2-135M-Instruct --prompt "Hello"
  ./.venv/bin/python scripts/run_all.py --model ... --save out/report.json

Runs prefill profile + decode + auto-quantization + auto-batching for a model,
prints a proof report, and persistently writes a canonical JSON + .md report so
the proof artifact always matches the latest run and never goes stale.

Default output: benchmarks/endtoend_<model>.json and .md (next to the other
measured records). Override with --save <path>.
"""

from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_optimizer import run_auto_optimizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is the city of")
    ap.add_argument("--n-new", type=int, default=12)
    ap.add_argument("--save", default=None,
                    help="write report to this path (json + .md side by side). "
                         "Defaults to benchmarks/endtoend_<model>.{json,md}.")
    ap.add_argument("--benchmarks", default=os.path.join(
        os.path.dirname(__file__), "..", "benchmarks"))
    args = ap.parse_args()

    print(f"[auto-optimizer] loading {args.model} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    print(f"model loaded in {time.time()-t0:.1f}s; running all measurements ...\n")

    result, report = run_auto_optimizer(
        model, tok, prompt=args.prompt, n_new=args.n_new,
    )
    print(report)
    print(f"\n[auto-optimizer] done in {result['elapsed_s']}s")

    # Canonical default path so the report is NEVER stale after a run.
    if args.save:
        base = args.save.replace(".json", "").replace(".md", "")
    else:
        short = args.model.split("/")[-1]
        base = os.path.join(args.benchmarks, f"endtoend_{short}")
    with open(base + ".json", "w") as f:
        json.dump(result, f, indent=2)
    with open(base + ".md", "w") as f:
        f.write(report)
    print(f"saved -> {base}.json and {base}.md")


if __name__ == "__main__":
    main()