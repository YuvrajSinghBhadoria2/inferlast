"""Console entry point for inferlast.

Installed as the `inferlast` command when the wheel is installed, so you can run:

    inferlast run --model Qwen/Qwen2.5-0.5B-Instruct

without cloning the repo. Behavior matches scripts/run_all.py (same imports,
same measurement pipeline); this CLI just adds a pip-installable front door.
"""

from __future__ import annotations
import argparse
import sys

try:
    from auto_optimizer import run_auto_optimizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trustcheck import audit_benchmark_file, summarize_audit
    from trustcheck import DIRECTION_LOWER_IS_BETTER, DIRECTION_HIGHER_IS_BETTER
except ImportError:  # when invoked from the source tree, not the wheel
    sys.path.insert(0, "src")
    from auto_optimizer import run_auto_optimizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trustcheck import audit_benchmark_file, summarize_audit
    from trustcheck import DIRECTION_LOWER_IS_BETTER, DIRECTION_HIGHER_IS_BETTER


def run(args: argparse.Namespace) -> None:
    print(f"[inferlast] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    draft_model = None
    draft_tok = None
    if getattr(args, "draft_model", None):
        print(f"[inferlast] loading draft {args.draft_model} for speculative decode ...")
        draft_tok = AutoTokenizer.from_pretrained(args.draft_model)
        draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model)
        draft_model.eval()
    _, report = run_auto_optimizer(model, tok, prompt=args.prompt,
                                   n_new=getattr(args, "n_new", 12),
                                   draft_model=draft_model, draft_tok=draft_tok)
    print(report)


def audit(args: argparse.Namespace) -> None:
    """Audit ANY benchmark JSON/CSV: is the claimed 'win' real or within noise?
    Works on latency (lower-is-better, default) or throughput/accuracy
    (higher-is-better, --higher-is-better). Without a control/treatment pair it
    splits the series into first-half vs second-half of the repeats."""
    if getattr(args, "higher_is_better", False):
        direction = DIRECTION_HIGHER_IS_BETTER
    else:
        direction = DIRECTION_LOWER_IS_BETTER
    declared = {}
    if args.declare:
        for item in args.declare:
            if "=" in item:
                k, _, v = item.partition("=")
                declared[k.strip()] = v.strip()
    result = audit_benchmark_file(
        args.benchmark,
        metric_name=args.metric_name,
        direction=direction,
        declared=declared,
    )
    print(summarize_audit(result, metric_label=args.metric_name or ""))


def main() -> None:
    ap = argparse.ArgumentParser(prog="inferlast",
                                 description="CPU-first LLM inference optimizer")
    sub = ap.add_subparsers(dest="command", required=True)

    cmd = sub.add_parser("run", help="profile -> pick -> prove a real win")
    cmd.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    cmd.add_argument("--prompt", default="The capital of France is the city of")
    cmd.add_argument("--draft-model",
                     help="optional draft model for speculative (assisted) decode "
                          "measurement, e.g. HuggingFaceTB/SmolLM2-135M-Instruct")
    cmd.set_defaults(func=run)

    cmd = sub.add_parser(
        "audit",
        help="audit ANY benchmark JSON/CSV: is the 'win' real or within noise?")
    cmd.add_argument("benchmark", help="path to a benchmark .json/.jsonl/.csv file")
    cmd.add_argument("--metric-name",
                     help="column to audit; for a before/after file pass "
                          "'control_col,after_col'")
    cmd.add_argument("--higher-is-better", action="store_true",
                     help="metric is throughput/accuracy (default: latency, "
                          "lower-is-better)")
    cmd.add_argument("--declare", action="append", metavar="KEY=VALUE",
                     help="declare a methodology field, e.g. "
                          "--declare input_len=2048 --declare cache_state=cold")
    cmd.set_defaults(func=audit)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()