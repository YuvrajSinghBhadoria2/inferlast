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
except ImportError:  # when invoked from the source tree, not the wheel
    sys.path.insert(0, "src")
    from auto_optimizer import run_auto_optimizer
    from transformers import AutoModelForCausalLM, AutoTokenizer


def run(args: argparse.Namespace) -> None:
    print(f"[inferlast] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    _, report = run_auto_optimizer(model, tok, prompt=args.prompt, n_new=args.n_new)
    print(report)


def main() -> None:
    ap = argparse.ArgumentParser(prog="inferlast",
                                 description="CPU-first LLM inference optimizer")
    sub = ap.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("run", help="profile -> pick -> prove a real win")
    cmd.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    cmd.add_argument("--prompt", default="The capital of France is the city of")
    cmd.set_defaults(func=run)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()