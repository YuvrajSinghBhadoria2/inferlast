# Contributing to inferlast

Thanks for being here. inferlast is CPU-first inference optimization with honest
benchmarks, and every contributor makes the "no GPU required" story stronger.

## Ways to help (no code needed)

- **Try it and tell us what happened** — run `scripts/run_all.py --model <m>` on
  a model you care about and open a Discussion with the report.
- **Report a bug or a surprising verdict** — especially if `trustcheck` calls a
  benchmark FALSE and you think it's wrong. That's a real bug worth fixing.
- **Add a model to the tested list** — small CPU-capable models are very welcome.

## Setting up

```bash
git clone https://github.com/YuvrajSinghBhadoria2/inferlast.git
cd inferlast
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
python -m pytest          # 46 tests, no model downloads
```

## Running your first report

```bash
python scripts/run_all.py --model Qwen/Qwen2.5-0.5B-Instruct
python scripts/bench.py --model HuggingFaceTB/SmolLM2-135M-Instruct --trustcheck --collect-repeats 2
```

## Submitting a PR

1. Open a Discussion first if the change is non-trivial.
2. Add a test for anything you fix or add (tests run with `pytest`, no downloads).
3. Run `python -m pytest` and make sure it's green.
4. Keep `README.md` numbers honest — they must match `benchmarks/` evidence.

## Honesty rules (the point of the project)

- Never invent or round up a measurement.
- If an optimization doesn't help, say so — refusing a bad win is the feature.
- Published failure/negative results are welcome and valuable.