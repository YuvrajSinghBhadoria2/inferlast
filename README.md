# LLM Inference Auto-Optimizer (Phase 1)

Automatically finds where inference time really goes, selects a safe optimization,
applies it, and **proves** the outcome with a reproducible before/after benchmark.
It refuses to recommend a change it cannot measure as helpful.

**This is the decision-intelligence layer of inference engineering**: profile →
pick → apply → prove. It complements (does not re-implement) serving engines like
vLLM or llama.cpp.

> **Hardware scope: Phase 1 is CPU-only.** Built and measured on a 2019 Intel
> MacBook Pro 16" (i7-9750H, 6C/12T, 16 GB RAM, no GPU). Findings are CPU-specific
> and stated as such; on GPU the same model is typically weight-bandwidth-bound,
> not overhead-bound, so results there would differ.

## Why this exists (the honest insight it encodes)

For tiny models on CPU the tool measures that **~95–98% of decode wall time is
framework overhead, not model matmuls** — so **blindly quantizing the weights will
not speed up an overhead-bound model**. Rather than guess, the tool *measures*
BEFORE/AFTER and only recommends a change it can prove helps. That refusal-to-guess
behavior is the core value: it avoids the classic inference-engineer trap of
applying a "standard optimization" that doesn't help your real workload.

## Quick start

```bash
# Python 3.11 recommended (transformers 4.x, torch CPU 2.2)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python "transformers>=4.40,<4.47" "numpy<2"

# one command -> full proof report (bottleneck + decode + quant + batch)
./.venv/bin/python scripts/run_all.py --model Qwen/Qwen2.5-0.5B-Instruct

# individual stages
./.venv/bin/python scripts/bench.py --model Qwen/Qwen2.5-0.5B-Instruct          # prefill profile
./.venv/bin/python scripts/bench.py --model Qwen/Qwen2.5-0.5B-Instruct --decode  # decode
./.venv/bin/python scripts/bench.py --model <m> --quant                            # auto-quantization
./.venv/bin/python scripts/bench.py --model <m> --batch                            # auto-batching
```

Deep links: full examples in `docs/PROGRESS.md`; raw measured records in
`benchmarks/`; the end-to-end reports in `benchmarks/endtoend_*.{json,md}`.

## What each module does

| Module | Role |
|---|---|
| `src/profiler.py` | Per-category (attention/MLP/norm/embed/head) wall-clock forward and decode profiler; reports the overhead-vs-weight split. |
| `src/quantize.py` | Auto-quantization (INT8 dynamic, torch.ao). Measures fp32 vs INT8 decode speed **averaged over repeats** plus a **robust quality metric** (per-token logit cosine + top-5 overlap) — not brittle greedy-token identity. |
| `src/batcher.py` | Latency-vs-throughput sweep over batch size, with a 90%-of-peak best-batch picker. |
| `src/auto_optimizer.py` | Orchestrator: runs all four stages, emits a combined proof report + JSON. |
| `scripts/run_all.py` | One-command entry point. |
| `scripts/bench.py` | Stage-by-stage CLI for profiling/quant/batch. |

## Representative result (Qwen2.5-0.5B, this CPU)

```
Bottleneck (prefill): MLP 47%, attention 41%, norm 11%; ~99% of wall time is
  framework overhead on CPU (CPU-specific; GPU differs).
Decode: ~880 ms/token (1.1 tok/s), overhead-bound.
Quantization (fp32 vs INT8, n=3, spread reported):
  fp32 0.80 -> 0.52 s/tok = 1.4x, worst repeat 1.3x; quality logit-cos 0.68,
  top-5 overlap 0.15  -> KEEP fp32 (INT8 does not reliably help on CPU + shifts
  token ranking).   <-- honest "no", not a fake "everything is faster"
Batching: B=16 maximizes total throughput (~52 tok/s) at higher per-req latency.
```

Raw numbers: `benchmarks/endtoend_Qwen2.5-0.5B-Instruct.md`,
`benchmarks/endtoend_SmolLM2-135M-Instruct.md`.

## Honest limitations (Phase 1)

- **CPU-only.** No GPU profiling, no CUDA/Triton kernels. This is a deliberate
  Phase-1 scope; GPU extension is natural follow-up work.
- **INT8 only** (via torch.ao dynamic). No FP8/FP4/INT4 yet.
- **Average latency only** — p50/p99 latency percentiles are not yet captured.
- **No memory measurement** (peak RAM / KV-cache footprint) yet.
- Timing on a shared laptop CPU is noisy; the quantizer averages over repeats and
  reports the worst-case speedup because of this.

## Tests

```bash
uv pip install --python .venv/bin/python pytest
./.venv/bin/python -m pytest       # 33 fast tests, no model downloads
```

Covers profiler categorisation & timing, the robust INT8 quality metric +
honest decision rule, batcher best-batch selection, and a regression guard that
the report always emits the new metrics (never the stale `output_match_pct`).

## Verification / provenance

This tool is being tracked against a 360-degree audit that independently reviewed
the results. Key audit-driven fixes already applied: (1) quality metric changed
from greedy token-identity to per-token logit cosine + top-5 overlap (greedy
identity is brittle and overstated INT8 degradation); (2) quantization speed
averaged over repeats with spread + worst-case reported, resolving run-to-run
contradictions. Audit log: see initiative `BRIEF.md` and `docs/PROGRESS.md`.

## Layout

- `src/` — the four modules + orchestrator
- `scripts/` — CLI entry points
- `benchmarks/` — raw measured evidence (JSON) and reports (MD)
- `docs/PROGRESS.md` — milestone log and honest findings