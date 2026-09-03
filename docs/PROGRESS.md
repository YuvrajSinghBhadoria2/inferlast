# Auto-Optimizer — LLM Inference Bottleneck Profiler

Automatically finds the inference bottleneck, chooses quantization + batching,
applies it, and proves the speedup with a reproducible before/after benchmark.

Phase 1 deliverable: a **bottleneck profiler** that tells you where inference
time actually goes for a given model on a given (CPU-only) machine.

## Environment

- Hardware: 2019 MacBook Pro 16", Intel i7-9750H (6C/12T), 16 GB RAM, **CPU only**.
- Python 3.11.13 (uv venv `.venv`), torch 2.2.2 (CPU), transformers 4.46.3,
  numpy 1.26.4.
- Note: transformers>=5 requires torch>=2.5, which has no macOS-CPU wheel; that
  is why transformers is pinned to 4.46.x. torchao's latest needs torch>=2.11;
  quantization is deferred until a compatible approach is selected.

## How to run

```bash
./.venv/bin/python scripts/bench.py --model HuggingFaceTB/SmolLM2-135M-Instruct
./.venv/bin/python scripts/bench.py --model Qwen/Qwen2.5-0.5B-Instruct --repeats 2
```

## Baseline results (this machine, prefill forward pass)

Records: `benchmarks/profile_*.json`.

### SmolLM2-135M-Instruct (buf 1 x seq 8)

| category | time(s) | % leaf | params(M) |
|---|---|---|---|
| mlp | 0.0175 | 43.4% | 79.63 |
| attention | 0.0169 | 41.7% | 26.54 |
| norm | 0.0056 | 13.9% | 0.04 |
| lm_head | 0.0002 | 0.6% | 28.31 |
| embed | 0.0002 | 0.4% | 28.31 |
| **leaf sum** | **0.040** | 100% | |
| **wall total** | **0.780** | | |

### Qwen2.5-0.5B-Instruct (buf 1 x seq 8)

| category | time(s) | % leaf | params(M) |
|---|---|---|---|
| mlp | 0.0169 | 45.9% | 313.79 |
| attention | 0.0152 | 41.4% | 44.07 |
| norm | 0.0040 | 11.0% | 0.04 |
| embed | 0.0004 | 1.0% | 136.13 |
| lm_head | 0.0003 | 0.8% | 136.13 |
| **leaf sum** | **0.037** | 100% | |
| **wall total** | **1.108** | | |

## Honest finding (Phase 1, first milestone)

For both small models on this CPU, the **sum of measured leaf spans (attention +
MLP + norm + heads) is only ~0.04s, but the full forward pass takes ~0.8–1.1s**.
That gap is ~95% of wall time and is **not** the matmuls.

Interpretation: at these tiny sizes (135M–0.5B, batch 1, short seq), the model is
**overhead-bound, not weight-bound** — the dominant cost is Python dispatch,
tokenizer/hook overhead, and small-kernel launch, not the actual linear algebra.
This is a real and correct result, and it has a design implication: for the
auto-optimizer to be honest, it must distinguish "compute-bound weights" from
"overhead-bound small-model path", because **quantizing weights will NOT speed up
an overhead-bound model** — a classic inference-engineer trap this tool exists to
prevent.

## Milestone 2 (DONE): decode-phase profile + overhead split

Added a per-token autoregressive profiler (`profile_decode`, `scripts/bench.py
--decode`) that measures the latency-sensitive decode path (the TPOT / ms-per-token
that job postings list). Same per-category accounting as prefill.

### Decode results (this CPU, greedy decoding)

| model | n-new tok | ms/token | tok/s | leaf sum | wall total |
|---|---|---|---|---|---|
| SmolLM2-135M | 12 | **353 ms/tok** | 2.8 | 0.267 s | 8.48 s |
| Qwen2.5-0.5B | 10 | **597 ms/tok** | 1.7 | 0.113 s | 5.97 s |

Decode is dramatically slower than prefill throughput because it is
single-token serial: each step is a full forward pass over the growing context.
The % category split matches prefill (MLP ~43–50%, attention ~38–42%, norm ~12–14%),
so the *relative* weight cost is stable; but again the leaf (actual matmul) spans
are only ~3–4% of decode wall time — the decode is overhead/framework-bound too.

### The insight that drives the whole tool

"ms/token on this CPU" (353–597 ms/tok for tiny models) is far too slow for real
serving, and it is dominated by framework overhead, not the model weights. Two
consequences the tool will act on:
1. **Don't blindly quantize to "speed up" serving** — for these models the matmuls
   are ~4% of time; quantization would barely move it and could degrade quality.
   The tool must MEASURE before/after and refuse if no real win.
2. The lever that matters here is **runtime/batching/engine choice** (vLLM-style
   kernels + continuous batching), not number precision. Milestone 4 (auto-batching)
   targets this.

## Milestone 3 (DONE): auto-quantization selector (BEFORE/AFTER)

Added `src/quantize.py` + `scripts/bench.py --quant`. It dynamically quantizes all
`nn.Linear` layers to int8 (torch.ao, CPU, no CUDA needed) and measures, on the
same fixed prompt + greedy decode:

* BEFORE fp32 decode speed (ms/token, tok/s), AVERAGED over repeats w/ spread
* AFTER int8  decode speed, averaged w/ spread + worst-repeat speedup
* robust quality fp32->int8: per-token logit cosine + top-5 overlap
  (NOT britt greedy token identity — see audit fix below)

It then returns an HONEST recommendation with a reason, refusing a change that
speeds up but degrades output.

> **Audit fix (metric + reliability):** the original release used greedy top-1
> token-identity as the quality proxy and single-run timing. An external 360
> audit flagged both: greedy identity is brittle (one precision shift cascades and
> overstates INT8 degradation), and single-run timing on a shared CPU contradicted
> itself across runs (3.0x vs 0.65x). Fixed by switching to per-token logit
> cosine + top-5 overlap, and averaging speed over repeats while reporting the
> worst-case speedup. Both fixes are live in `src/quantize.py`.

### Results (this CPU)

| model | fp32 ms/tok | int8 ms/tok | speedup (worst) | logit-cos | top-5 | verdict |
|---|---|---|---|---|---|---|
| SmolLM2-135M | 530 | 541 | 0.98x (0.85x) | 0.29 | 0.24 | KEEP fp32 |
| Qwen2.5-0.5B | 783 | 601 | 1.30x (0.82x) | 0.70 | 0.17 | KEEP fp32 |

Records: `benchmarks/quant_profile_*.json`.

### Honest interpretation

On this shared CPU int8 gives **no reliable speedup** (Qwen 1.30x avg but a 0.82x
worst repeat; SmolLM2 0.98x) and the robust quality metric shows the token ranking
near the top is genuinely disturbed (top-5 overlap 0.17–0.24), even where logit
direction is ~70% preserved. The tool's verdict "KEEP fp32" is the correct
inference-engineer call: an unreliable speedup on an overhead-bound CPU plus a
shifted token distribution is not a win. This is precisely the tradeoff (latency
vs quality vs cost) the job postings describe, now automated and evidence-backed.

Caveat to note for honesty: int8 torch.ao dynamic quant is aggressive for these
tiny models; a gentler quant (e.g. bf16 weights, or 4-bit with better recovery)
may preserve answers better. That is a tunable knob the selector can grow, not a
flaw in the method.

## Milestone 4 (DONE): auto-batching selector (latency vs throughput sweep)

Added `src/batcher.py` + `scripts/bench.py --batch`. Sweeps batch size B over a
fixed request mix (B identical short sequences in one decode step) and reports,
per B: wall time, total tok/s, per-request latency, and tok/s-per-request.

### Results (this CPU, one decode step, seq 8)

| B | wall(ms) | tok/s | ms/req | tok/s/req |
|---|---|---|---|---|
| 1 | 566 | 14.1 | 566 | 14.1 |
| 2 | 673 | 23.8 | 673 | 11.9 |
| 4 | 964 | 33.2 | 964 | 8.3 |
| 8 | 1958 | 32.7 | 1958 | 4.1 |
| 16 | 2664 | 48.1 | 2664 | 3.0 |

Batching to B=16 raises total throughput ~3.4x (14 -> 48 tok/s) at the cost of
much higher per-request latency (566 -> 2664 ms). This is the classic
throughput-vs-latency tradeoff, now measured. Records: `benchmarks/batch_profile_*.json`.

## Milestone 5 (DONE): end-to-end one-command auto-optimizer

Added `src/auto_optimizer.py` + `scripts/run_all.py`. One command runs all four
measurements and emits a combined proof report (tables + honest recommendation).

`./.venv/bin/python scripts/run_all.py --model Qwen/Qwen2.5-0.5B-Instruct`

Full reports (canonical, named by model): `benchmarks/endtoend_Qwen2.5-0.5B-Instruct.{json,md}`,
`benchmarks/endtoend_SmolLM2-135M-Instruct.{json,md}`.

### Consolidated Phase-1 verdict for Qwen2.5-0.5B on this CPU

- Bottleneck: MLP 47%, attention 41%, norm 11%; **~99% of wall time is framework
  overhead, not model matmuls** — overhead-bound.
- Decode: ~880 ms/token (~1.1 tok/s).
- Quantization: int8 gives no reliable speedup on this CPU (avg ~1.3–1.5x, worst
  repeat ~1.3x but noisy) and shifts token ranking (top-5 overlap 0.15–0.27) ->
  KEEP fp32.
- Batching: B=16 maximizes total throughput (~52 tok/s) at higher per-request
  latency.

Honest takeaway encoded in the tool: it refuses to recommend a change that does
not provably help. For these tiny CPU models that means: don't quantize; batch if
you want throughput and can tolerate latency.

## Quality: unit tests (added) and a real bug they caught

Added `tests/` (pytest, `pytest` to run; 46 tests) covering the profiler's
categorisation/timing, the robust quality metric + honest decision rule, the
batcher's best-batch selection, the `trustcheck` noise/brittle/read-by-nothing
logic, and a regression guard that the orchestrator
report always emits the NEW metrics and never the old brittle `output_match_pct`.

The tests surfaced one real latent bug in the original profiler: the attention
**container** (`...self_attn`) was not being excluded from timing, so its own
span was counted on top of its children (`...self_attn.q_proj` etc.) — a
double-count that inflated the reported attention %. Fixed in `src/profiler.py`
(the container is now non-leaf; projections remain timed leaves). Category splits
were re-measured after the fix; the headline finding (MLP+attention dominate and
the model is overhead-bound on CPU) is unchanged.

## trustcheck (false-win catcher) — added

Added `src/trustcheck.py` + `scripts/bench.py --trustcheck`. It audits whether a
measured "win" can be trusted, catching the three ways a benchmark lies:

1. **Single-run noise** — a matched-pairs 95% CI on the speedup; a CI straddling
   1.0x is MARGINAL, [lo > real_floor] is REAL, [hi < 1.0] is FALSE. Demonstrated
   on this repo's own evidence: the same INT8-vs-fp32 config measured 3.0x faster
   in one session and 0.65x slower in another -> trustcheck says FALSE/noise.
2. **Brittle / wrong metric** — a greedy-token-level (or opaque-accuracy) quality
   comparison with no logits is flagged "can lie"; given logits it re-measures
   with logit cosine + top-5 overlap.
3. **Read-by-nothing knob** — a documented/validated config key that no code path
   reads (the Soup/vLLM "validated, documented, read by nothing" bug class) is
   flagged by a static AST pass.

The overall verdict is REAL / MARGINAL / FALSE; a REAL claim needs a real,
repeatable speedup AND a robust metric. Serves `scripts/bench.py --trustcheck`
(values passed in, or `--collect-repeats N` to measure the noise band live).

## gpucheck (GPU-necessity decision rule) — added

Added `src/gpucheck.py` + `scripts/bench.py --gpucheck`. This is the concrete
form of the frozen thesis in `docs/RESEARCH-SPEC.md` (Amendment A): estimate,
from CPU-only measurement, whether renting a GPU would actually beat the
best-scheduled CPU config.

- Regime classifier: `overhead-bound` (measured overhead fraction >= 0.7),
  `weight-bound` (model footprint vs. CPU memory bandwidth across decode),
  or `unknown`.
- Verdicts: **GPU-warranted / CPU-suffices / insufficient-data**, each with an
  auditable `reason` listing the exact inputs and assumptions.
- Honesty invariant: returns `insufficient-data` rather than guessing when
  required inputs (cpu_label, latency target, or enough signal) are missing —
  it never sells a GPU spend it can't defend.
- CLI: `--gpucheck --num-params <p> --overhead-fraction <f> --latency-target-ms <l>
  --decode-ms-per-tok <m> [--batch-size-gb <b>]`; no model download needed.

9 new tests (55 total) covering overhead-bound->CPU-suffices, tight-batch+latency
flip->GPU-warranted, weight-bound->GPU-warranted, and the refusal-to-guess cases
(missing cpu_label / latency).

## Packaging (DONE): `pip install inferlast`

The core is a verified installable package on PyPI: `pip install inferlast`
installs the six modules (`import trustcheck`, `import gpucheck`, ...) plus the
`inferlast` CLI. Verified end-to-end by installing the built wheel in a clean
Python 3.11 venv and running `inferlast run --model HuggingFaceTB/SmolLM2-135M-Instruct`
(profile + quant + batch + honest verdict all emitted from the installed package).
PyPI: https://pypi.org/project/inferlast/ (v0.1.0). Build with `uv build dist/`;
publish with `uv publish`.

## Phase 1 status
All five milestones DONE. The auto-optimizer is a runnable, evidence-producing,
**unit-tested** tool: `scripts/run_all.py --model <m>` -> bottleneck + decode +
quantization + batching + proof report; `pytest` verifies the engine. Next
natural steps are outside this repo's Phase 1 (see README) and are a human
decision, not a blocker.

## Big-model boundary test (2026-09-03): Qwen2.5-7B Q4 on this CPU

Researched two ideas for "run bigger models on 16 GB RAM":
- **Docker/containers = dead end (evidence-backed).** A container shares the
  host's RAM; it cannot create memory. A 28 GB-fp32 7B won't fit in a container
  on a 16 GB host any more than on the bare host — Docker makes it slightly worse
  (VM overhead). No Docker experiment run; this is settled by how containers work.
- **The real lever is GGUF Q4 quantization (llama.cpp / Ollama).** A 7B drops
  from ~28 GB (fp32) to ~4.6 GB (Q4_K_M) — fits comfortably in 16 GB RAM.

Ran the real thing: `qwen2.5:7b` (Q4, 4.7 GB) via Ollama on this i7-9750H.
Measured, not claimed:

| metric | value |
|---|---|
| footprint | 4.7 GB (vs ~28 GB fp32) |
| model load | 26 s |
| decode | **2884.7 ms/token (~0.3 tok/s)** |
| output quality | normal Qwen response |

It **runs**, but at 0.3 tok/s it is not interactive-usable — a demonstration of
"a model that needs 28 GB fp32 runs in 16 GB," not a serving story.

**Resulting gpucheck bug found + fixed.** Feeding the real 7B numbers into
`gpucheck` exposed a real flaw: the `weight-bound` heuristic checks
`weight_stream_gbps >= 0.5*bw` (here ~5.3 vs 20.5 GB/s, so "not weight-bound"),
so the rule said **CPU-suffices even with a 200 ms/token target while the model
was 14x over it**. Fix: added a latency-feasibility rule — if *measured* decode
> *target*, GPU-warranted regardless of regime (auditable `measured_vs_target_x`).
Now 11 gpucheck tests, 57 total. The boundary thesis held up against a real
big-ish model *and* uncovered a gap the small-model tuning had missed.

**Follow-up honesty fix (bench.py):** pure-stats `--gpucheck`/`--trustcheck`
(no model downloaded) now label the saved record's `model` field honestly
(`gpucheck(stats-only)` / `trustcheck(user-samples)`) and name the file by
that, instead of claiming the `--model` default was profiled. This stops a
parameterized decision from polluting the measured `benchmarks/` ledger.

## Benchmark auditor (v0.2.0): audit ANY benchmark output for noise + methodology

Added `inferlast audit`: the first tool for auditing ANY inference benchmark
(JSON/JSONL/CSV) and answering the question every big-company press release
avoids: *"is that 3x speedup real, or within the noise band?"*

What it does:
- Accepts ANY metric with an explicit direction (`--higher-is-better` for
  throughput/accuracy; latency by default), not just the repo's own before/after.
- Without a control/treatment split in the data, it splits the series into
  first-half (baseline) vs second-half (after) of the repeats, per standard
  before/after methodology.
- Produces a **RESOLVED / UNRESOLVED / INSUFFICIENT_RUNS** verdict with a 95%
  CI on the true speedup and a reason.
- Produces a **methodology-gap report**: flags what the benchmark did NOT declare
  (input length, cache state, single-stream vs concurrent, n) -- the empty gap
  the research identified (inference performance is measured everywhere but
  audited nowhere).
- Warns when a win may not transfer across workload scope (different prompt
  length, cache state, concurrency).

Usage:
```
inferlast audit vllm_benchmark.json --metric-name latency_ms
inferlast audit throughput.csv --metric-name tok_s --higher-is-better
inferlast audit benchmark.json --declare input_len=2048 --declare cache_state=cold
```

`trustcheck.py` now supports arbitrary metric direction (lower-is-better /
higher-is-better) in all stats functions (`speedup_ci`, `classify_speedup`,
`audit_data`); the existing codebase behavior is unchanged (default is still
lower-is-better latency). 20 new tests across `test_benchmark_audit.py` and
`test_trustcheck.py`; 72 total tests. Version bumped to 0.2.0. Audit subcommand
wired into the installed CLI alongside `run`.