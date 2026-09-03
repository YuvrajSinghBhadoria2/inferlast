# RESEARCH SPEC — inferlast: the GPU-necessity decision rule for CPU-first inference

**Status:** DRAFT — this is the frozen, falsifiable thesis that defines the project's moat.
Everything (that second "geek, look at how this thing breaks down" concern from the reviewer, the
feature roadmap, the company narrative, the resume story) hangs on getting this claim crisp before
building further.

## 1. The one claim this project stands on

> **H0 (null, what we hope is often TRUE):** For overhead-bound small models on CPU, there exists a
> measurable boundary — in (overhead fraction, model size, batch size) — beyond which spending on GPU
> acceleration yields **no statistically significant latency/throughput win** over the best-scheduled
> CPU configuration. inferlast can estimate that boundary from **local CPU measurements alone, without
> ever renting a GPU.**

This is the falsifiable, novel contribution: **a decision rule for when GPU spend is warranted.** It is
what a company would be built on and what the literature does not cleanly provide.

## 2. Why this is the honest moat (standing on the prior work)

- The **engines** (llama.cpp, Ollama, ONNX Runtime, OpenVINO, TensorRT) benchmark and serve
  — they do not answer *"should you have spent on this at all."*
- The **GPU-serving** incumbents (vLLM/Inferact, SGLang) assume you already have GPUs.
- The **local-CPU** tier (Ollama ~9M users) is large and growing but has **no decide+trust layer.**
- Conclusion: inferlast is positioned as the **decision & trust layer on top of CPU engines**, not as
  another engine. Competing as an engine is not defensible; this gap is.

## 3. The falsifiable sub-claims (what a test must separately confirm)

1. **Overhead-boundness is measurable and dominant** for small models on CPU
   (decode wall-time is mostly framework/kernel overhead, not weight math). *Have evidence:* ~98–99%.
2. **Quantization is not a free win on overhead-bound models** — a robust logit-geometry metric
   (cosine + top-5 overlap) can reveal a real quality shift that greedy-token identity hides.
   *Have evidence:* INT8 1.5x but top-5 0.15 → KEEP fp32.
3. **A "reported benchmark win" is an epistemic claim** needing a falsification route:
   REAL ⟺ repeatable speedup (matched-pairs CI above the noise floor) ∧ non-brittle metric ∧ no
   "read-by-nothing" knob. *Have evidence:* trustcheck caught this repo's own 3.0x-vs-0.65x as noise.
4. **The GPU-necessity boundary is estimable from local data alone** *(NEW — not yet proven)* —
   the central claim this project must now test and publish.

## 4. The independent variables for the decision rule

The proposed boundary model, to be tested:

```
GPU warranted  iff:  f(overhead_fraction, model_size, batch_size, latency_target) crosses a threshold
   overhead_fraction: of decode wall-time that is NOT matmul/kernel (this repo already measures it)
   model_size:        params / weights that exceed CPU memory bandwidth (ratio of model-bytes to mem BW)
   batch_size:        server-style batching changes the regime
   latency_target:    the user's required p50/p99 — an edge scheduler tolerates higher latency
```

Rule shape (hypothesis): **low overhead-fraction AND high batch AND tight latency → GPU may help.
Otherwise → best-scheduled CPU config suffices; document the spend as unwarranted.**

## 5. What is explicitly NOT claimed (boundaries)

- Not a replacement or rival for vLLM / TensorRT / llama.cpp.
- Not a guarantee that CPU beats GPU on GPU-class workloads.
- Not "all inference steps automatic" — that remains the long-term roadmap, not today.
- Decode/CPU results are measured on specific hardware (2019 Intel 6C/12T CPU-only); GPU side of the
  boundary must be validated independently, not invented.

## 6. The success test (falsification, frozen before building)

A reviewer must be able to reproduce, from `benchmarks/` evidence on a CPU-only laptop:

1. Re-run the overhead-boundness and trustcheck claims (already hashed).
2. See a labeled decision boundary: for a given model at a given batch/latency target,
   inferlast returns **GPU-warranted / CPU-suffices / insufficient-data** with the reason,
   derived from local measurement — and the "insufficient-data" case when it cannot yet say.

If inferlast cannot distinguish those three from local data, the central claim is falsified and we
record the negative result honestly rather than stretch the claim.

## 7. Immediate next moves (in order)

1. ✅ Freeze this spec (this document).
2. Get ONE real external user to run inferlast and post a report in Discussions (un-fakeable seed).
3. Implement the decision rule as a `gpucheck` capability: inputs = measured overhead fraction,
   model size, batch, latency target → outputs GPU-warranted / CPU-suffices / insufficient-data.
4. Package `trustcheck` to read any benchmark JSON (and a CI surface) — the future paid-tier anchor.

## 8. Evidence ledger (append-only)

- **2026-09-02 — gpucheck shipped.** Sub-claim 4 now has an implementation and
  tests: `src/gpucheck.py` labels GPU-warranted / CPU-suffices / insufficient-data
  from CPU-only inputs, per the design in section 4, and refuses to guess when
  data is missing (honesty invariant, tested). Evidence: `tests/test_gpucheck.py`
  (9 tests), README, PROGRESS. The decision **rule** is shipped; the *boundary's
  empirical validation* across more model/hardware combos is the open next step.

- **2026-09-03 — big-model boundary test + a found-and-fixed gpucheck bug.**
  Ran `Qwen2.5-7B` (Q4 GGUF via Ollama) on the same i7-9750H CPU: measured
  **decode 2884.7 ms/token (~0.3 tok/s), 4.7 GB footprint, 26 s load** (a model
  that needs ~28 GB in fp32 runs in 16 GB RAM via Q4). Then fed these real
  numbers to `gpucheck`: the pre-fix rule said **CPU-suffices even at a 200 ms
  target while the model was 14x over it**, because its `weight-bound` heuristic
  (`weight_stream_gbps >= 0.5*bw`) missed this too-slow-model case (streams only
  ~5.3 GB/s, under the 20.5 GB/s threshold). **Fixed:** added a latency-feasibility
  check — if *measured* decode exceeds the *target*, GPU-warranted regardless of
  regime, with an auditable `measured_vs_target_x`. Regression tests:
  `tests/test_gpucheck.py` (now 11 gpucheck tests, 57 total). This validated the
  thesis boundary against a real big-ish model and hardened the rule. Evidence:
  this entry, `src/gpucheck.py`, `tests/test_gpucheck.py`.

- **2026-09-03 — "what fits on a 6-core/16 GB laptop CPU" map researched.**
  Literature + local measurement answered the mission question ("anyone runs any
  size on CPU, no GPU"): quantization (Q4 GGUF) lets a 16 GB box run up to ~8B
  comfortably and ~14B for batch; 32B+ does not fit; disk-offload (mmap) is a
  load-time win, not a capacity win. Discovers our earlier 0.3 tok/s 7B reading
  is likely a swap/load artifact on a heavily-loaded box, not the hardware's
  truth (~6-9 tok/s expected). Map + levers + GPU line in
  `docs/CPU-FEASIBILITY-MAP.md`. One consequence for the thesis: the strongest
  CPU speed lever is **speculative decoding** (~1.7-2x), which `gpucheck`'s model
  does not yet count — a candidate extension.

Every claim added here must link to a file under `benchmarks/` and a passing test.