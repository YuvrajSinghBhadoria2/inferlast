# DECIDE → DEPLOY SPEC — inferlast as the one-stop answer to the inference-cost decision

## 1. The core problem this closes (research-grounded, not invented)

LLM inference spend is now the dominant AI cost and it is a *permanent, recurring*
line item, unlike training. Industry data and peer-reviewed work converge on this:

- "Inference now accounts for roughly two-thirds of all AI compute in 2026, having
  overtaken training." — Spheron GPU decision guide, 2026.
- "Inference spend overtakes training spend within months of launch." — Leaseweb
  EPYC CPU-vs-GPU benchmark, 2026.
- "Serving single requests wastes ~90% of GPU capacity on memory transfers;
  batching 32 requests reduces per-token costs ~85% while raising latency only
  ~20%." — Introl cost-per-token analysis, 2026.

The industry consensus is that the CPU-vs-GPU call is **workload-dependent**, not a
fixed rule: it depends on model size, traffic, latency target, batch size, and
budget. The break-even point for when GPU throughput justifies its cost is
estimated around 5–10 sustained queries/sec (inferensys) and around 7B parameters
for when CPUs "run well enough" (aimadetools, markaicode, sitepoint).

**The gap:** the serving *engines* (llama.cpp, Ollama, vLLM, BentoML's llm-optimizer,
VRAM calculators, GPU recommenders) either serve, benchmark on an *already-acquired*
GPU, or recommend *which* GPU — but none of them answer the prior question: **"should
I rent/keep a GPU at all, given MY model, MY latency target, MY traffic — measured
on MY machine, without paying to find out?"** BentoML's `llama-bench`-style guidance
is explicit that tok/s is only meaningful measured on *your* hardware.

**inferlast's thesis:** this decision AND the resulting deployment config belong in
one honest, measured, CPU-only tool. `inferlast run` measures; `inferlast deploy`
turns the verdict into a ready-to-run serving config. That is the "one-stop" answer:
**measure → decide → deploy → (still) prove.**

## 2. The decision rule (already built, now fed forward)

`gpucheck.decide(...)` already returns an auditable `GpuSuggestion`:

- `CPU-suffices` — workload is overhead-bound/small/loosely-timed; the
  best-scheduled CPU config is the cheaper, defensible default.
- `GPU-warranted` — workload is weight/compute-bound or the latency target beats
  the measured CPU decode; a GPU can plausibly give a real, non-noise win.
- `insufficient-data` — cannot decide; refuses to guess (honesty feature).

Peer-reviewed support for the empirical, per-workload style of this rule:
- **Batching Configuration Advisor (BCA)**, arXiv:2503.08311, "Mind the Memory Gap":
  batch size has a *measured throughput plateau* past the knee point; the optimal
  batch must be chosen per workload against a latency SLO. This is exactly
  inferlast's measured batch sweep, not a one-size rule.
- **"Optimizing LLM Inference Throughput via Memory-aware and SLA-constrained
  Dynamic Batching,"** arXiv:2503.05248: batch size must adapt to memory + SLA; a
  static-default is wrong.

## 3. What `inferlast deploy` emits (one command, two branches)

Inputs (all optional, all measured-or-declared):
- `--model` (HF id, required)
- `--latency-target-ms` (user's SLO per token, default 1000 = loose)
- `--batch` (server-style batch, default 1)
- `--req-per-hr` (declared traffic, default none)
- `--no-exec` (default): emit config without running it.

Output — a single deploy report that is **decision + artifact + honesty note**:

1. **Decide:** GPU-warranted / CPU-suffices / insufficient-data, with the full
   auditable reason from `gpucheck`.
2. **Deploy (CPU branch):** a ready-to-run `llama.cpp` `llama-server` command —
   pure CPU (`-ngl 0`), GGUF **Q4_K_M** (the researched-sweet-spot quant), threads
   from core count, KV-cache space, host/port, context. This is the industry's
   "best CPU stack" (aimadetools, markaicode, sitepoint): llama.cpp with AVX2,
   GGUF quantization, memory-mapped files.
3. **Deploy (GPU branch):** a ready-to-run **vLLM** command (the industry-default
   high-throughput GPU server), with `--gpu-memory-utilization`, max-num-seqs, and
   dtype guidance.
4. **Cost sanity (honest, decision-rule not fake numbers):** states the *rule* —
   CPU is the low-fixed-cost choice for low-volume / loose-latency / small-model
   workloads and becomes break-even-unfavorable roughly past ~5–10 sustained
   req/s or ~7B+ params — and always says "re-measure on YOUR target hardware
   before committing." No invented per-token prices are fabricated; where a
   ballpark is given it is cited as a reported range, not measured by inferlast.
5. **Prove:** reminds the operator to run the emitted server and re-check the
   measured tok/s against the target (ties back to `inferlast audit`).

If `--out` is given, writes `deploy.json` (the structured decision) plus a
`run.sh` (the executable snippet) to that directory — a single deploy artifact a
platform/CI user can inspect and run.

## 4. Honesty and anti-claim guardrails

- inferlast **never claims to have measured a GPU**. It estimates a *regime* and
  labels CPU-suffices / GPU-warranted / insufficient-data. GPU-warranted means
  "a GPU can plausibly win, verify on GPU" — not "here is 3x."
- Every config is the *researched default* for its branch and is clearly labeled:
  "generated from measured verdict; verify tok/s on your target hardware."
- No fabricated numbers anywhere: cost guidance is a cited decision rule, never a
  masqueraded measurement.

## 5. Sources (all consulted; none invented)

- arXiv:2503.08311 — Batching Configuration Advisor; throughput plateau vs SLO.
- arXiv:2503.05248 — memory-aware / SLA-constrained dynamic batching.
- Spheron 2026 GPU decision guide — inference ≈ two-thirds of AI compute.
- Leaseweb AMD EPYC CPU-vs-GPU benchmark, 2026 — CPU right for batch/async/TTS;
  cost-per-token math.
- Introl cost-per-token analysis, 2026 — batching amortizes GPU cost.
- inferensys small-model cost-benefit (2026) — CPU-vs-GPU break-even ~5–10 req/s.
- aimadetools / markaicode / sitepoint "when to use CPU vs GPU" (2026) — CPU
  "best stack" = llama.cpp + GGUF Q4_K_M + `-ngl 0`; ~7B CPU sufficiency line.
- Red Hat "The CPU is back" (2026) — concrete CPU-serving config knobs
  (`VLLM_CPU_KVCACHE_SPACE`, thread pinning) — the knobs this spec exposes.
