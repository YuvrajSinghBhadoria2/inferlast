# Optimization techniques: what inferlast ships now, and the honest gate for each

**Status:** RESEARCH + ENGINEERING (2026-09-03). Every technique below is mapped
to an honest gate: **SHIPPED** (implemented + measured in this repo), **SHIPPED/MEASURED**
(implemented, real-hardware number), **DEFERRED-EXPLAINED** (not shippable on this
stack, reason given), or **EXTERNAL** (only achievable by switching engines/tools —
inferlast reports the decision, it does not fake the work). Numbers tagged
`[MEASURED]` were taken on this repo's own hardware/stack; nothing invented.

The principle that governs every row: **a tool that reports a speedup must measure
the path a real user actually runs, on the real stack.** If inferlast cannot
honestly exercise a technique on torch 2.2.2 / CPU, it says so and tells the user
which engine can — rather than reporting a fake number.

## Summary table

| Technique | Gate | Evidence on this stack |
|---|---|---|
| KV cache (`use_cache`) | **SHIPPED/MEASURED** | 1.7-2.0x decode `[MEASURED]` |
| Speculative (assisted) decode | **SHIPPED/MEASURED (measurement)** | **0.55x — SLOWER** `[MEASURED]` |
| Dynamic batching / batch sweep | **SHIPPED** | est. max-tput batch `[MEASURED]` |
| int8 dynamic quantization (torch.ao) | **SHIPPED** | before/after measured, gated by quality |
| GGUF Q4_K_M | **EXTERNAL** (engine) | needs llama.cpp/Ollama torch, not PyTorch |
| KV-cache quantization (q8_0/q4_0 K/V) | **EXTERNAL** (engine) | needs llama.cpp `--cache-type` |
| AQT / int4 weight-only (torchao) | **DEFERRED-EXPLAINED** | needs torch≥2.3 + torchao; not installable here |
| FlashAttention/Attention | **DEFERRED-EXPLAINED** | needs cuDNN/GPU (CUDA-only) |
| Engine choice (llama.cpp vs torch) | **SHIPPED (decision)** | `gpucheck`/report labels the honest engine line |
| Thread count / `torch.set_num_threads` | **SHIPPED** | used in measurement |

## Shipped now (implemented in this repo)

### KV cache (`use_cache`)
**Gate: SHIPPED/MEASURED.** The decode path now uses `DynamicCache` + explicit
`position_ids` (modern, non-deprecated HF API). This is the single largest CPU
decode win: instead of re-forwarding the whole growing context each token (the
O(n²) model most ad-hoc CPU decoders use), each decode step reuses cached
keys/values and forwards one token.

- `[MEASURED]` Qwen2.5-0.5B, this i7-9750H, fp32: **KV 225.5 ms/tok (4.43 tok/s) vs
  no-cache 447.1 ms/tok (2.24 tok/s) → 1.98x**. Full `run_auto_optimizer` run:
  **1.66x** (259.6 vs 430.9 ms/tok).
- **Correctness gate passed:** KV-cache and recompute outputs are bit-identical
  (`kv == recompute: True`), and output matches HF `generate()` when `position_ids`
  are maintained — Qwen-family models diverge without this (verified).
- Not usable on the `MiniLM` test fixture (its `forward` has no cache API), so
  support is auto-detected via `kv_supported()` and the tool falls back to the
  recompute path for such models without crash. Honest: the fallback measures the
  O(n²) path, so numbers for non-HF models are labeled for what they are.

### Dynamic batching / batch sweep
**Gate: SHIPPED.** `best_batch` sweeps batch sizes and reports the throughput-max
batch on the current hardware rather than assuming one. Batch strategy is the only
throughput lever that scales on CPU; the tool measures and chooses, it does not guess.

### int8 dynamic quantization (torch.ao)
**Gate: SHIPPED.** fp32 vs int8 measured **before/after on the same hardware**,
with a frozen quality gate: quantization is only recommended when the logit-cosine
(semantic agreement) clears the floor and the speedup clears the floor. If quality
drops below the floor, inferlast says "quantization is NOT recommended here" — it
does not force the change.

## External (only achievable by switching engine/tools — inferlast reports, not fakes)

### GGUF Q4_K_M
**Gate: EXTERNAL (engine).** Q4 GGUF lives in llama.cpp/Ollama, not PyTorch. This is
the mechanism behind the CPU-FEASIBILITY-MAP (≤3B Q4 interactive, ≤8B Q4 batch).
inferlast's CPU-first philosophy names it and `gpucheck` labels the size/speed line;
it cannot run Q4 GGUF on torch 2.2.2 itself, and must not pretend to.

### Speculative decoding
**Gate: SHIPPED/MEASURED (measurement) — and measured SLOWER on this stack.**
inferlast now measures assisted decode honestly via `src/speculative.py`
(`inferlast run --draft-model ...`), instead of trusting the literature's ~1.7-2x
CPU claim. `[MEASURED]` on this hardware: SmolLM2-135M drafting Qwen2.5-0.5B =
**0.55x (SLOWER)** — because on bandwidth-bound CPU the draft and target both
stream their weights through the same memory bandwidth, and a separate HF draft
does not share an optimized kernel path. The literature win is engine- and
scale-dependent (llama.cpp draft shares loaded weights/kernels); it does NOT
transfer to a naive two-HF-model torch-2.2.2 setup. inferlast reports the real
number and a FASTER/SLOWER/FLAT verdict — this is the honest measurement that
pre-empts the classic false "speculative = free speedup" claim.

### KV-cache quantization (q8_0/q4_0 K/V)
**Gate: EXTERNAL (engine).** Shrinks long-context KV RAM so a bigger model/longer
context fits in 16 GB. A llama.cpp `--cache-type` flag, not a torch operation.
inferlast documents the lever and the resulting feasibility (see
CPU-FEASIBILITY-MAP) rather than implementing a partial torch stub.

### FlashAttention / fused SDPA attention
**Gate: DEFERRED-EXPLAINED.** FlashAttention is a CUDA/cuDNN/GPU primitive; on this
CPU-only stack it does not apply. Named and honestly excluded, not stubbed.

### AQT / int4 weight-only (torchao)
**Gate: DEFERRED-EXPLAINED.** torchao's AQT int4 needs **torch≥2.3 + torchao
installed**, and the int4 weight-only kernels are primarily CUDA/torchao-experimental
— not portable to this 2019 MacBook's torch 2.2.2 / CPU. Attempting it here (as the
project earlier discovered) fails at install and is GPU/advanced anyway. The honest
action is **defer and record**, which is what the feasibility research did, instead
of shipping a broken int4 path. If the user later runs torch≥2.3 with torchao on a
CUDA box, AQT becomes a new project.

### Thread count (`torch.set_num_threads`)
**Gate: SHIPPED.** Used in measurement so the benchmark reflects a pinned thread
count, not whatever the OS happened to give.

## How the reporter surfaces all of this

`run_auto_optimizer` now emits a `techniques.kv_cache` slot (avail / measured
ms-per-token on KV vs no-cache path / speedup / honest note) and a
`techniques.speculative` slot (off unless a `--draft-model` is supplied; FASTER /
SLOWER / FLAT verdict from real measurement). The markdown report renders a
"2b. Optimization techniques measured (KV cache)" section and a "2c. Speculative
(assisted) decoding" section when a draft is provided. When a model
has no KV-cache API, `avail: False` is reported with the reason — never a silent or
fabricated number.

## Bottom line

inferlast **ships** the techniques it can honestly exercise on CPU/torch 2.2.2 — KV
cache (measured win), batch sweep, int8 quantization with a quality gate — and it
**names, explains, and defers** the GPU/engine-bound ones (AQT int4, FlashAttention,
GGUF, speculative, KV-quant) instead of reporting numbers it cannot produce. That is
the whole point of a CPU-first tool whose value is *measured-not-claimed*.
