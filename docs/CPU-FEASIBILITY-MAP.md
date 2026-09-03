# What fits on a 6-core / 16 GB CPU-only laptop (and the honest GPU line)

**Status:** RESEARCH FINDING (2026-09-03). Evidence map for inferlast's CPU-first
mission: *"anyone can run inference on CPU, no GPU — down to what model size, at
what speed, and where you genuinely need to rent a GPU."* Numbers tagged
`[EST]` (bandwidth-extrapolated) vs `[MEASURED]` (on real hardware). Nothing here
is invented; sources for the literature numbers are cited in the research agent
report, and the local caveats are spelled out.

## TL;DR

- **Comfortably usable tier:** **≤ 3B at Q4_K_M** (~2 GB), ~15-22 tok/s `[EST]`.
  This is a laptop's daily-driver for interactive CPU inference. Llama 3.2 3B /
  Qwen2.5 3B / Qwen3 4B all fit in ~3-4 GB runtime.
- **Stretch-but-works tier:** **7-8B at Q4_K_M** (~4.7-4.9 GB), ~6-9 tok/s `[EST]`
  on this 6-core/12-thread i7-9750H (dual-channel DDR4-2666 ~42 GB/s). Usable for
  patient chat / batch / "leave it running" work, not real-time.
- **Hard ceiling in 16 GB RAM:** **14B Q4 (~9 GB)** fits but runs ~3-4.5 tok/s
  `[EST]` — batch/offline only. **32B (20 GB) does NOT fit.** **70B is out.**
- **Disk-offload (mmap) is a load-time win, NOT a capacity win.** You cannot run a
  model bigger than RAM at usable speed by mapping it from disk — beyond RAM the
  OS swaps and inference collapses to ~0.02-0.05 tok/s (unusable). The "70B on a
  laptop" stories are distributed clusters, not single-machine mmap.

## Why this is honest about "any size on any CPU"

The hard physical limit is **RAM, not CPU speed**: a model needs
`params × bytes-per-param` resident to run at usable speed.
- fp32 = 4 B/param → 7B ≈ 28 GB (won't fit 16 GB)
- fp16 = 2 B/param → 7B ≈ 14 GB (marginal)
- **Q4_K_M ≈ 0.55 B/param → 7B ≈ 4.6 GB, 14B ≈ 9 GB, 32B ≈ 20 GB** (fits / fits-tight / doesn't fit)

So quantization is the lever that makes "everyone runs LLMs on CPU" *mostly* true
for the sizes people actually want (up to ~14B on a 16 GB box). Containers/Docker
do not help — a container shares host RAM; it cannot create it. The ceiling is
memory, and quantization + fitting-in-RAM is how you push it.

## Feasibility map (Q4_K_M)

| Tier | Q4_K_M size | fp16 (scale) | Fits 16 GB? | Realistic tok/s on i7-9750H `[EST]` |
|---|---|---|---|---|
| ~1B | 0.81 GB | ~2.5 GB | ✅ very comfy | ~35-50 |
| 3B | ~2.0 GB | ~6.4 GB | ✅ very comfy | ~15-22 |
| 7-8B | 4.68-4.92 GB | ~15 GB | ✅ fits (+KV headroom) | ~6-9 |
| 13-14B | 7.9-8.99 GB | ~26-29.5 GB | ⚠️ fits, tight | ~3-4.5 |
| 32-34B | ~19.9 GB | ~65 GB | ❌ exceeds RAM | ~1.5-2 |
| 70B | ~42.5 GB | ~130 GB | ❌ way over | 0.5-1.5 (even in-RAM on big machines) |

`[EST]` decode is bandwidth-bound: `~w=model-bytes/token ÷ ~42 GB/s (this CPU)`,
anchored to published CPU measurements on dual-channel DDR4/5 systems (e.g. 8B Q4
= 11-12 tok/s on high-bandwidth DDR5 desktops, scaled to ~half that bandwidth).

## Local evidence and a caveat about it

- `[MEASURED]` on this machine via Ollama: `qwen2.5:7b` (Q4, 4.7 GB) decode was
  **2884.7 ms/token (~0.3 tok/s)** — but the machine was under heavy load/swap
  (11.2 GB of 12 GB swap in use; load average ~15 on a 12-thread CPU) at the time.
  That reading is **not a hardware truth**; it is ~10-20x below the ~6-9 tok/s
  this hardware class should achieve with a clean, idle system and a native/AVX2
  llama.cpp build. The research flagged this; the honest conclusion is: re-measure
  on an idle system before trusting any per-token figure for the map.
- A controlled re-measure (idle RAM, no competing load) was attempted but the
  background model pull put the box under sustained load, so the clean reading was
  not reliably obtainable in this session. This is a documented limitation, not a
  hidden number.

## The honest "you need a GPU" line (evidence-backed)

Rent/buy a GPU when you need **any** of:
1. reliable interactive chat **above ~10-20 tok/s with an 8B or larger**,
2. **a 14B+ model at any reasonable speed**,
3. **true batching / multi-user throughput** — CPU is bandwidth-bound and does not
   scale throughput with batch the way a GPU does,
4. **32B+ / 70B** — no single-CPU-in-16 GB path exists; even 70B in-RAM is
   0.5-1.5 tok/s on far bigger machines.

Until then, on a 6-core/16 GB box, the CPU-first answer is: a **3B Q4** for
interactive + an **8B Q4** for quality/batch, both GPU-free. This is exactly the
decision inferlast's `gpucheck` is meant to make from local measurement.

## Levers that stretch CPU-only further (Intel, this machine)

- **Speculative decoding** — the single highest-leverage CPU win `[MEASURED by
  others]`: CPU 3B got ~1.7-2x (e.g. 12.9 -> 22.1 tok/s) with a ~10x-smaller draft.
  Bandwidth-starved CPUs benefit most. This machine has AVX2 (9th-gen i7); it does
  NOT have AVX-512 (that needs Ice Lake+), so build `native`/AVX2.
- **Lower-bit quants (Q2/Q3/IQ)** to shrink footprint (7B Q2 ≈ 2.7 GB) and raise
  decode speed at some quality cost — a last resort to fit more in RAM.
- **KV-cache quantization** (q8_0/q4_0 K/V) to cut long-context KV RAM and run a
  bigger model / longer context within 16 GB.
- **BitNet / sub-2-bit models** — promising frontier (Microsoft BitNet 2B ~5x
  faster than a Q3 14B on the same old CPU), but toolchain immature on non-AVX2;
  not yet a safe daily driver.

## Bottom line

"Anyone can run inference on CPU of any size, no GPU" is *almost* true for the
sizes people actually want to run — **up to ~8B comfortably and ~14B for batch on
a modest 6-core / 16 GB laptop, using Q4 GGUF**. The hard ceiling is RAM; the
levers are quantization (Q4) and fitting-in-memory, plus speculative decoding for
speed. Beyond ~14B (or for true throughput), you genuinely need GPU — and that is
precisely the boundary `gpucheck` should label from local CPU measurement.
