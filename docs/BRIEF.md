# BRIEF — LLM Inference Auto-Optimizer

## Original request (preserved verbatim, user's own words)

> "can we something like project for the llm inference or anything you think that is good or best"
>
> "through that we can make company what you think"
>
> "no i want that inference which automatically find the bottleneck in you model do all the quantization, prune, post-fill ... allll inference step automatically what you think ... it is just idea ... i want to also make something crazy"
>
> "people early said we not fine tune model in the laptop of 4 gb but that guy do ... see this what the fuck i am doing ... i also build something crazy"
>
> "i want that inference which automatically find the bottleneck, do all the quantization, prune ... automatically"
>
> "yes do ..."

## The verified, locked interpretation

Build an **Auto-Optimizer for LLM inference**: given a model + a description of the target hardware, it
(1) **profiles** where inference time actually goes (per-layer and per-phase),
(2) **auto-selects** optimizations — most critically quantization depth and batching — based on measured bottleneck,
(3) **applies** the chosen optimization,
(4) **proves** the outcome with a reproducible before/after benchmark table (throughput, latency/TTFT/TPOT, accuracy).

This was researched and VERIFIED against real 2026 inference-engineer job postings (Nebius, Hippocratic,
d-Matrix, Nava, MakerMaker). The project's core capabilities map 1-to-1 to explicitly stated responsibilities
(profile bottlenecks; implement/benchmark quantization; run engine/config comparisons; build reproducible
benchmark harnesses). It is therefore not just relevant but IS the core inference-engineer competency,
packaged as a demoable portfolio artifact that also seeds the larger "automate every inference step" vision.

## Goal and outcome

- A working, laptop-runnable auto-optimizer (Phase 1): bottleneck profiler + auto-quantization selector +
  auto-batching selector + reproducible before/after benchmark, running real small LLMs on CPU.
- A public, verified portfolio artifact under the user's GitHub identity (YuvrajSinghBhadoria2) producing
  measurable numbers (e.g. "INT8 gives 1.8x throughput at 0.3% accuracy loss").
- Foundation that can grow toward: pruning, dispatch, KV-cache, auto-kernel, and eventually a product/company.

## Constraints and resources

- Hardware: 2019 Intel MacBook Pro 16", i7-9750H (6C/12T), 16 GB RAM. **CPU-only. No GPU/CUDA.**
- Python 3.10-3.12 only for PyTorch (system python is 3.13; use uv-managed 3.11 env).
- Models that fit CPU RAM comfortably: Qwen2.5-0.5B/1.5B, Llama-3.2-1B, SmolLM2-135M.
- Quantization on CPU must avoid CUDA-only paths; prefer torchao / gemlite / bitsandbytes won't work on CPU —
  use torchao-int8/int4 or llama.cpp-compatible approaches. Confirm what actually runs before committing.
- Zero cost local experimentation is authorized. No external spending, compute rental, publishing, or public
  credential access without asking.

## Forbidden / out of scope (Phase 1)

- Do NOT promise "all inference steps automatic" as a day-one deliverable — that is the long-term roadmap,
  not Phase 1. Keep the claim precise and honestly scoped.
- Do NOT rebuild vLLM. This is the *decision-intelligence* layer (profile -> pick -> apply -> prove), which
  complements serving engines; it is not a re-implementation of one.
- No GPU kernel work (impossible on this hardware).

## Authority and amendments

Created by the user's explicit request ("yes do ..."). Doubling down on the user's own #54339 vLLM PR and this
auto-optimizer are independent lines; neither blocks the other. If a later instruction conflicts, it is an
amendment recorded here; the latest instruction controls.