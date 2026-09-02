# Auto-Optimizer proof report
hardware: CPU (2019 MacBook Pro 16in i7-9750H, 16GB)   prompt: The capital of France is the city of

## 1. Bottleneck (from profile)
prefill throughput ~8.7 tok/s; wall 1.83s for 16 tokens.
per-category (% of measured leaf work): mlp 43.9%, attention 41.8%, norm 13.7%, lm_head 0.4%, embed 0.2%
overhead-vs-weight: 98% of prefill wall time and 98% of decode wall time is framework overhead on CPU. NOTE: this is CPU-specific; on a GPU the same model (with fused kernels) is typically weight/compute-bound, so the picture differs. Here it means: on CPU, tiny models are overhead-bound.

## 2. Decode
830 ms/token (~1.21 tok/s). Single-token serial, latency-sensitive.

## 3. Quantization (fp32 vs int8, measured BEFORE/AFTER, averaged)
fp32 417 ms/tok -> int8 517 ms/tok = 0.81x (worst repeat 0.61x), quality logit-cos 0.24, top-5 0.27.
suggested: KEEP fp32 — int8 degrades quality too much: logit-cos 0.24, top5 0.27 (need >= 0.65/0.6). Do NOT quantize.

## 4. Batching (latency vs throughput)
B | wall(ms) | tok/s | ms/req | tok/s/req
1 | 616 | 13.0 | 616 | 12.98
2 | 468 | 34.2 | 468 | 17.09
4 | 797 | 40.1 | 797 | 10.03
8 | 1573 | 40.7 | 1573 | 5.08
16 | 1978 | 64.7 | 1978 | 4.04
best batch for goal='throughput': B=16 -> 64.7 tok/s total.

## Honest takeaway
The model was never permanently modified; any optimization is advised ONLY if it provably helps.