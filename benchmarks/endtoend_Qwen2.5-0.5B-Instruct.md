# Auto-Optimizer proof report
hardware: CPU (2019 MacBook Pro 16in i7-9750H, 16GB)   prompt: The capital of France is the city of

## 1. Bottleneck (from profile)
prefill throughput ~4.3 tok/s; wall 3.68s for 16 tokens.
per-category (% of measured leaf work): mlp 47.0%, attention 41.3%, norm 10.8%, lm_head 0.6%, embed 0.2%
overhead-vs-weight: 99% of prefill wall time and 99% of decode wall time is framework overhead on CPU. NOTE: this is CPU-specific; on a GPU the same model (with fused kernels) is typically weight/compute-bound, so the picture differs. Here it means: on CPU, tiny models are overhead-bound.

## 2. Decode
1220 ms/token (~0.82 tok/s). Single-token serial, latency-sensitive.

## 3. Quantization (fp32 vs int8, measured BEFORE/AFTER, averaged)
fp32 1306 ms/tok -> int8 873 ms/tok = 1.50x (worst repeat 1.23x), quality logit-cos 0.68, top-5 0.15.
suggested: KEEP fp32 — int8 degrades quality too much: logit-cos 0.68, top5 0.15 (need >= 0.65/0.6). Do NOT quantize.

## 4. Batching (latency vs throughput)
B | wall(ms) | tok/s | ms/req | tok/s/req
1 | 769 | 10.4 | 769 | 10.41
2 | 986 | 16.2 | 986 | 8.11
4 | 909 | 35.2 | 909 | 8.80
8 | 2851 | 22.4 | 2851 | 2.81
16 | 2678 | 47.8 | 2678 | 2.99
best batch for goal='throughput': B=16 -> 47.8 tok/s total.

## Honest takeaway
The model was never permanently modified; any optimization is advised ONLY if it provably helps.