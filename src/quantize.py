"""Auto-quantization selector with measured BEFORE/AFTER proof.

Strategy: dynamically quantize the linear (`nn.Linear`) modules of a HuggingFace
causal LM to int8 on CPU using torch's built-in `torch.ao.quantization.quantize_dynamic`
(FX/ready, no CUDA, no torchao). We then measure, for the SAME fixed prompt and
decode length:

  BEFORE  -> full fp32 model
  AFTER   -> int8-quantized model

Two things that matter to an inference engineer:
  * decode speed (ms/token, throughput)  -- averaged over multiple repeats so a
                                            busy CPU can't swing the verdict
  * output QUALITY, measured robustly    -- per-step logit cosine similarity and
                                            top-5 token-overlap, averaged over all
                                            generated tokens (NOT brittle greedy
                                            top-1 identity, which cascades and
                                            overstates real INT8 degradation).

The tool gives an HONEST verdict: recommend INT8 ONLY if averaged speedup is above
a floor AND the robust quality metric is preserved. It explicitly does NOT trust a
single noisy timing run.

V2 (audit fix): metric swapped from greedy-token-match to logit cosine + top-5
overlap; timing averaged over N repeats with the spread (min/max) reported.
"""

from __future__ import annotations
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import quantize_dynamic
from transformers import DynamicCache
from decode import kv_supported


@dataclass
class QuantVerdict:
    before: dict            # {"ms_per_token":, "tok_per_s":}
    after: dict             # {"ms_per_token":, "tok_per_s":}
    speedup: float          # after_throughput / before_throughput (<1 = slower)
    speedup_min: float      # worst-case speedup across repeats (honest spread)
    logit_cosine: float     # mean cosine similarity per generated token (0..1)
    top5_overlap: float     # mean fraction of top-5 tokens shared (0..1)
    recommended: bool
    reason: str


def _collect_logits(model, tok, prompt, n_new, use_cache=None):
    """Return (ms_per_token, per-step fp32 logits at last position, generated ids).

    Uses the KV-cache decode path (the real served path) by default so measured
    decode speed is not artificially slow; `use_cache=False` falls back to the
    O(n^2) recompute loop for A/B comparison. If `use_cache` is left None it is
    auto-detected from whether the model's forward accepts KV-cache kwargs."""
    if use_cache is None:
        use_cache = kv_supported(model)
    ids = tok([prompt], return_tensors="pt").input_ids
    am = torch.ones_like(ids, dtype=torch.long)
    model.eval()
    cur, am_cur, pos = ids, am, torch.arange(ids.shape[1]).unsqueeze(0)
    cache = DynamicCache() if use_cache else None
    logits_last = []
    t0 = time.perf_counter()
    with torch.no_grad(), torch.inference_mode():
        for _ in range(n_new):
            if use_cache:
                out = model(cur, attention_mask=am_cur, past_key_values=cache,
                            use_cache=True, position_ids=pos)
            else:
                out = model(cur) if not kv_supported(model) else model(cur, attention_mask=am_cur)
            logits_last.append(out.logits[:, -1, :].float().detach())
            nxt = out.logits[:, -1, :].argmax(-1)
            cur = nxt.unsqueeze(0) if use_cache else torch.cat([cur, nxt.unsqueeze(0)], dim=1)
            if use_cache:
                pos = pos[:, -1:] + 1
            am_cur = torch.cat([am_cur, torch.ones_like(cur, dtype=am_cur.dtype)], dim=1)
        dt = time.perf_counter() - t0
    gen = cur[:, ids.shape[1]:] if not use_cache else cur.clone()
    return _ms_per_token(dt, n_new), torch.cat(logits_last, dim=0), gen


def _ms_per_token(dt, n_new):
    return dt / max(n_new, 1e-9) * 1000.0


def quantize_model_int8(model):
    return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8, inplace=False)


def _logit_quality(fp_logits: torch.Tensor, iq_logits: torch.Tensor):
    """Return (mean cosine per token, mean top-5 overlap per token)."""
    # cosine per token
    cos = F.cosine_similarity(fp_logits, iq_logits, dim=-1)  # [n_new]
    mean_cos = cos.clamp(0.0, 1.0).mean().item()
    # top-5 overlap per token
    fp_top5 = fp_logits.topk(5, dim=-1).indices
    iq_top5 = iq_logits.topk(5, dim=-1).indices
    overlap = []
    for a, b in zip(fp_top5, iq_top5):
        overlap.append(len(set(a.tolist()) & set(b.tolist())) / 5)
    return mean_cos, float(sum(overlap) / len(overlap))


def auto_quantize(
    model,
    tok,
    prompt: str = "The capital of France is the city of",
    n_new: int = 10,
    repeats: int = 3,
    speedup_floor: float = 1.03,
    quality_floor: float = 0.65,   # mean cosine >= 0.65 required to recommend
    top5_floor: float = 0.60,      # and top-5 overlap >= 0.60
) -> QuantVerdict:
    """Averaged, quality-aware auto-quantization verdict.

    Quality is measured ONCE (logits are stable for greedy/non-sampling decode,
    so multiple repeats would give the same cosines); speed is averaged over
    `repeats` runs because a shared/busy CPU makes single timings unreliable.
    """
    m_fp = model
    m_q = quantize_model_int8(model)

    # Discard the first (cold/jit) timing of each model, then average.
    _ = _collect_logits(m_fp, tok, prompt, n_new)  # warm load
    _ = _collect_logits(m_q, tok, prompt, n_new)

    fp_mpt, fp_logits, _ = _collect_logits(m_fp, tok, prompt, n_new)
    iq_mpt, iq_logits, _ = _collect_logits(m_q, tok, prompt, n_new)
    logit_cos, top5_ov = _logit_quality(fp_logits, iq_logits)

    # Now average decode speed over `repeats` runs each.
    fp_times = [fp_mpt]
    iq_times = [iq_mpt]
    for _ in range(repeats - 1):
        fp_t, _, _ = _collect_logits(m_fp, tok, prompt, n_new)
        iq_t, _, _ = _collect_logits(m_q, tok, prompt, n_new)
        fp_times.append(fp_t)
        iq_times.append(iq_t)

    mpt_fp = sum(fp_times) / len(fp_times)
    mpt_iq = sum(iq_times) / len(iq_times)
    throughput_fp = 1000.0 / mpt_fp
    throughput_iq = 1000.0 / mpt_iq
    speedup = throughput_iq / throughput_fp

    # worst-case speedup across per-run throughput ratio
    ratios = [1000.0 / i / (1000.0 / f) for f, i in zip(fp_times, iq_times)]
    speedup_min = min(ratios)

    before = {"ms_per_token": mpt_fp, "tok_per_s": throughput_fp,
              "repeats": repeats, "ms_spread": (min(fp_times), max(fp_times))}
    after = {"ms_per_token": mpt_iq, "tok_per_s": throughput_iq,
             "repeats": repeats, "ms_spread": (min(iq_times), max(iq_times))}

    speed_ok, quality_ok, rec, reason = _decide(
        speedup, speedup_min, logit_cos, top5_ov,
        speedup_floor, quality_floor, top5_floor,
    )

    return QuantVerdict(before, after, speedup, speedup_min, logit_cos, top5_ov,
                        rec, reason)


def _decide(speedup, speedup_min, logit_cos, top5_ov,
            speedup_floor, quality_floor, top5_floor):
    """Pure, unit-testable decision rule: recommend INT8 ONLY if BOTH the speed
    floor and the quality floor hold. Anything that degrades quality is refused
    even if it is fast. Extracted so the 'honest refusal' logic can be tested
    without a clock."""
    speed_ok = speedup >= speedup_floor
    quality_ok = (logit_cos >= quality_floor and top5_ov >= top5_floor)

    if speed_ok and quality_ok:
        rec, reason = True, (
            f"int8 helps ({speedup:.2f}x avg, min {speedup_min:.2f}x), quality "
            f"preserved (cos {logit_cos:.2f}, top5 {top5_ov:.2f}). Recommend INT8."
        )
    elif not quality_ok:
        rec, reason = False, (
            f"int8 degrades quality too much: logit-cos {logit_cos:.2f}, "
            f"top5 {top5_ov:.2f} (need >= {quality_floor}/{top5_floor}). Do NOT quantize."
        )
    else:
        rec, reason = False, (
            f"int8 gives no real speedup ({speedup:.2f}x avg, min {speedup_min:.2f}x). "
            f"Overhead-bound; Do NOT quantize."
        )

    return speed_ok, quality_ok, rec, reason


def summarize_verdict(v: QuantVerdict) -> str:
    n = v.after["repeats"] if "repeats" in v.after else 1
    return (
        f"BEFORE (fp32): {v.before['ms_per_token']:.1f} ms/tok, "
        f"{v.before['tok_per_s']:.2f} tok/s "
        f"(n={v.before.get('repeats','?')}, spread {v.before['ms_spread'][0]:.0f}-"
        f"{v.before['ms_spread'][1]:.0f} ms)\n"
        f"AFTER  (int8): {v.after['ms_per_token']:.1f} ms/tok, "
        f"{v.after['tok_per_s']:.2f} tok/s "
        f"(n={n}, spread {v.after['ms_spread'][0]:.0f}-{v.after['ms_spread'][1]:.0f} ms)\n"
        f"speedup (after/before): {v.speedup:.2f}x  (worst repeat {v.speedup_min:.2f}x)\n"
        f"quality: logit-cosine {v.logit_cosine:.2f}, top-5 overlap {v.top5_overlap:.2f}\n"
        f"RECOMMENDATION: {'USE int8' if v.recommended else 'KEEP fp32'}\n"
        f"{v.reason}"
    )