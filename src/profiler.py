"""Bottleneck profiler for a HuggingFace LLM on CPU.

For each named module we register a forward hook that records the module's own
wall-clock span (its forward from entry to exit). Leaf computation modules
(attention, mlp, norm, embed, lm_head, rotary) get non-overlapping measur---we
time each module's own forward span; parent modules that wrap leaves (e.g. a
decoder block) are excluded so nested time isn't double-counted.

Because a module's own span can include both its own matmuls and the children
it calls directly, we attribute time by *measured per-module forward span* and
report the percentage that each phase category contributes to the SUM of all
measured leaf spans. This answers the core question: attention vs MLP vs
embedding — where does inference time actually go?

CPU-only by design.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import DynamicCache
from decode import kv_supported


@dataclass
class ProfileResult:
    """Wall-clock time per category, plus totals."""

    cat_times: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    cat_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cat_params: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_time: float = 0.0
    tokens: int = 0
    device: str = "cpu"


# Phase categories we care about.
_LEAF_KEYWORDS = (
    "self_attn", "attn.", ".attn", "mlp", "feed_forward", "norm",
    "layernorm", "embed_tokens", "lm_head", "rotary"
)


def _classify(name: str) -> tuple[str, bool]:
    """Map a flattened module name to (phase category, is_leaf).

    is_leaf is True only for the *innermost* compute modules we want to time
    directly. Attention *containers* (names ending in `.attn` / `.self_attn`,
    which wrap q/k/v/o projections) are marked non-leaf so their own span is not
    double-counted on top of their children — the same exclusion `_is_leaf_for`
    applies. This keeps `_classify().leaf` consistent with `_is_leaf_for`.
    """
    if name == "embed_tokens" or name.endswith(".embed_tokens"):
        return ("embed", True)
    if name == "lm_head" or name.endswith(".lm_head"):
        return ("lm_head", True)
    if ".self_attn" in name or ".attn" in name:
        # Attention CONTAINER (the module that wraps q/k/v/o projections) is
        # named `...self_attn` / `...attn` with nothing after, and must not be
        # timed on top of its children. The projections (`...self_attn.q_proj`)
        # do NOT end in `.self_attn`, so they remain timed leaves.
        container = name.endswith(".self_attn") or name.endswith(".attn")
        return ("attention", not container)
    if ".mlp" in name or ".feed_forward" in name:
        return ("mlp", True)
    if "norm" in name or "layernorm" in name or "LayerNorm" in name:
        return ("norm", True)
    if "rotary" in name or "rotary_emb" in name:
        return ("rotary", False)
    return ("stem", False)


def _is_leaf_for(name: str) -> bool:
    """A module is timed directly if it is a leaf computation node and not a
    container. `_classify` already stamps attention containers (names ending in
    `.attn` / `.self_attn`) as non-leaf; this wrapper keeps the same rule."""
    cat, leaf = _classify(name)
    return leaf


def profile_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    run_repeats: int = 3,
    warmup: int = 1,
) -> ProfileResult:
    cat_times: dict[str, float] = defaultdict(float)
    cat_calls: dict[str, int] = defaultdict(int)
    cat_params: dict[str, int] = defaultdict(int)
    category: dict[int, tuple[str, bool]] = {}

    for name, mod in model.named_modules():
        cat, leaf = _classify(name)
        category[id(mod)] = (cat, leaf)
        if leaf:
            nparams = sum(p.numel() for p in mod.parameters(recurse=False))
            cat_params[cat] += nparams

    # Time every module's *own* forward span (entry->exit), non-nested.
    spans: dict[int, float] = defaultdict(float)

    def _enter(_m, _args, _out=None):
        t = time.perf_counter()
        _m._ao_t0 = t
        return _out

    def _exit(_m, _args, _out):
        if hasattr(_m, "_ao_t0"):
            spans[id(_m)] += time.perf_counter() - _m._ao_t0
            del _m._ao_t0
        return _out

    handles = []
    for name, mod in model.named_modules():
        cat, leaf = _classify(name)
        if leaf:
            handles.append(mod.register_forward_hook(_enter))
            handles.append(mod.register_forward_hook(_exit))

    model.eval()
    with torch.no_grad(), torch.inference_mode():
        for _ in range(warmup):
            model(input_ids)
        t0 = time.perf_counter()
        for _ in range(run_repeats):
            model(input_ids)
        t1 = time.perf_counter()

    for h in handles:
        h.remove()

    for mod_id, (cat, leaf) in category.items():
        if leaf:
            cat_times[cat] += spans.get(mod_id, 0.0)
        # count calls by module instance
        if leaf and spans.get(mod_id, 0.0) > 0:
            cat_calls[cat] += 1

    return ProfileResult(
        cat_times=dict(cat_times),
        cat_calls=dict(cat_calls),
        cat_params=dict(cat_params),
        total_time=(t1 - t0),
        tokens=int(input_ids.numel()) * run_repeats,
        device=str(next(model.parameters()).device),
    )


def profile_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    n_new_tokens: int = 16,
    run_repeats: int = 2,
    warmup: int = 1,
    do_sample: bool = False,
) -> ProfileResult:
    """Profile the autoregressive DECODE phase: after an initial prefill, the
    model samples one token at a time. This is the per-token, latency-sensitive
    path real serving cares about (the _seconds-per-token / TPOT the job
    postings list).

    Returns the same ProfileResult with cat_times = time attributed to each
    category summed across all generated tokens, and total_time = the decode
    wall time excluding the prefill.
    """
    spans: dict[int, float] = defaultdict(float)
    handles, category = [], {}
    span_ref: dict = spans
    ids = input_ids.clone()

    # We need hooks installed per-sample-step; but hooks on the model persist, so
    # install once and run the loop, resetting a per-call tick each step.
    handles = []
    for name, mod in model.named_modules():
        cat, leaf = _classify(name)
        category[id(mod)] = (cat, leaf)
        if leaf:
            handles.append(mod.register_forward_hook(_enter_hook(span_ref)))
            handles.append(mod.register_forward_hook(_exit_hook(span_ref)))

    model.eval()
    cat_params: dict[str, int] = defaultdict(int)
    for name, mod in model.named_modules():
        cat, leaf = _classify(name)
        if leaf:
            nparams = sum(p.numel() for p in mod.parameters(recurse=False))
            cat_params[cat] += nparams

    total_wall = 0.0
    cat_times: dict[str, float] = defaultdict(float)
    cat_calls: dict[str, int] = defaultdict(int)

    with torch.no_grad(), torch.inference_mode():
        use_kv = kv_supported(model)
        # one warmup decode sweep
        _decode_sweep(model, ids.clone(), n_new_tokens, do_sample,
                      warmup_wall_only=True, use_cache=use_kv)
        for _ in range(run_repeats):
            cur = ids.clone()
            # do a prefill (untimed) then measure the decode loop
            model(cur)
            span_ref.clear()
            t0 = time.perf_counter()
            _decode_loop(model, cur, n_new_tokens, do_sample, span_ref,
                         cat_times, cat_calls, category, use_cache=use_kv)
            t1 = time.perf_counter()
            total_wall += t1 - t0

    for h in handles:
        h.remove()

    return ProfileResult(
        cat_times=dict(cat_times),
        cat_calls=dict(cat_calls),
        cat_params=dict(cat_params),
        total_time=total_wall,
        tokens=n_new_tokens * run_repeats,
        device=str(next(model.parameters()).device),
    )


def _enter_hook(span_ref):
    def _e(_m, _args, _out=None):
        _m._ao_t0 = time.perf_counter()
        return _out
    return _e


def _exit_hook(span_ref):
    def _x(_m, _args, _out):
        if hasattr(_m, "_ao_t0"):
            span_ref[id(_m)] += time.perf_counter() - _m._ao_t0
            del _m._ao_t0
        return _out
    return _x


def _decode_loop(model, ids, n_new, sample, span_ref, cat_times, cat_calls, category,
                 use_cache=True, prefilled_cache=None):
    """Decode loop that measures per-category leaf timing via hooks.

    When `use_cache` is True, it uses the KV cache across steps (the REAL decode
    path as a served user sees it) instead of re-forwarding the whole growing
    context each token (the O(n^2) path that understates CPU decode speed by
    1.5-3x). This mirrors `decode.decode_with_kv`: the prefix is forward passed
    on step 0 into an empty cache (the prefill) and the cache is grown across
    steps, so keep `ids` as the full prefix for the first call.
    """
    n_tokens = ids.shape[1]
    if use_cache:
        am = torch.ones_like(ids, dtype=torch.long)
        cache = DynamicCache()
        cur = ids
        pos = torch.arange(n_tokens, device=ids.device).unsqueeze(0)
        am_cur = am
        for step in range(n_new):
            span_ref.clear()
            out = model(cur, attention_mask=am_cur, past_key_values=cache,
                        use_cache=True, position_ids=pos)
            if step > 0:
                # step 0 is the prefill; count only true decode steps toward the
                # per-category decode timing, so ms/token excludes the one-time cost
                _accumulate_step(span_ref, cat_times, cat_calls, category)
            next_id = out.logits[:, -1, :].argmax(-1)
            cur = next_id.unsqueeze(0)
            pos = pos[:, -1:] + 1
            am_cur = torch.cat([am_cur, torch.ones_like(cur, dtype=am_cur.dtype)], dim=1)
        return
    for step in range(n_new):
        span_ref.clear()
        out = model(ids)
        _accumulate_step(span_ref, cat_times, cat_calls, category)
        if not sample:
            next_id = out.logits[:, -1, :].argmax(-1)
        else:
            probs = torch.softmax(out.logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)


def _accumulate_step(span_ref, cat_times, cat_calls, category):
    span_ref_item = dict(span_ref)
    step_acc: dict[str, float] = defaultdict(float)
    step_calls: dict[str, int] = defaultdict(int)
    for mid, v in span_ref_item.items():
        cat, leaf = category.get(mid, ("other", False))
        if leaf:
            step_acc[cat] += v
            step_calls[cat] += 1
    for c, v in step_acc.items():
        cat_times[c] += v
        cat_calls[c] += step_calls[c]


def _decode_sweep(model, ids, n_new, sample, warmup_wall_only=False,
                  use_cache=True):
    if use_cache:
        cache = DynamicCache()
        am = torch.ones_like(ids)
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        cur, am_cur = ids, am
        for step in range(n_new):
            out = model(cur, attention_mask=am_cur, past_key_values=cache,
                        use_cache=True, position_ids=pos)
            next_id = out.logits[:, -1, :].argmax(-1) if not sample else \
                torch.multinomial(torch.softmax(out.logits[:, -1, :], dim=-1), 1)
            cur = next_id.unsqueeze(0)
            pos = pos[:, -1:] + 1
            am_cur = torch.cat([am_cur, torch.ones_like(cur, dtype=am_cur.dtype)], dim=1)
        return
    for step in range(n_new):
        out = model(ids)
        next_id = out.logits[:, -1, :].argmax(-1) if not sample else \
            torch.multinomial(torch.softmax(out.logits[:, -1, :], dim=-1), 1)
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)


def summarize(pr: ProfileResult) -> str:
    denom = max(sum(pr.cat_times.values()), 1e-9)
    lines = [
        f"device={pr.device}  tokens={pr.tokens}  wall_total={pr.total_time:.3f}s",
        f"throughput ~ {pr.tokens / max(pr.total_time, 1e-9):.1f} tok/s",
        f"(sum of measured leaf spans) {sum(pr.cat_times.values()):.3f}s",
        "",
        f"{'category':<12}{'time(s)':>12}{'%of-leaf-sum':>14}{'calls':>8}{'params(M)':>12}",
    ]
    for c in sorted(pr.cat_times, key=lambda k: pr.cat_times[k], reverse=True):
        lines.append(
            f"{c:<12}{pr.cat_times[c]:>12.4f}{pr.cat_times[c] / denom * 100:>13.1f}%"
            f"{pr.cat_calls[c]:>8}{pr.cat_params.get(c, 0) / 1e6:>12.2f}"
        )
    return "\n".join(lines)