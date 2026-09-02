"""trustcheck: is a measured benchmark "win" real?

Most inference benchmarks report *a* number, not whether the number can be
trusted. `trustcheck` closes that gap. It takes a before/after benchmark and
returns an evidence-backed verdict on the claimed win:

  REAL      -- the win is outside the noise band AND measured against a robust,
               relevant metric AND the claimed knob is actually read by the code.
  MARGINAL  -- the win exists but is (a) within the single-run noise band, or
               (b) only real on a brittle/irrelevant proxy, or (c) too small to
               matter given the spread. "Don't ship, re-measure."
  FALSE     -- the win is NOT real: it's within noise (single-run fluke), or it
               only exists on a brittle metric that can lie, or the knob tested
               is documented/validated but read by nothing.

Three distinct ways a benchmark can lie -- each caught here:

 1. SINGLE-RUN NOISE: one measurement isn't a measurement. Given repeated
      before/after samples we compute a confidence interval on the speedup; if
      it covers 1.0x the "win" is not reliably faster. (This repo itself once
      measured INT8 as 3.0x faster in one run and 0.65x slower in another.)
 2. BRITTLE / WRONG METRIC: comparing outputs by greedy top-1 token identity is
      an extremely brittle quality proxy -- one early precision shift cascades
      and makes outputs diverge wildly, so it overstates real degradation for
      INT8. `trustcheck` flags a known-losing proxy and, when logits are given,
      re-measures with a robust proxy (per-token logit cosine + top-5 overlap).
 3. CONFIG READ BY NOTHING: a flag that is documented, validated, and then never
      read by any code -- the "validated, documented, read by nothing" class of
      bug that makes a tuned knob change a config fingerprint and nothing else.
      Static analysis flags documented config keys with no read in the codebase.

The output is a verdict + a reason, never just more numbers. CPU-only.
"""

from __future__ import annotations
import ast
import math
import os
import statistics
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# --- noise / confidence interval on a measured speedup -----------------------

def speedup_scale_pair(controls: list[float], treatments: list[float]) -> float:
    """Median speedup across paired runs, where LOWER samples = better (e.g.
    ms/token latency). speedup = control/treatment, so 1.0 = no change and >1.0 =
    the 'after' configuration is faster."""
    if not controls or not treatments:
        return float("nan")
    ratios = [c / t for c, t in zip(controls, treatments)]
    return float(statistics.median(ratios))


def _t_score(n: int) -> float:
    """Two-sided ~95% Student-t critical value for n samples (small-n safe)."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return table.get(n, 1.96)


def speedup_ci(controls: list[float], treatments: list[float],
               alpha: float = 0.05) -> tuple[float, float]:
    """95% CI on the true speedup, via the distribution of per-run speedup
    ratios (LOWER-is-better samples; speedup_i = controls[i]/treatments[i], so
    speedup > 1.0 = after is faster). Safe to use when the paired samples share
    the same noisy machine conditions.

    Returns (lo, hi) for the true speedup. A CI straddling 1.0 means the apparent
    win is not reliably distinguishable from noise.
    """
    n = min(len(controls), len(treatments))
    if n == 0:
        return (float("nan"), float("nan"))
    ratios = [c / t for c, t in zip(controls[:n], treatments[:n])]
    mean_r = statistics.fmean(ratios)
    if n == 1:
        return (mean_r, mean_r)
    sd = statistics.pstdev(ratios) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    tcrit = _t_score(n)
    return (mean_r - tcrit * se, mean_r + tcrit * se)


def classify_speedup(lo: float, hi: float, real_floor: float = 1.15,
                     marginal_floor: float = 1.05) -> tuple[str, str]:
    """Label a speedup CI as REAL / MARGINAL / FALSE based on whether the whole
    interval (or its low end) clears the noise band."""
    if math.isnan(lo) or math.isnan(hi):
        return "FALSE", "cannot classify: no repeated measurements"
    if lo >= real_floor:
        return "REAL", (
            f"speedup CI [{lo:.2f}x, {hi:.2f}x] clears {real_floor}x even at its "
            f"low end -> reliably faster (assuming a relevant metric)."
        )
    if hi < 1.0:
        return "FALSE", (
            f"speedup CI [{lo:.2f}x, {hi:.2f}x] is strictly below 1.0x -> the "
            f"configuration is reliably slower, not faster."
        )
    if hi >= marginal_floor:
        return "MARGINAL", (
            f"speedup CI [{lo:.2f}x, {hi:.2f}x] straddles 1.0x and includes "
            f">{marginal_floor}x -> appears faster but within single-run noise; "
            f"NOT a reliable win. Re-measure with more repeats."
        )
    return "FALSE", (
        f"speedup CI [{lo:.2f}x, {hi:.2f}x] never clears {marginal_floor}x and "
        f"straddles 1.0x -> no reliable win. Re-measure or abandon."
    )


# --- robust vs brittle quality metrics ----------------------------------------

KNOWN_BRITTLE_HINTS = (
    "token.match", "token_match", "== b", "greedy", "identical", "exact.equal",
    "acc", "accuracy", "output_match", "tok_match",
)


def flag_brittle_metric(metric_name: str = "",
                        comparison_level: str = "token",
                        uses_logits: bool = False) -> tuple[bool, str]:
    """Returns (is_brittle, reason). Flags quality comparisons that operate at a
    token/string level (or on an opaque accuracy) WITHOUT logits -- the class of
    metrics that can lie about INT8/F16 quality."""
    name = (metric_name or "").lower()
    if uses_logits:
        return False, "comparison uses logits -> measurable, not brittle."
    if comparison_level == "token" or any(h in name for h in KNOWN_BRITTLE_HINTS):
        return True, (
            f"quality compared at token/string level (or opaque accuracy) with no "
            f"logits available -> this proxy can lie (one precision shift cascades)."
        )
    return False, "no brittle-metric signature detected."


def robust_logit_quality(fp_logits: torch.Tensor, iq_logits: torch.Tensor):
    """Per-token logit cosine + top-5 overlap (the robust proxies)."""
    cos = F.cosine_similarity(fp_logits, iq_logits, dim=-1).clamp(0.0, 1.0)
    mean_cos = cos.mean().item()
    fp_top5 = fp_logits.topk(5, dim=-1).indices
    iq_top5 = iq_logits.topk(5, dim=-1).indices
    overlap = []
    for a, b in zip(fp_top5, iq_top5):
        overlap.append(len(set(a.tolist()) & set(b.tolist())) / 5)
    return mean_cos, float(sum(overlap) / len(overlap))


# --- "read by nothing" static analysis ---------------------------------------

def find_doctype_paras(docstring: str) -> list[str]:
    """Extract (roughly) config/flag names mentioned in a docstring, for the
    static check: a documented-but-unread knob is a liability, not a feature."""
    names = []
    for tok in docstring.replace("\n", " ").replace(",", " ").split():
        tok = tok.strip("`\"'().:")
        if tok and (tok.startswith("--") or "." in tok or tok.startswith("cfg")
                    or tok in ("stream_layers", "double_quant", "use_cache")):
            names.append(tok)
    return sorted(set(names))


def flag_read_by_nothing(config_key: str, source_root: str) -> tuple[bool, str]:
    """Returns (unread, reason): True if `config_key` appears in docstrings/docs
    but no source module ever reads it as a parameter or attribute."""
    unread = True
    usage = []
    for root, _dirs, files in os.walk(source_root):
        if any(seg in root for seg in (".venv", "__pycache__", ".git", "tests")):
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path) as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                # attribute access  cfg.foo  (a read) and  getattr / kwargs reads
                if isinstance(node, ast.Attribute) and node.attr == config_key:
                    unread = False
                    usage.append(f"{fn}:{getattr(node, 'lineno', '?')}")
                if isinstance(node, ast.Name) and node.id == config_key:
                    unread = False
                    usage.append(f"{fn}:{getattr(node, 'lineno', '?')}")
    if unread:
        return True, (
            f"config key '{config_key}' is documented but read by NO code path -> "
            f"changing it alters a fingerprint and nothing else (a real silent bug)."
        )
    return False, f"config key '{config_key}' is read (e.g. {', '.join(usage[:3])})."


# --- top-level verdict --------------------------------------------------------

@dataclass
class TrustVerdict:
    speed_verdict: str            # REAL / MARGINAL / FALSE
    speed_reason: str
    metric_brittle: bool
    metric_reason: str
    robust_cos: float | None      # populated if logits given
    robust_top5: float | None
    unread_keys: list[str]
    verdict: str                  # overall REAL / MARGINAL / FALSE
    reason: str


def audit(controls: list[float], treatments: list[float],
          source_root: str | None = None,
          metric_name: str = "",
          comparison_level: str = "token",
          fp_logits: torch.Tensor | None = None,
          iq_logits: torch.Tensor | None = None,
          config_key: str | None = None,
          real_floor: float = 1.15, marginal_floor: float = 1.05,
          ) -> TrustVerdict:
    """Audit a before/after benchmark and return a verdict on whether the claimed
    win is trustworthy. Feed it:
      controls    -- repeated 'before' samples (e.g. ms/token or latency)
      treatments  -- repeated 'after' samples
      source_root -- to run the read-by-nothing check (optional)
      metric_name / comparison_level -- to assess the quality proxy
      fp_logits / iq_logits -- to re-measure a robust quality proxy (optional)
      config_key  -- a knob to check is actually read (optional)
    """
    los, hi = speedup_ci(controls, treatments)
    speed_v, speed_r = classify_speedup(los, hi, real_floor, marginal_floor)

    brittle, metric_r = flag_brittle_metric(
        metric_name, comparison_level, uses_logits=fp_logits is not None)

    robust_cos = robust_top5 = None
    if fp_logits is not None and iq_logits is not None:
        robust_cos, robust_top5 = robust_logit_quality(fp_logits, iq_logits)

    unread_keys = []
    if config_key and source_root:
        unread, _ = flag_read_by_nothing(config_key, source_root)
        if unread:
            unread_keys.append(config_key)

    parts = [f"speed: {speed_v}."]
    if brittle:
        parts.append("metric: BRITTLE." + metric_r)
    if robust_cos is not None:
        parts.append(
            f"robust quality (logit-cos={robust_cos:.2f}, top-5={robust_top5:.2f}).")
    if unread_keys:
        parts.append(f"unread knob(s): {', '.join(unread_keys)}.")

    reason = " ".join(parts)
    verdict = _overall(speed_v, brittle, unread_keys)
    return TrustVerdict(speed_v, speed_r, brittle, metric_r, robust_cos,
                        robust_top5, unread_keys, verdict, reason)


def _overall(speed_verdict: str, brittle: bool, unread_keys: list[str]) -> str:
    """A win is only REAL if it is a real speedup AND the metric can't lie AND no
    tested knob is documentation-only. Any hard FALSE in those -> FALSE."""
    if brittle:
        return "FALSE"
    if unread_keys:
        return "FALSE"
    return speed_verdict  # REAL / MARGINAL / FALSE


def summarize_trust(v: TrustVerdict) -> str:
    return (
        f"TRUST VERDICT: {v.verdict.upper()}\n"
        f"  speed:     {v.speed_verdict}  ({v.speed_reason})\n"
        f"  metric:    {'BRITTLE - can lie' if v.metric_brittle else 'robust'}  ({v.metric_reason})\n"
        + (f"  robust q:  logit-cos {v.robust_cos:.2f}, top-5 {v.robust_top5:.2f}\n"
           if v.robust_cos is not None else "")
        + (f"  unread:    {', '.join(v.unread_keys)}\n" if v.unread_keys else "")
        + f"  reason:    {v.reason}"
    )