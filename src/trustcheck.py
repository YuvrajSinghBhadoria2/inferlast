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
import csv
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn.functional as F


# --- noise / confidence interval on a measured speedup -----------------------

DIRECTION_LOWER_IS_BETTER = "lower-is-better"
DIRECTION_HIGHER_IS_BETTER = "higher-is-better"


def speedup_scale_pair(controls: list[float], treatments: list[float],
                       direction: str = DIRECTION_LOWER_IS_BETTER) -> float:
    """Median speedup across paired runs. speedup > 1.0 always means the 'after'
    configuration is better on the given metric `direction`:
      - lower-is-better     (e.g. ms/token latency): speedup = control/treatment
      - higher-is-better    (e.g. tokens/sec, accuracy): speedup = treatment/control
    """
    if not controls or not treatments:
        return float("nan")
    if direction == DIRECTION_HIGHER_IS_BETTER:
        ratios = [t / c for c, t in zip(controls, treatments)]
    else:
        ratios = [c / t for c, t in zip(controls, treatments)]
    return float(statistics.median(ratios))


def _t_score(n: int) -> float:
    """Two-sided ~95% Student-t critical value for n samples (small-n safe)."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return table.get(n, 1.96)


def _speedup_ratios(controls: list[float], treatments: list[float],
                    direction: str) -> list[float]:
    n = min(len(controls), len(treatments))
    if direction == DIRECTION_HIGHER_IS_BETTER:
        return [t / c for c, t in zip(controls[:n], treatments[:n])]
    return [c / t for c, t in zip(controls[:n], treatments[:n])]


def speedup_ci(controls: list[float], treatments: list[float],
               direction: str = DIRECTION_LOWER_IS_BETTER,
               alpha: float = 0.05) -> tuple[float, float]:
    """95% CI on the true speedup, via the distribution of per-run speedup
    ratios. speedup > 1.0 always means 'after' is BETTER on the metric:
      - lower-is-better  -> speedup_i = controls[i]/treatments[i]
      - higher-is-better -> speedup_i = treatments[i]/controls[i]
    Safe to use when the paired samples share the same noisy machine conditions.

    Returns (lo, hi) for the true speedup. A CI straddling 1.0 means the apparent
    win is not reliably distinguishable from noise.
    """
    n = min(len(controls), len(treatments))
    if n == 0:
        return (float("nan"), float("nan"))
    ratios = _speedup_ratios(controls, treatments, direction)
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
            f"low end -> reliably {_better_than_phrase(real_floor)} (assuming a "
            "relevant metric)."
        )
    if hi < 1.0:
        return "FALSE", (
            f"speedup CI [{lo:.2f}x, {hi:.2f}x] is strictly below 1.0x -> the "
            f"configuration is reliably worse, not better."
        )
    if hi >= marginal_floor:
        return "MARGINAL", (
            f"speedup CI [{lo:.2f}x, {hi:.2f}x] straddles 1.0x and includes "
            f">{marginal_floor}x -> appears better but within single-run noise; "
            f"NOT a reliable win. Re-measure with more repeats."
        )
    return "FALSE", (
        f"speedup CI [{lo:.2f}x, {hi:.2f}x] never clears {marginal_floor}x and "
        f"straddles 1.0x -> no reliable win. Re-measure or abandon."
    )


def _better_than_phrase(x: float) -> str:
    return f"{x}x better"


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


# --- audit ANY benchmark output: file input + methodology gaps -----------------
#
# The research hole: whoever runs a benchmark, they publish ONE number, and nobody
# has a tool that tells them whether that number is within noise or whether the
# claim even declared enough methodology. `audit_benchmark_data` is the universal
# auditor: give it before/after samples for ANY metric (latency ms/token, tokens/se,
# accuracy, ...) with its direction, and it returns a verdict. `audit_benchmark_file`
# does the same for a JSON/CSV file from any benchmark tool (vLLM, Ollama, HF
# inference-benchmarker, ...), plus a methodology-gap report.

VERDICT_RESOLVED = "RESOLVED"
VERDICT_UNRESOLVED = "UNRESOLVED"
VERDICT_INSUFFICIENT = "INSUFFICIENT_RUNS"
# Short aliases so the verdict strings are ergonomic in one place.
RESOLVED, UNRESOLVED, INSUFFICIENT_RUNS = VERDICT_RESOLVED, VERDICT_UNRESOLVED, VERDICT_INSUFFICIENT


@dataclass
class BenchmarkAudit:
    verdict: str            # RESOLVED / UNRESOLVED / INSUFFICIENT_RUNS
    reason: str
    speed_verdict: str      # REAL / MARGINAL / FALSE (or "" if no valid pair)
    lo: float
    hi: float
    n: int
    methodology_gaps: list[str]
    warning: str | None


METHODOLOGY_REQUIRED = (
    ("n_runs", "how many repeats (n) to see if the number is within noise"),
    ("metric", "the metric being compared (tok/s, ms/token, latency, ...)"),
    ("input_len", "the input/prompt length the number was measured at"),
    ("output_len", "the output length / decode length the number was measured at"),
    ("cache_state", "warm vs cold cache (a 3-5x p95 gap can hide here)"),
    ("load_concurrency", "single-stream vs concurrent (batch vs single never compare)"),
)


def _parse_number(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        s = str(value).strip().strip('"\'')
    except Exception:
        return None
    if s in ("", "nan", "NaN", "None", "-"):
        return None
    mult = 1.0
    ulow = s.lower()
    for suffix, factor in (("ms", 1.0), ("gb/s", 1e9), ("mb/s", 1e6), ("kb/s", 1e3)):
        if ulow.endswith(suffix):
            mult = factor
            s = s[: -len(suffix)].strip()
            break
    try:
        return float(s) * mult
    except ValueError:
        return None


def _parse_rows_to_records(data) -> tuple[list[dict], str | None]:
    """Normalize arbitrary benchmark output into list-of-dict rows, or return
    (None, error) if we cannot understand the shape."""
    if isinstance(data, dict):
        # Common shapes:
        #   {"results": [ ... ]}  /  {"benchmarks": [...]} / list-valued keys
        for key in ("results", "benchmarks", "rows", "data", "samples", "runs"):
            val = data.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return val, None
            if isinstance(val, list) and val and isinstance(val[0], (int, float)):
                return [{"value": v} for v in val], None
        # A flat dict of scalar fields (e.g. a single summary record).
        scalars = {k: v for k, v in data.items()
                   if isinstance(v, (int, float, str)) and _parse_number(v) is not None}
        if scalars:
            return [{"record": k, "value": v} for k, v in data.items()
                    if _parse_number(v) is not None], None
    if isinstance(data, list):
        return data, None
    return None, "could not interpret benchmark data shape"


def load_benchmark_file(path: str) -> list[dict]:
    """Load a benchmark JSON or CSV file into list-of-dict rows."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, newline="") as f:
        if ext == ".csv":
            return [{k.strip(): v.strip() for k, v in row.items() if k.strip()}
                    for row in csv.DictReader(f)]
        if ext in (".json", ".jsonl"):
            text = f.read()
            if ext == ".jsonl":
                return [json.loads(line) for line in text.splitlines() if line.strip()]
            data = json.loads(text)
            rows, err = _parse_rows_to_records(data)
            if err:
                raise ValueError(f"{os.path.basename(path)}: {err}")
            return rows
        raise ValueError(f"unsupported file type: {ext} (use .json/.jsonl/.csv)")


def _extract_metric_column(rows: list[dict], metric_hint: str | None,
                           direction: str) -> tuple[list[float] | None, str | None]:
    """Find a numeric series to audit. Prefers the named `metric_hint`; otherwise
    picks the most 'timing/speed'-like numeric column. Direction must be given by
    the caller; we only auto-guess the COLUMN, never the direction (honesty rule)."""
    if not rows:
        return None, "no rows to audit"
    # 1. Explicit hint.
    if metric_hint:
        for row in rows:
            if metric_hint in row:
                vals = [_parse_number(row[metric_hint]) for row in rows]
                num = [v for v in vals if v is not None]
                if num:
                    return num, metric_hint
        return None, f"metric column '{metric_hint}' not found in data"
    # 2. Auto-guess a column by keyword, preferring latency/speed names.
    preferred = ("ms/token", "ms_per_token", "ms_per_tok", "tokens/sec", "tok/s",
                 "tokens_per_second", "latency_ms", "ttft_ms", "latency", "throughput",
                 "value", "tokens_per_sec")
    keys = list(rows[0].keys())
    for key in keys:
        if any(p in key.lower() for p in preferred) and any(
                _parse_number(r[key]) is not None for r in rows):
            return [_parse_number(r[key]) for r in rows], key
    # 3. First all-numeric column.
    for key in keys:
        if all(_parse_number(r[key]) is not None for r in rows):
            return [_parse_number(r[key]) for r in rows], key
    return None, "no numeric metric column found (use --metric-name to name it)"


def _methodology_gaps(rows: list[dict], declared: dict | None) -> list[str]:
    """Check which of the checkbox fields a trustworthy claim must declare are
    present (either in the source rows or the caller's `declared` map). Missing
    ones are reportable gaps, not automatically fatal -- but they limit how far a
    number can be trusted across workloads."""
    present: set[str] = set()
    for row in rows:
        for key in row.keys():
            k = key.strip().lower().replace(" ", "_").replace("-", "_")
            for field, _ in METHODOLOGY_REQUIRED:
                if field == k or field in k or k in field:
                    present.add(field)
    for key, value in (declared or {}).items():
        k = key.strip().lower().replace(" ", "_").replace("-", "_")
        for field, _ in METHODOLOGY_REQUIRED:
            if (field == k or field in k or k in field) and value not in (None, "", False):
                present.add(field)
    gaps = []
    for field, desc in METHODOLOGY_REQUIRED:
        if field not in present:
            gaps.append(f"{field}: {desc}")
    return gaps


def audit_data(values: list[float], direction: str = DIRECTION_LOWER_IS_BETTER,
               real_floor: float = 1.15, marginal_floor: float = 1.05,
               ) -> tuple[str, str, float | None, float | None, int]:
    """Audit a SINGLE metric series by splitting into first-half (control) and
    second-half (treatment) of the repeated runs. Returns (speed_verdict, reason,
    lo, hi, n_samples_per_side). For a true before/after pair use `audit` directly;
    this is the fallback for data that is one series of repeats."""
    n = len(values)
    if n < 2:
        return ("FALSE", f"n={n}: cannot audit a single-run number (needs repeats).",
                None, None, n)
    half = n // 2
    controls = values[:half]
    treatments = values[half:] if n % 2 == 0 else values[half:]
    lo, hi = speedup_ci(controls, treatments, direction)
    speed_v, speed_r = classify_speedup(lo, hi, real_floor, marginal_floor)
    return speed_v, speed_r, lo, hi, min(len(controls), len(treatments))


def _final_verdict(speed_verdict: str, methodology_gaps: list[str], n: int) -> tuple[str, str]:
    missing_noise = not methodology_gaps or not any(g.startswith("n_runs") for g in methodology_gaps)
    if n < 3:
        return VERDICT_INSUFFICIENT, (
            f"only n={n} runs per side: too few to separate a real win from noise "
            "(need >=3, ideally >=5). Re-measure."
        )
    if speed_verdict == "REAL":
        if missing_noise:
            return VERDICT_RESOLVED, "clean, repeated measurement outside the noise band."
        return VERDICT_RESOLVED, "speedup is outside the noise band (REAL)."
    if speed_verdict == "MARGINAL":
        return VERDICT_UNRESOLVED, "appears better but is within the single-run noise band; re-measure."
    return VERDICT_UNRESOLVED, "no reliable win in this data."


def audit_benchmark_data(controls: list[float], treatments: list[float],
                         direction: str = DIRECTION_LOWER_IS_BETTER,
                         declared: dict | None = None,
                         real_floor: float = 1.15,
                         marginal_floor: float = 1.05) -> BenchmarkAudit:
    """Audit a before/after benchmark on ANY metric, with its direction, and
    return a RESOLVED / UNRESOLVED / INSUFFICIENT_RUNS verdict plus methodology
    gaps. `declared` lets the caller state methodology fields (n_runs, metric,
    input_len, output_len, cache_state, load_concurrency) that the benchmark
    report declared."""
    lo, hi = speedup_ci(controls, treatments, direction)
    speed_v, speed_r = classify_speedup(lo, hi, real_floor, marginal_floor)
    n = min(len(controls), len(treatments))
    declared_meta = dict(declared or {})
    if "n_runs" not in declared_meta:
        declared_meta["n_runs"] = n
    gaps = _methodology_gaps([], declared_meta)
    verdict, reason = _final_verdict(speed_v, gaps, n)
    warning = None
    if speed_v in ("REAL", "MARGINAL") and any(
            g.startswith(("input_len", "output_len", "cache_state", "load_concurrency"))
            for g in gaps):
        warning = "the win may not transfer to a different workload (missing methodology); scope it to the reported length/cache/concurrency."
    return BenchmarkAudit(verdict, reason, speed_v, lo, hi, n, gaps, warning)


def audit_benchmark_file(path: str, metric_name: str | None = None,
                         direction: str = DIRECTION_LOWER_IS_BETTER,
                         declared: dict | None = None,
                         single_series: bool = False,
                         real_floor: float = 1.15,
                         marginal_floor: float = 1.05) -> BenchmarkAudit:
    """Audit any benchmark JSON/CSV file. Without a control/treatment split in the
    data, it audits the named (or auto-guessed) metric series by first-half vs
    second-half of the repeats (`single_series=True` is the default fallback).
    If the file has before/after columns (control_*/after_*), pass their names via
    `metric_name` as 'control_col,after_col' and `single_series=False`."""
    rows = load_benchmark_file(path)
    if not rows:
        return BenchmarkAudit(VERDICT_UNRESOLVED, "benchmark file is empty.",
                              "FALSE", float("nan"), float("nan"), 0, [], None)

    declared_meta = dict(declared or {})
    if metric_name and "," in metric_name:
        control_col, after_col = (c.strip() for c in metric_name.split(",", 1))
        controls = [_parse_number(r[control_col]) for r in rows]
        after = [_parse_number(r[after_col]) for r in rows]
        controls = [c for c in controls if c is not None]
        after = [a for a in after if a is not None]
        if "n_runs" not in declared_meta:
            declared_meta["n_runs"] = min(len(controls), len(after))
        gaps = _methodology_gaps(rows, declared_meta)
        lo, hi = speedup_ci(controls, after, direction)
        speed_v, speed_r = classify_speedup(lo, hi, real_floor, marginal_floor)
        n = min(len(controls), len(after))
        verdict, reason = _final_verdict(speed_v, gaps, n)
        warning = None
        if speed_v in ("REAL", "MARGINAL") and any(
                g.startswith(("input_len", "output_len", "cache_state", "load_concurrency"))
                for g in gaps):
            warning = "the win may not transfer to a different workload; scope it to the reported length/cache/concurrency."
        return BenchmarkAudit(verdict, reason, speed_v, lo, hi, n, gaps, warning)

    # Single-series mode: auto-guess the numeric column and split into halves.
    values, used_col = _extract_metric_column(rows, metric_name, direction)
    if values is None:
        return BenchmarkAudit(VERDICT_UNRESOLVED, f"{used_col} (in {os.path.basename(path)}).",
                              "FALSE", float("nan"), float("nan"), 0,
                              _methodology_gaps(rows, declared_meta), None)
    n = len(values)
    if "n_runs" not in declared_meta:
        declared_meta["n_runs"] = n
    gaps = _methodology_gaps(rows, declared_meta)
    half = n // 2
    controls = values[:half]
    treatments = values[half:] if n % 2 == 0 else values[half:]
    lo, hi = speedup_ci(controls, treatments, direction)
    speed_v, speed_r = classify_speedup(lo, hi, real_floor, marginal_floor)
    per_side = min(len(controls), len(treatments))
    verdict, reason = _final_verdict(speed_v, gaps, per_side)
    warning = None
    if speed_v in ("REAL", "MARGINAL") and any(
            g.startswith(("input_len", "output_len", "cache_state", "load_concurrency"))
            for g in gaps):
        warning = "the win may not transfer to a different workload; scope it to the reported length/cache/concurrency."
    return BenchmarkAudit(verdict, reason, speed_v, lo, hi, per_side, gaps, warning)


def summarize_audit(a: BenchmarkAudit, metric_label: str = "") -> str:
    lines = [f"AUDIT VERDICT: {a.verdict}"]
    if metric_label:
        lines[0] += f"  (metric: {metric_label})"
    lines.append(f"  speed:    {a.speed_verdict}"
                 + (f"  CI [{a.lo:.2f}x, {a.hi:.2f}x], n={a.n} per side" if a.lo == a.lo else ""))
    lines.append(f"  reason:   {a.reason}")
    if a.methodology_gaps:
        lines.append("  methodology gaps (the claim did not declare):")
        lines += [f"     - {g}" for g in a.methodology_gaps]
    else:
        lines.append("  methodology: all required fields declared.")
    if a.warning:
        lines.append(f"  warning:  {a.warning}")
    return "\n".join(lines)