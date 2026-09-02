"""gpucheck: a GPU-necessity decision rule for CPU-first inference.

inferlast's central falsifiable claim (docs/RESEARCH-SPEC.md): for overhead-bound
small models on CPU there is a measurable boundary -- in (overhead fraction, model
size, batch size, latency target) -- beyond which renting a GPU actually buys a
statistically real win, and before which the best-scheduled CPU config suffices.

`gpucheck` turns that claim into an honest, local, CPU-only estimator. It does NOT
benchmark a GPU (we couldn't measure one without renting it); it estimates the
*regime* the workload is in and labels the decision:

  CPU-suffices      -- the workload is small/overhead-bound/loosely-timed; a GPU
                       rental is very unlikely to beat the best CPU schedule.
  GPU-warranted     -- the workload is large/compute-bound/tightly-timed enough
                       that a GPU could plausibly give a real, non-noise win.
  insufficient-data -- we cannot yet tell; you must either re-measure on CPU or
                       (rarely) the question genuinely needs a GPU side-check.

Every label carries a reason listing the exact inputs and assumptions, so the
decision is auditable, not a black box. CPU-only, no downloads.
"""

from __future__ import annotations
from dataclasses import dataclass


# Rough, labeled constants. These are defensible starting points, not magic:
# - CPU memory bandwidth ~= a few tens of GB/s on laptop-class CPUs
#   (used to estimate whether weights are the bottleneck at all).
# - The decode 'regime' flips when model bytes exceed what CPU bandwidth can
#   stream per unit time, or when latency demand beats a CPU-only schedule.
@dataclass(frozen=True)
class GpuSuggestion:
    verdict: str          # "CPU-suffices" | "GPU-warranted" | "insufficient-data"
    reason: str           # human-readable, auditable reason
    regime: str           # "overhead-bound" | "weight-bound" | "compute-bound" | "unknown"
    score: float          # 0..1 likelihood that GPU spend is warranted
    inputs: dict          # the exact inputs used, for auditability


def _footprint_gb(num_params: float | None, bytes_per_param: float = 2.0) -> float | None:
    """Model memory footprint in GB, fp16-ish by default (weights dominate)."""
    if num_params is None or num_params <= 0:
        return None
    return num_params * bytes_per_param / 1e9  # params * bytes/param -> GB


def _cpu_mem_bw_gbps(cpu_label: str | None) -> float | None:
    """Very rough peak memory bandwidth for a CPU label, in GB/s.

    If unknown, we return None and force 'insufficient-data' rather than guess.
    This keeps the honesty discipline: no output without an input.
    """
    if cpu_label is None:
        return None
    label = cpu_label.lower()
    # Laptop / consumer-class CPUs (single socket DIMM DDR4/DDR5):
    if "i7-9750h" in label:
        return 41.0  # mobile 9th gen, dual-channel DDR4-2666
    if "apple m1" in label:
        return 68.0
    if "m2" in label and "m2 pro" not in label and "m2 max" not in label:
        return 100.0
    # Server-ish / unknown -> don't fake it.
    return None


def _estimate_regime(
    overhead_fraction: float | None,
    num_params: float | None,
    mem_bw_gbps: float | None,
    decode_ms_per_tok: float | None,
) -> tuple[str, dict]:
    """Estimate whether the workload is overhead-bound, weight-bound, or unknown.

    Returns (regime, supporting_facts). CPU-only, evidence-based, honest about
    when it cannot tell.
    """
    facts: dict = {}

    # 1. Overhead-boundness is our strongest signal (measured).
    if overhead_fraction is not None:
        facts["overhead_fraction"] = overhead_fraction
        if overhead_fraction >= 0.7:
            return "overhead-bound", facts

    # 2. Weight-boundness via footprint vs. memory bandwidth across the decode
    #    speed. If we must stream weights across the CPU memory bus near its
    #    peak just to keep up with one token decode, weights ARE the wall.
    gb = _footprint_gb(num_params)
    bw = mem_bw_gbps
    if gb is not None and bw is not None and decode_ms_per_tok is not None and decode_ms_per_tok > 0:
        facts["footprint_gb"] = round(gb, 3)
        facts["mem_bw_gbps"] = bw
        weights_per_sec_gb = gb / (decode_ms_per_tok / 1000.0)
        facts["weight_stream_gbps"] = round(weights_per_sec_gb, 3)
        if weights_per_sec_gb >= 0.5 * bw:
            # We are streaming near/above half the memory bus to produce a single
            # token; a bandwidth-bound GPU can potentially beat this regime.
            return "weight-bound", facts

    return "unknown", facts


def decide(
    overhead_fraction: float | None = None,
    num_params: float | None = None,
    batch_size: int | None = None,
    latency_target_ms: float | None = None,
    decode_ms_per_tok: float | None = None,
    cpu_label: str | None = None,
) -> GpuSuggestion:
    """Label GPU necessity from CPU-only local estimates.

    Returns an auditable GpuSuggestion. `insufficient-data` is returned whenever
    the estimator genuinely cannot decide from the given inputs -- refusing to
    guess is a feature, not a gap.
    """
    mem_bw = _cpu_mem_bw_gbps(cpu_label)
    regime, facts = _estimate_regime(overhead_fraction, num_params, mem_bw, decode_ms_per_tok)

    inputs = {
        "overhead_fraction": overhead_fraction,
        "num_params": num_params,
        "batch_size": batch_size,
        "latency_target_ms": latency_target_ms,
        "decode_ms_per_tok": decode_ms_per_tok,
        "cpu_label": cpu_label,
    }
    # Feed the auditable facts we worked out:
    inputs.update({k: v for k, v in facts.items() if k not in inputs})

    # --- Cases that force insufficient-data (do not fake an answer) ---
    needing_latency = latency_target_ms is None and regime in ("overhead-bound", "unknown")
    if needing_latency or cpu_label is None:
        missing = []
        if cpu_label is None:
            missing.append("cpu_label (so memory bandwidth is known)")
        if needing_latency:
            missing.append("latency_target_ms (so timing feasibility is known)")
        return GpuSuggestion(
            verdict="insufficient-data",
            reason=f"cannot decide without: {', '.join(missing)}.",
            regime=regime,
            score=0.5,
            inputs=inputs,
        )

    # --- Weight-bound / strongly compute-bound -> GPU becomes more defensible ---
    if regime == "weight-bound":
        return GpuSuggestion(
            verdict="GPU-warranted",
            reason=(
                f"workload is weight-bound on CPU (streaming ~{inputs.get('weight_stream_gbps')} GB/s "
                f"vs ~{mem_bw} GB/s peak) and latency target of {latency_target_ms} ms/token is tight; "
                "a GPU that is bandwidth-bound can plausibly give a real, non-noise win."
            ),
            regime=regime,
            score=0.85,
            inputs=inputs,
        )

    # --- Overhead-bound / unknown -> the CPU-first default ---
    # Batch size raises the bar: a server-style batch can flip the regime.
    is_server_batch = batch_size is not None and batch_size >= 16

    if regime == "overhead-bound":
        if is_server_batch and latency_target_ms <= 200:
            return GpuSuggestion(
                verdict="GPU-warranted",
                reason=(
                    f"overhead-bound on CPU but you asked for batch={batch_size} at "
                    f"{latency_target_ms} ms/token; a scheduling/batching regime a GPU can "
                    "fairly claim. Re-measure on CPU first, then sanity-check a GPU."
                ),
                regime=regime,
                score=0.7,
                inputs=inputs,
            )
        return GpuSuggestion(
            verdict="CPU-suffices",
            reason=(
                f"{overhead_fraction*100:.0f}% of wall time is framework overhead (not model "
                f"math); ~{mem_bw} GB/s CPU bandwidth and a {latency_target_ms} ms/token target. "
                "A GPU will largely re-spend that overhead. Best-scheduled CPU config suffices."
            ),
            regime=regime,
            score=0.25,
            inputs=inputs,
        )

    # unknown regime, but we do have latency + cpu -> decide on timing alone
    if latency_target_ms <= 150:
        return GpuSuggestion(
            verdict="insufficient-data",
            reason=(
                "regime unknown and the latency target is tight enough to be ambiguous; "
                "re-measure overhead_fraction on CPU before committing to GPU."
            ),
            regime=regime,
            score=0.5,
            inputs=inputs,
        )
    return GpuSuggestion(
        verdict="CPU-suffices",
        reason="regime unknown but latency demand is loose; CPU-first is the cheaper default.",
        regime=regime,
        score=0.4,
        inputs=inputs,
    )