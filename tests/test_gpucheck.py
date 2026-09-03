"""Tests for gpucheck: the GPU-necessity decision rule.

The core of inferlast's CPU-first thesis (docs/RESEARCH-SPEC.md): for overhead-bound
small models on CPU a GPU rental is usually NOT warranted. Verify the rule labels the
regimes correctly AND refuses to guess when data is missing.
"""

import pytest

from gpucheck import decide, _footprint_gb, _estimate_regime, GpuSuggestion


# --- overhead-bound (the measured reality of tiny models on CPU) --------------

def test_overhead_bound_cpu_loose_latency_is_cpu_suffices():
    # 98% overhead, small model, loose latency -> CPU-first default
    v = decide(overhead_fraction=0.98, num_params=0.5e9, latency_target_ms=2000,
               decode_ms_per_tok=1220, cpu_label="i7-9750H")
    assert isinstance(v, GpuSuggestion)
    assert v.verdict == "CPU-suffices"
    assert v.regime == "overhead-bound"
    assert "overhead" in v.reason.lower()


def test_overhead_bound_with_tight_batch_and_latency_can_be_warranted():
    # Server-style batching + tight latency crosses the boundary
    v = decide(overhead_fraction=0.9, num_params=7e9, batch_size=32,
               latency_target_ms=100, decode_ms_per_tok=2000, cpu_label="i7-9750H")
    assert v.verdict == "GPU-warranted"
    assert v.regime == "overhead-bound"


# --- weight-bound (large model, CPU can't stream weights) ---------------------

def test_weight_bound_large_model_is_gpu_warranted():
    # Huge footprint vs. CPU bandwidth -> GPU defensible
    v = decide(overhead_fraction=None, num_params=70e9, batch_size=1,
               latency_target_ms=500, decode_ms_per_tok=5000, cpu_label="i7-9750H")
    assert v.verdict == "GPU-warranted"
    assert v.regime == "weight-bound"


# --- refusal to guess (honesty: insufficient-data) ----------------------------

def test_missing_cpu_label_is_insufficient_data():
    v = decide(overhead_fraction=0.9, num_params=0.5e9, latency_target_ms=2000,
               cpu_label=None)
    assert v.verdict == "insufficient-data"
    assert "cpu_label" in v.reason


def test_missing_latency_on_overhead_bound_is_insufficient_data():
    v = decide(overhead_fraction=0.9, num_params=0.5e9, cpu_label="i7-9750H")
    assert v.verdict == "insufficient-data"
    assert "latency_target_ms" in v.reason


# --- latency-feasibility regression (discovered on a real Q4 7B) --------------
# A measured decode far over the latency target must be GPU-warranted, even when
# the bandwidth heuristic calls the regime "unknown" (a too-slow big model
# streams well under the bus limit yet cannot meet the requirement).

def test_measured_latency_over_target_is_gpu_warranted():
    # The exact real case: Qwen2.5-7B Q4 measured at 2884.7 ms/tok on i7-9750H.
    v = decide(num_params=7.61e9, decode_ms_per_tok=2884.7,
               latency_target_ms=500, cpu_label="i7-9750H")
    assert v.verdict == "GPU-warranted"
    assert "measured_vs_target_x" in v.inputs

def test_latency_check_ignores_missing_target():
    # No target -> the latency-feasibility check cannot fire; cpu_label path wins.
    v = decide(num_params=7.61e9, decode_ms_per_tok=2884.7, cpu_label="i7-9750H")
    assert v.verdict == "insufficient-data"


# --- helpers unit -------------------------------------------------------------

def test_footprint_gb_converts_params_to_gb():
    assert _footprint_gb(1e9) == pytest.approx(2.0)      # 1B params * 2 bytes
    assert _footprint_gb(0) is None
    assert _footprint_gb(-5) is None


def test_estimate_regime_caps_overhead_bound():
    regime, facts = _estimate_regime(0.98, 0.5e9, 41.0, 1220)
    assert regime == "overhead-bound"
    assert facts["overhead_fraction"] == 0.98


def test_estimate_regime_weight_bound_high_stream_rate():
    # Model bytes so large relative to CPU bandwidth that weights dominate decode.
    regime, facts = _estimate_regime(0.3, 70e9, 41.0, 5000)
    # overhead is low and weight stream far exceeds bandwidth fraction -> weight-bound
    assert regime == "weight-bound"


def test_estimate_regime_unknown_when_inputs_missing():
    regime, _ = _estimate_regime(None, None, None, None)
    assert regime == "unknown"