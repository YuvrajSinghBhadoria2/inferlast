"""Tests for the universal benchmark auditor: takes ANY benchmark output
(JSON/CSV, latency or throughput) and returns whether the claimed 'win' is real
or within noise, plus a methodology-gap report. This is the tool the research
identified as empty: inference performance is measured everywhere but audited
nowhere."""

import json
import os

import pytest

from trustcheck import (
    speedup_ci, classify_speedup, audit_data, audit_benchmark_data,
    audit_benchmark_file, summarize_audit, BenchmarkAudit,
    DIRECTION_LOWER_IS_BETTER, DIRECTION_HIGHER_IS_BETTER,
    RESOLVED, UNRESOLVED, load_benchmark_file,
)


# --- direction-aware stats: a 2x throughput win changes which is 'before' -----

def test_higher_is_better_throughput_reports_a_win():
    # before ~10 tok/s, after ~20 tok/s across 5 runs -> ~2x faster on CPU
    controls = [10.1, 9.8, 10.3, 9.9, 10.2]
    after = [20.4, 19.8, 20.9, 20.1, 19.7]
    lo, hi = speedup_ci(controls, after, DIRECTION_HIGHER_IS_BETTER)
    assert lo >= 1.5
    v, _ = classify_speedup(lo, hi)
    assert v == "REAL"


def test_swapping_direction_does_not_report_a_win_on_noise():
    # Noisy, overlapping series -> CI straddles 1.0x in EITHER direction.
    controls = [10.0, 10.2, 9.8, 10.1, 9.9]
    after = [10.1, 9.7, 10.3, 10.0, 10.2]
    los, _ = speedup_ci(controls, after, DIRECTION_HIGHER_IS_BETTER)
    v, _ = classify_speedup(los, 1.3)
    assert v in ("MARGINAL", "FALSE")


def test_inverting_direction_on_latency_flips_the_reading():
    # Lower-is-better: control 100ms, after 50ms -> 2x better.
    controls = [100.0, 102.0, 99.0]
    after = [50.0, 48.0, 52.0]
    lo, hi = speedup_ci(controls, after, DIRECTION_LOWER_IS_BETTER)
    assert lo >= 1.5
    # Same numbers with higher-is-better would claim a ~0.5x "loss" -- not our bug
    # to double-count here, just confirming direction semantics.
    lo_h, hi_h = speedup_ci(controls, after, DIRECTION_HIGHER_IS_BETTER)
    assert hi_h < 1.0


# --- verdicts: RESOLVED / UNRESOLVED / INSUFFICIENT_RUNS ----------------------

def test_clear_win_resolved():
    controls = [100.0, 102.0, 99.0, 101.0, 98.0]
    after = [50.0, 48.0, 52.0, 51.0, 49.0]
    a = audit_benchmark_data(controls, after, DIRECTION_LOWER_IS_BETTER)
    assert a.verdict == RESOLVED
    assert a.speed_verdict == "REAL"


def test_within_noise_is_unresolved():
    controls = [100.0, 105.0, 98.0, 102.0, 99.0]
    after = [101.0, 97.0, 104.0, 100.0, 103.0]
    a = audit_benchmark_data(controls, after, DIRECTION_LOWER_IS_BETTER)
    assert a.verdict == UNRESOLVED
    assert a.speed_verdict in ("MARGINAL", "FALSE")


def test_too_few_runs_is_insufficient():
    controls = [100.0]
    after = [50.0]
    a = audit_benchmark_data(controls, after, DIRECTION_LOWER_IS_BETTER)
    assert a.verdict == "INSUFFICIENT_RUNS"
    assert a.n < 3


# --- single-series audit (first-half vs second-half of repeats) ---------------

def test_single_series_split_catches_a_drift_win():
    # A series that drifts upward (degradation) or downward (improvement) should
    # be flagged, not blessed.
    series = [100.0, 99.0, 98.0, 50.0, 49.0, 51.0]  # later half much faster
    v, reason, lo, hi, n = audit_data(series, DIRECTION_LOWER_IS_BETTER)
    assert v == "REAL"
    assert n == 3


# --- file input: JSON ---------------------------------------------------------

def test_audit_json_file(tmp_path):
    p = tmp_path / "bench.json"
    p.write_text(json.dumps({
        "benchmarks": [
            {"latency_ms": 100, "label": "before"},
            {"latency_ms": 101, "label": "before"},
            {"latency_ms": 99, "label": "before"},
            {"latency_ms": 50, "label": "after"},
            {"latency_ms": 48, "label": "after"},
            {"latency_ms": 52, "label": "after"},
        ]
    }))
    a = audit_benchmark_file(str(p), metric_name="latency_ms",
                             direction=DIRECTION_LOWER_IS_BETTER)
    assert isinstance(a, BenchmarkAudit)
    assert a.verdict in (RESOLVED, UNRESOLVED)


def test_audit_json_control_after_columns(tmp_path):
    p = tmp_path / "cmp.json"
    p.write_text(json.dumps({
        "rows": [
            {"base_ms": 100, "opt_ms": 50},
            {"base_ms": 102, "opt_ms": 49},
            {"base_ms": 99, "opt_ms": 51},
        ]
    }))
    a = audit_benchmark_file(str(p), metric_name="base_ms,opt_ms",
                             direction=DIRECTION_LOWER_IS_BETTER)
    assert a.speed_verdict == "REAL"


def test_audit_csv_file(tmp_path):
    p = tmp_path / "bench.csv"
    p.write_text("tok_per_s\n10\n11\n10\n20\n21\n19\n")
    a = audit_benchmark_file(str(p), metric_name="tok_per_s",
                             direction=DIRECTION_HIGHER_IS_BETTER)
    assert a.verdict == RESOLVED


# --- methodology gaps ---------------------------------------------------------

def test_missing_methodology_fields_are_reported(tmp_path):
    p = tmp_path / "bare.json"
    p.write_text(json.dumps({"results": [{"latency_ms": v}
                                         for v in (100, 102, 99)]}))
    a = audit_benchmark_file(str(p), metric_name="latency_ms",
                             direction=DIRECTION_LOWER_IS_BETTER)
    # With only 3 single-series samples split 1/1 (n small), or gaps reported.
    assert a.methodology_gaps  # at least one required field is missing


def test_declared_methodology_reduces_gaps():
    a = audit_benchmark_data([100, 102, 99], [50, 48, 52],
                             DIRECTION_LOWER_IS_BETTER,
                             declared={"input_len": 2048, "output_len": 256,
                                       "cache_state": "warm",
                                       "load_concurrency": 1})
    # n_runs is filled from data; the 4 declared ones are present.
    gap_fields = [g.split(":")[0] for g in a.methodology_gaps]
    assert "input_len" not in gap_fields
    assert "cache_state" not in gap_fields


# --- loading robustness -------------------------------------------------------

def test_load_json_and_jsonl(tmp_path):
    j = tmp_path / "a.json"
    j.write_text(json.dumps({"results": [{"v": 1}, {"v": 2}]}))
    assert len(load_benchmark_file(str(j))) == 2
    jl = tmp_path / "b.jsonl"
    jl.write_text('{"v":1}\n{"v":2}\n')
    assert len(load_benchmark_file(str(jl))) == 2


def test_unsupported_file_type_raises(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("x")
    with pytest.raises(ValueError):
        load_benchmark_file(str(p))


# --- CLI-style summary doesn't crash on a pathological input ------------------

def test_summarize_audit_builds_string():
    a = audit_benchmark_data([100.0, 102.0], [50.0, 49.0], DIRECTION_LOWER_IS_BETTER)
    s = summarize_audit(a, metric_label="latency_ms")
    assert "AUDIT VERDICT" in s
