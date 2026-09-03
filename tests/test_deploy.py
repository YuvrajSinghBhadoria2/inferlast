"""Tests for `inferlast deploy` — the one-stop decide->deploy step.

Guards the honesty contract: never overclaim a GPU win, never fabricate a
deploy target, and prefer the measured batch over a caller default.
"""

import json
import os

from deploy import (
    build_deployment, summarize, write_artifact,
)


def _qwen_out():
    return {
        "hardware": "CPU (2019 MacBook Pro 16in i7-9750H, 16GB)",
        "num_params": 494_000_000,
        "decode": {"ms_per_token": 258.0, "overhead_fraction": 0.98,
                   "tok_per_s": 3.87},
        "batching": {"rows": [{"tok_per_s": 22.0}, {"tok_per_s": 79.2}],
                     "best_batch": 16, "best_tok_per_s": 79.2},
    }


def test_cpu_suffices_emits_llamacpp_and_is_honest():
    dep = build_deployment(_qwen_out(), model="Qwen/Qwen2.5-0.5B-Instruct",
                           latency_target_ms=1000, batch=1, req_per_hr=100)
    assert dep.verdict == "CPU-suffices"
    assert dep.target == "cpu"
    # emits a ready-to-run pure-CPU llama.cpp command
    assert "llama-server" in dep.command
    assert "-ngl 0" in dep.command
    # uses the MEASURED best batch, not the caller default of 1
    assert "-b 16" in dep.command
    # honesty: cost note is a cited decision rule, flagged as NOT a measurement
    assert "not measurements by inferlast" in dep.cost_note
    # honesty: reminder to verify on target hardware
    assert "verify tok/s on your target hardware" in dep.config_note.lower() \
        or "verify tok/s on your target hardware" in dep.config_note


def test_gpu_warranted_when_latency_beats_measured_decode():
    # 100ms target vs measured 258ms decode -> CPU cannot meet the SLO
    dep = build_deployment(_qwen_out(), model="Qwen/Qwen2.5-0.5B-Instruct",
                           latency_target_ms=100)
    assert dep.verdict == "GPU-warranted"
    assert dep.target == "gpu"
    assert "vllm serve" in dep.command
    # must not overclaim a measured GPU win
    assert "can plausibly win" in dep.config_note


def test_insufficient_data_refuses_to_guess_target():
    # no hardware label + no latency -> cannot decide, and MUST NOT emit a config
    dep = build_deployment({"decode": {"ms_per_token": 258.0}}, model="X")
    assert dep.verdict == "insufficient-data"
    assert dep.target == "decide-first"
    assert dep.command == ""


def test_write_artifact_writes_json_and_run_script():
    dep = build_deployment(_qwen_out(), model="Qwen/Qwen2.5-0.5B-Instruct",
                           latency_target_ms=1000)
    out_dir = "tmp-test-deploy-artifacts"
    json_path, sh_path = write_artifact(dep, out_dir)
    assert os.path.exists(json_path) and os.path.exists(sh_path)
    with open(json_path) as f:
        data = json.load(f)
    assert data["verdict"] == "CPU-suffices"
    assert data["command"].startswith("llama-server")
    with open(sh_path) as f:
        script = f.read()
    assert "llama-server" in script
    # cleanup test-owned artifacts
    os.remove(json_path)
    os.remove(sh_path)
    os.rmdir(out_dir)
