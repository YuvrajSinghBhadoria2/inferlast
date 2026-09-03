"""deploy: turn inferlast's measured verdict into a ready-to-run serving config.

The one-stop answer to the inference-cost question: after `inferlast run` measures
on YOUR CPU, `inferlast deploy` consumes that verdict and emits a concrete,
executable deployment -- CPU (llama.cpp) or GPU (vLLM) -- plus an honest cost
sanity note and a reminder to re-prove the result on the target hardware.

Honesty contract (mirrors the whole project):
  * deploy NEVER claims to have measured a GPU. It labels the *regime* the
    workload is in (via gpucheck) and emits the researched default server for it.
  * Every emitted config is labeled "generated from measured verdict; verify
    tok/s on your target hardware."
  * Cost guidance is a cited decision rule, never a fabricated measurement.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict

from gpucheck import decide

# Researched decision-rule bounds (reported ranges in industry cost analyses,
# NOT measured by inferlast). Kept as named constants so they are auditable.
CPU_BREAKEVEN_REQ_PER_HOUR = 5.0 * 3600  # ~5 sustained req/s (inferensys 2026)
CPU_OK_MODEL_PARAMS = 7e9              # ~7B CPU-sufficiency line (multiple 2026 sources)


def _cpu_label_from_report(out: dict) -> str | None:
    # Prefer what `run` actually recorded; fall back to the literal prompt text.
    hw = out.get("hardware") or ""
    hl = hw.lower()
    if "i7-9750h" in hl:
        return "i7-9750h"
    if "m1" in hl:
        return "apple m1"
    return None


def _model_params(out: dict) -> float | None:
    for key in ("model_size", "num_params"):
        if out.get(key):
            return out[key]
    m = out.get("model")
    if isinstance(m, dict) and m.get("num_params"):
        return m["num_params"]
    return None


def _decode_ms_per_tok(out: dict) -> float | None:
    d = out.get("decode")
    if isinstance(d, dict) and d.get("ms_per_token"):
        return d["ms_per_token"]
    return None


def _overhead_fraction(out: dict) -> float | None:
    d = out.get("decode") or out.get("prefill")
    if isinstance(d, dict) and d.get("overhead_fraction") is not None:
        return d["overhead_fraction"]
    return None


@dataclass
class Deployment:
    model: str
    verdict: str                 # CPU-suffices | GPU-warranted | insufficient-data
    reason: str                  # auditable reason from gpucheck
    regime: str
    target: str                  # "cpu" | "gpu" | "decide-first"
    engine: str
    command: str                 # ready-to-run shell command
    config_note: str             # what the flags do + verify-on-target caveat
    cost_note: str               # cited decision rule, not a fabricated number
    inputs: dict                 # exact inputs used (auditability)

    def to_dict(self) -> dict:
        return asdict(self)


def _cpu_command(model: str, threads: int, ctx: int, batch: int) -> str:
    # Researched "best CPU stack": llama.cpp/llama-server, pure CPU (-ngl 0),
    # GGUF Q4_K_M (sweet-spot quant), threads from core count, mapped KV space.
    return (
        f"llama-server -m {model}-Q4_K_M.gguf -ngl 0 "
        f"-t {threads} -c {ctx} -b {max(batch, 1)} "
        f"--host 0.0.0.0 --port 8080"
    )


def _gpu_command(model: str, batch: int) -> str:
    # Industry-default high-throughput GPU server.
    return (
        f"vllm serve {model} --dtype auto "
        f"--gpu-memory-utilization 0.9 --max-num-seqs {max(batch, 16)} "
        f"--port 8000"
    )


def build_deployment(
    out: dict,
    model: str,
    latency_target_ms: float = 1000.0,
    batch: int = 1,
    req_per_hr: float | None = None,
    cpu_label: str | None = None,
    ctx: int = 4096,
) -> Deployment:
    """Turn a measured `run` verdict (`out`) into a Deployment."""
    num_params = _model_params(out)
    d_ms = _decode_ms_per_tok(out)
    overhead = _overhead_fraction(out)

    # Prefer the measured best batch from the run verdict; only fall back to the
    # caller's --batch when the report has no batching measurement.
    b = out.get("batching")
    eff_batch = batch
    if isinstance(b, dict) and b.get("best_batch"):
        eff_batch = b["best_batch"]

    cpu_label = cpu_label or _cpu_label_from_report(out)

    sug = decide(
        overhead_fraction=overhead,
        num_params=num_params,
        batch_size=eff_batch,
        latency_target_ms=latency_target_ms,
        decode_ms_per_tok=d_ms,
        cpu_label=cpu_label,
    )

    inputs = dict(sug.inputs)
    inputs["effective_batch"] = eff_batch
    if req_per_hr is not None:
        inputs["req_per_hr"] = req_per_hr

    # Cost sanity: a cited decision rule, never a fabricated per-token price.
    cost_lines = []
    if req_per_hr is not None:
        # break-even ~5 sustained req/s (inferensys 2026)
        if req_per_hr <= CPU_BREAKEVEN_REQ_PER_HOUR:
            cost_lines.append(
                f"~{req_per_hr:.0f} req/hr is below the reported CPU-vs-GPU "
                "break-even (~5 sustained req/s); CPU is the low-fixed-cost call."
            )
        else:
            cost_lines.append(
                f"~{req_per_hr:.0f} req/hr is above the reported CPU-vs-GPU "
                "break-even (~5 sustained req/s); throughput demand may justify GPU."
            )
    if num_params is not None and num_params <= CPU_OK_MODEL_PARAMS:
        cost_lines.append(
            f"model <= ~7B (reported CPU-quotable line): small models run "
            "acceptably on CPU for many workloads."
        )
    elif num_params is not None:
        cost_lines.append(
            f"model > ~7B: CPU decode for large models is slow; GPU is usually "
            "the defensible route."
        )
    cost_lines.append(
        "These are decision rules from cited industry studies, not measurements "
        "by inferlast; re-measure tok/s on YOUR target hardware before committing."
    )
    cost_note = " ".join(cost_lines)

    # Choose the deploy target from the verdict (never overclaim a GPU win).
    target = "decide-first"
    engine = "none"
    command = ""
    if sug.verdict == "CPU-suffices":
        target = "cpu"
        engine = "llama.cpp (llama-server)"
        threads = os.cpu_count() or 8
        command = _cpu_command(model, threads, ctx, eff_batch)
        config_note = (
            f"llama.cpp with GGUF Q4_K_M (researched quality/size sweet spot), "
            f"pure CPU decode (-ngl 0), threads from core count, KV-cache + batch "
            f"B={eff_batch} from the measured batch sweep. Generated from your "
            "measured verdict; verify tok/s on your target hardware."
        )
    elif sug.verdict == "GPU-warranted":
        target = "gpu"
        engine = "vLLM"
        command = _gpu_command(model, eff_batch)
        config_note = (
            "vLLM (high-throughput GPU server default) sized from the measured "
            "verdict. GPU-warranted means a GPU can plausibly win -- verify tok/s "
            "on the actual GPU before scaling spend."
        )
    else:  # insufficient-data -> refuse to guess a deploy target
        target = "decide-first"
        engine = "none"
        command = ""
        config_note = (
            "inferlast could not decide from the available inputs; provide "
            "latency-target-ms and a cpu-label (or a measured `run` report) so "
            "the decision becomes honest rather than a guess."
        )

    return Deployment(
        model=model,
        verdict=sug.verdict,
        reason=sug.reason,
        regime=sug.regime,
        target=target,
        engine=engine,
        command=command,
        config_note=config_note,
        cost_note=cost_note,
        inputs=inputs,
    )


def summarize(dep: Deployment) -> str:
    lines = [
        "# inferlast deploy — decision + ready-to-run config",
        f"model: {dep.model}",
        f"verdict: {dep.verdict}",
        f"reason: {dep.reason}",
        f"deploy target: {dep.target}",
    ]
    if dep.command:
        lines += ["", f"## Run this ({dep.engine})", dep.command]
    lines += [
        "",
        f"config note: {dep.config_note}",
        "",
        f"cost sanity (decision rule, not a measurement): {dep.cost_note}",
    ]
    return "\n".join(lines)


def write_artifact(dep: Deployment, out_dir: str) -> tuple[str, str]:
    """Write deploy.json (structured) + run.sh (executable) into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "deploy.json")
    with open(json_path, "w") as f:
        json.dump(dep.to_dict(), f, indent=2)
    sh_path = os.path.join(out_dir, "run.sh")
    with open(sh_path, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        if dep.command:
            f.write(f"# generated by inferlast deploy for {dep.model}\n")
            f.write(dep.command + "\n")
        else:
            f.write("# no deploy target: inferlast could not decide (insufficient-data)\n")
    os.chmod(sh_path, 0o755)
    return json_path, sh_path
