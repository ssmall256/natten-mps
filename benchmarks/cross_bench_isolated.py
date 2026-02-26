#!/usr/bin/env python3
"""Cross-project benchmark runner — each project in its own venv.

Spawns separate Python processes for natten-mps, NATTEN PR, and natten-mlx,
each using their own virtual environment. Collects JSON results and prints
a combined comparison table.

Usage:
    python benchmarks/cross_bench_isolated.py
    python benchmarks/cross_bench_isolated.py --backward
    python benchmarks/cross_bench_isolated.py --dim 1d 2d
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project venv paths
# ---------------------------------------------------------------------------

NATTEN_MPS_PYTHON = Path.home() / "Code/natten-mps/.venv/bin/python"
NATTEN_MLX_PYTHON = Path.home() / "Code/natten-mlx/.venv/bin/python"
NATTEN_PR_PYTHON = Path.home() / "Code/natten-mps/.venv-pr/bin/python"

# ---------------------------------------------------------------------------
# Shared benchmark configs (as JSON-serializable dicts)
# ---------------------------------------------------------------------------

def get_configs(dims: List[str]) -> List[Dict[str, Any]]:
    configs = []
    if "1d" in dims:
        B, H, D = 4, 8, 64
        configs += [
            {"name": "1d_L64_K7", "dim": "1d", "shape": [B, 64, H, D], "kernel_size": 7},
            {"name": "1d_L128_K7", "dim": "1d", "shape": [B, 128, H, D], "kernel_size": 7},
            {"name": "1d_L256_K7", "dim": "1d", "shape": [B, 256, H, D], "kernel_size": 7},
            {"name": "1d_L512_K7", "dim": "1d", "shape": [B, 512, H, D], "kernel_size": 7},
            {"name": "1d_L1024_K7", "dim": "1d", "shape": [B, 1024, H, D], "kernel_size": 7},
            {"name": "1d_L256_K3", "dim": "1d", "shape": [B, 256, H, D], "kernel_size": 3},
            {"name": "1d_L256_K13", "dim": "1d", "shape": [B, 256, H, D], "kernel_size": 13},
            {"name": "1d_L256_K7_causal", "dim": "1d", "shape": [B, 256, H, D], "kernel_size": 7, "is_causal": True},
            {"name": "1d_L256_K7_dil2", "dim": "1d", "shape": [B, 256, H, D], "kernel_size": 7, "dilation": 2},
        ]
    if "2d" in dims:
        B, H, D = 2, 4, 64
        configs += [
            {"name": "2d_8x8_K3", "dim": "2d", "shape": [B, 8, 8, H, D], "kernel_size": 3},
            {"name": "2d_16x16_K7", "dim": "2d", "shape": [B, 16, 16, H, D], "kernel_size": 7},
            {"name": "2d_32x32_K3", "dim": "2d", "shape": [B, 32, 32, H, D], "kernel_size": 3},
            {"name": "2d_32x32_K7", "dim": "2d", "shape": [B, 32, 32, H, D], "kernel_size": 7},
            {"name": "2d_32x32_K13", "dim": "2d", "shape": [B, 32, 32, H, D], "kernel_size": 13},
            {"name": "2d_64x64_K7", "dim": "2d", "shape": [B, 64, 64, H, D], "kernel_size": 7},
            {"name": "2d_32x32_K7_causal", "dim": "2d", "shape": [B, 32, 32, H, D], "kernel_size": 7, "is_causal": [True, True]},
            {"name": "2d_32x32_K7_dil2", "dim": "2d", "shape": [B, 32, 32, H, D], "kernel_size": 7, "dilation": 2},
        ]
    if "3d" in dims:
        B, H, D = 1, 2, 32
        configs += [
            {"name": "3d_8x8x8_K3", "dim": "3d", "shape": [B, 8, 8, 8, H, D], "kernel_size": 3},
            {"name": "3d_8x8x8_K5", "dim": "3d", "shape": [B, 8, 8, 8, H, D], "kernel_size": 5},
            {"name": "3d_16x16x16_K3", "dim": "3d", "shape": [B, 16, 16, 16, H, D], "kernel_size": 3},
            {"name": "3d_8x8x8_K3_causal", "dim": "3d", "shape": [B, 8, 8, 8, H, D], "kernel_size": 3, "is_causal": [True, True, True]},
        ]
    return configs

# ---------------------------------------------------------------------------
# Worker script template — executed in each project's venv
# ---------------------------------------------------------------------------

WORKER_MPS = '''
import json, sys, time, torch

WARMUP = 5
REPEATS = 20

def median(t):
    s = sorted(t)
    return s[len(s)//2]

def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

configs = json.loads(sys.argv[1])
backward = sys.argv[2] == "true"

import natten_mps
from natten_mps.functional import na1d, na2d, na3d
natten_mps.set_backend("metal")

fn_map = {"1d": na1d, "2d": na2d, "3d": na3d}
results = []

for cfg in configs:
    name = cfg["name"]
    dim = cfg["dim"]
    shape = cfg["shape"]
    ks = cfg["kernel_size"]
    dil = cfg.get("dilation", 1)
    causal = cfg.get("is_causal", False)

    na_fn = fn_map[dim]
    q = torch.randn(shape, device="mps", dtype=torch.float32)
    k = torch.randn(shape, device="mps", dtype=torch.float32)
    v = torch.randn(shape, device="mps", dtype=torch.float32)

    kwargs = {"kernel_size": ks}
    if dil != 1: kwargs["dilation"] = dil
    if causal: kwargs["is_causal"] = causal

    for _ in range(WARMUP):
        na_fn(q, k, v, **kwargs)
        sync()
    fwd_times = []
    for _ in range(REPEATS):
        sync()
        t0 = time.perf_counter()
        na_fn(q, k, v, **kwargs)
        sync()
        fwd_times.append((time.perf_counter() - t0) * 1000)
    fwd_ms = median(fwd_times)

    bwd_ms = None
    if backward:
        def fwd_bwd():
            q_ = q.detach().requires_grad_(True)
            k_ = k.detach().requires_grad_(True)
            v_ = v.detach().requires_grad_(True)
            out = na_fn(q_, k_, v_, **kwargs)
            out.sum().backward()
        for _ in range(WARMUP):
            fwd_bwd(); sync()
        total_times = []
        for _ in range(REPEATS):
            sync()
            t0 = time.perf_counter()
            fwd_bwd(); sync()
            total_times.append((time.perf_counter() - t0) * 1000)
        bwd_ms = max(0.01, median(total_times) - fwd_ms)

    results.append({"name": name, "fwd_ms": fwd_ms, "bwd_ms": bwd_ms})
    print(f"  {name}: {fwd_ms:.2f}ms", file=sys.stderr)

print(json.dumps(results))
'''

WORKER_PR = '''
import json, sys, time, signal, torch

WARMUP = 5
REPEATS = 20
TIMEOUT = 30

def median(t):
    s = sorted(t)
    return s[len(s)//2]

def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

class TimeoutError(Exception): pass
def timeout_handler(signum, frame): raise TimeoutError()

configs = json.loads(sys.argv[1])
backward = sys.argv[2] == "true"

import natten  # noqa: E402
fn_map = {"1d": natten.na1d, "2d": natten.na2d, "3d": natten.na3d}
results = []

for cfg in configs:
    name = cfg["name"]
    dim = cfg["dim"]
    shape = cfg["shape"]
    ks = cfg["kernel_size"]
    dil = cfg.get("dilation", 1)
    causal = cfg.get("is_causal", False)

    na_fn = fn_map[dim]
    q = torch.randn(shape, device="mps", dtype=torch.float32)
    k = torch.randn(shape, device="mps", dtype=torch.float32)
    v = torch.randn(shape, device="mps", dtype=torch.float32)

    kwargs = {"kernel_size": ks}
    if dil != 1: kwargs["dilation"] = dil
    if causal: kwargs["is_causal"] = causal

    # Timeout protection for PR which can hang
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT)
    try:
        for _ in range(WARMUP):
            na_fn(q, k, v, **kwargs)
            sync()
        fwd_times = []
        for _ in range(REPEATS):
            sync()
            t0 = time.perf_counter()
            na_fn(q, k, v, **kwargs)
            sync()
            fwd_times.append((time.perf_counter() - t0) * 1000)
        fwd_ms = median(fwd_times)

        bwd_ms = None
        if backward:
            def fwd_bwd():
                q_ = q.detach().requires_grad_(True)
                k_ = k.detach().requires_grad_(True)
                v_ = v.detach().requires_grad_(True)
                out = na_fn(q_, k_, v_, **kwargs)
                out.sum().backward()
            for _ in range(WARMUP):
                fwd_bwd(); sync()
            total_times = []
            for _ in range(REPEATS):
                sync()
                t0 = time.perf_counter()
                fwd_bwd(); sync()
                total_times.append((time.perf_counter() - t0) * 1000)
            bwd_ms = max(0.01, median(total_times) - fwd_ms)

        results.append({"name": name, "fwd_ms": fwd_ms, "bwd_ms": bwd_ms})
        print(f"  {name}: {fwd_ms:.2f}ms", file=sys.stderr)
    except TimeoutError:
        results.append({"name": name, "fwd_ms": None, "bwd_ms": None, "timeout": True})
        print(f"  {name}: TIMEOUT", file=sys.stderr)
    finally:
        signal.alarm(0)

print(json.dumps(results))
'''

WORKER_MLX = '''
import json, sys, time
import mlx.core as mx

WARMUP = 5
REPEATS = 20

def median(t):
    s = sorted(t)
    return s[len(s)//2]

configs = json.loads(sys.argv[1])
backward = sys.argv[2] == "true"

from natten_mlx import na1d, na2d, na3d, set_backend
set_backend("fast_metal")

fn_map = {"1d": na1d, "2d": na2d, "3d": na3d}
results = []

for cfg in configs:
    name = cfg["name"]
    dim = cfg["dim"]
    shape = cfg["shape"]
    ks = cfg["kernel_size"]
    dil = cfg.get("dilation", 1)
    causal = cfg.get("is_causal", False)

    na_fn = fn_map[dim]
    q = mx.random.normal(shape)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)
    mx.eval(q, k, v)

    kwargs = {"kernel_size": ks}
    if dil != 1: kwargs["dilation"] = dil
    if causal: kwargs["is_causal"] = causal

    for _ in range(WARMUP):
        out = na_fn(q, k, v, **kwargs)
        mx.eval(out)
    fwd_times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = na_fn(q, k, v, **kwargs)
        mx.eval(out)
        fwd_times.append((time.perf_counter() - t0) * 1000)
    fwd_ms = median(fwd_times)

    bwd_ms = None
    if backward:
        def loss_fn(q_in, k_in, v_in):
            return mx.sum(na_fn(q_in, k_in, v_in, **kwargs))
        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2))
        for _ in range(WARMUP):
            grads = grad_fn(q, k, v)
            mx.eval(*grads)
        bwd_times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            grads = grad_fn(q, k, v)
            mx.eval(*grads)
            bwd_times.append((time.perf_counter() - t0) * 1000)
        bwd_ms = max(0.01, median(bwd_times) - fwd_ms)

    results.append({"name": name, "fwd_ms": fwd_ms, "bwd_ms": bwd_ms})
    print(f"  {name}: {fwd_ms:.2f}ms", file=sys.stderr)

set_backend("auto")
print(json.dumps(results))
'''

# ---------------------------------------------------------------------------
# Run a worker in a specific venv
# ---------------------------------------------------------------------------

def run_worker(python: Path, script: str, configs: List[Dict], backward: bool,
               label: str, timeout: int = 600) -> Optional[List[Dict]]:
    if not python.exists():
        print(f"  [{label}] SKIP — {python} not found")
        return None

    configs_json = json.dumps(configs)
    bwd_str = "true" if backward else "false"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        print(f"  [{label}] Running {len(configs)} configs...")
        proc = subprocess.run(
            [str(python), script_path, configs_json, bwd_str],
            capture_output=True, text=True, timeout=timeout,
        )
        # Print stderr (progress) to our stderr
        if proc.stderr:
            for line in proc.stderr.strip().split("\n"):
                print(f"    {label}: {line.strip()}")

        if proc.returncode != 0:
            print(f"  [{label}] FAILED (exit {proc.returncode})")
            if proc.stderr:
                # Show last few lines of error
                for line in proc.stderr.strip().split("\n")[-5:]:
                    print(f"    {line}")
            return None

        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        print(f"  [{label}] TIMEOUT after {timeout}s")
        return None
    except json.JSONDecodeError:
        print(f"  [{label}] Failed to parse JSON output")
        if proc.stdout:
            print(f"    stdout: {proc.stdout[:200]}")
        return None
    finally:
        os.unlink(script_path)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_ms(ms: Optional[float]) -> str:
    if ms is None:
        return "n/a"
    if ms < 0.1:
        return f"{ms*1000:.0f}µs"
    return f"{ms:.2f}ms"


def print_table(configs: List[Dict], mps_results: Optional[List[Dict]],
                pr_results: Optional[List[Dict]], mlx_results: Optional[List[Dict]],
                backward: bool):
    # Index results by name
    mps = {r["name"]: r for r in (mps_results or [])}
    pr = {r["name"]: r for r in (pr_results or [])}
    mlx = {r["name"]: r for r in (mlx_results or [])}

    if backward:
        print(f"\n{'='*130}")
        print(f"{'Config':<25} {'MPS fwd':>9} {'PR fwd':>9} {'MLX fwd':>9} {'MPS bwd':>9} {'PR bwd':>9} {'MLX bwd':>9}  Notes")
        print(f"{'-'*130}")
    else:
        print(f"\n{'='*95}")
        print(f"{'Config':<25} {'MPS fwd':>9} {'PR fwd':>9} {'MLX fwd':>9}  Notes")
        print(f"{'-'*95}")

    fwd_wins = {"MPS": 0, "PR": 0, "MLX": 0}
    bwd_wins = {"MPS": 0, "PR": 0, "MLX": 0}

    for cfg in configs:
        name = cfg["name"]
        m = mps.get(name, {})
        p = pr.get(name, {})
        x = mlx.get(name, {})

        mps_fwd = m.get("fwd_ms")
        pr_fwd = p.get("fwd_ms")
        mlx_fwd = x.get("fwd_ms")
        pr_timeout = p.get("timeout", False)

        # Find winner
        fwd_vals = {}
        if mps_fwd is not None: fwd_vals["MPS"] = mps_fwd
        if pr_fwd is not None: fwd_vals["PR"] = pr_fwd
        if mlx_fwd is not None: fwd_vals["MLX"] = mlx_fwd

        if fwd_vals:
            best = min(fwd_vals, key=fwd_vals.get)
            worst = max(fwd_vals, key=fwd_vals.get)
            fwd_wins[best] += 1
            note = f"{best} wins ({fwd_vals[worst]/fwd_vals[best]:.1f}x vs {worst})"
        else:
            note = ""

        pr_fwd_str = "TIMEOUT" if pr_timeout else fmt_ms(pr_fwd)
        line = f"{name:<25} {fmt_ms(mps_fwd):>9} {pr_fwd_str:>9} {fmt_ms(mlx_fwd):>9}"

        if backward:
            mps_bwd = m.get("bwd_ms")
            pr_bwd = p.get("bwd_ms")
            mlx_bwd = x.get("bwd_ms")

            bwd_vals = {}
            if mps_bwd is not None: bwd_vals["MPS"] = mps_bwd
            if pr_bwd is not None: bwd_vals["PR"] = pr_bwd
            if mlx_bwd is not None: bwd_vals["MLX"] = mlx_bwd
            if bwd_vals:
                best_b = min(bwd_vals, key=bwd_vals.get)
                bwd_wins[best_b] += 1

            line += f" {fmt_ms(mps_bwd):>9} {fmt_ms(pr_bwd):>9} {fmt_ms(mlx_bwd):>9}"
            line += f"  {note}"
        else:
            line += f"  {note}"

        print(line)

    if backward:
        print(f"{'='*130}")
    else:
        print(f"{'='*95}")

    print("\nMPS = natten-mps (PyTorch Metal, split QK+AV)")
    print("PR  = NATTEN PR (PyTorch Metal, fused flash-attn)")
    print("MLX = natten-mlx (MLX Metal, fused + split)")

    total = sum(fwd_wins.values())
    print(f"\n--- Forward Winner Summary ---")
    print(f"MPS={fwd_wins['MPS']}/{total}  PR={fwd_wins['PR']}/{total}  MLX={fwd_wins['MLX']}/{total}")
    if backward:
        total_b = sum(bwd_wins.values())
        print(f"\n--- Backward Winner Summary ---")
        print(f"MPS={bwd_wins['MPS']}/{total_b}  PR={bwd_wins['PR']}/{total_b}  MLX={bwd_wins['MLX']}/{total_b}")


def main():
    parser = argparse.ArgumentParser(description="Cross-project benchmark (isolated venvs)")
    parser.add_argument("--dim", nargs="+", choices=["1d", "2d", "3d"], default=["1d", "2d", "3d"])
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--skip-pr", action="store_true", help="Skip NATTEN PR benchmark")
    args = parser.parse_args()

    configs = get_configs(args.dim)

    print("Cross-project benchmark (isolated venvs)")
    print("=" * 50)
    print(f"Dims: {args.dim}")
    print(f"Backward: {args.backward}")
    print(f"Configs: {len(configs)}")
    print(f"MPS python: {NATTEN_MPS_PYTHON}")
    print(f"PR python:  {NATTEN_PR_PYTHON}")
    print(f"MLX python: {NATTEN_MLX_PYTHON}")
    print()

    # Run MPS and MLX first (reliable), PR last (may hang or crash Metal service)
    mps_results = run_worker(NATTEN_MPS_PYTHON, WORKER_MPS, configs, args.backward, "MPS")
    mlx_results = run_worker(NATTEN_MLX_PYTHON, WORKER_MLX, configs, args.backward, "MLX")
    pr_results = None
    if not args.skip_pr:
        pr_results = run_worker(NATTEN_PR_PYTHON, WORKER_PR, configs, args.backward, "PR", timeout=120)
    else:
        print("  [PR] Skipped (--skip-pr flag)")

    print_table(configs, mps_results, pr_results, mlx_results, args.backward)

    # Save JSON
    output = {
        "mps": mps_results,
        "pr": pr_results,
        "mlx": mlx_results,
    }
    out_path = Path(__file__).parent / "cross_bench_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
