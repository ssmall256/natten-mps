#!/usr/bin/env python3
"""Cross-project benchmark v3: natten-mps vs natten-mlx (standard + varlen).

Spawns separate Python processes for each project (isolated venvs).
Covers forward, backward, and the new variable-length (varlen) forward paths.

Usage:
    python benchmarks/cross_bench_v3.py
    python benchmarks/cross_bench_v3.py --backward
    python benchmarks/cross_bench_v3.py --dim 1d 2d
    python benchmarks/cross_bench_v3.py --varlen-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

NATTEN_MPS_PYTHON = Path.home() / "Code/natten-mps/.venv/bin/python"
NATTEN_MLX_PYTHON = Path.home() / "Code/natten-mlx/.venv/bin/python"

WARMUP = 5
REPEATS = 20


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------

def get_standard_configs(dims: List[str]) -> List[Dict[str, Any]]:
    configs = []
    if "1d" in dims:
        B, H, D = 1, 4, 32
        configs += [
            {"name": "1D L=256 K=7", "dim": "1d", "shape": [B, 256, H, D], "kernel_size": 7},
            {"name": "1D L=1024 K=7", "dim": "1d", "shape": [B, 1024, H, D], "kernel_size": 7},
            {"name": "1D L=256 K=7 causal", "dim": "1d", "shape": [B, 256, H, D], "kernel_size": 7, "is_causal": True},
            {"name": "1D L=256 K=7 dil2", "dim": "1d", "shape": [B, 256, H, D], "kernel_size": 7, "dilation": 2},
        ]
    if "2d" in dims:
        B, H, D = 1, 4, 32
        configs += [
            {"name": "2D 32×32 K=7", "dim": "2d", "shape": [B, 32, 32, H, D], "kernel_size": 7},
            {"name": "2D 64×64 K=7", "dim": "2d", "shape": [B, 64, 64, H, D], "kernel_size": 7},
            {"name": "2D 32×32 K=7 causal", "dim": "2d", "shape": [B, 32, 32, H, D], "kernel_size": 7, "is_causal": [True, True]},
        ]
    if "3d" in dims:
        B, H, D = 1, 4, 32
        configs += [
            {"name": "3D 16³ K=3", "dim": "3d", "shape": [B, 16, 16, 16, H, D], "kernel_size": 3},
            {"name": "3D 8³ K=3 causal", "dim": "3d", "shape": [B, 8, 8, 8, H, D], "kernel_size": 3, "is_causal": [True, True, True]},
        ]
    return configs


def get_varlen_configs(dims: List[str]) -> List[Dict[str, Any]]:
    configs = []
    if "1d" in dims:
        B, H, D = 4, 4, 32
        configs += [
            {"name": "varlen 1D B=4 L=128 K=7", "dim": "1d", "shape": [B, 128, H, D],
             "kernel_size": 7, "varlen": True,
             "seq_lens": [128, 96, 64, 48]},
            {"name": "varlen 1D B=4 L=256 K=7", "dim": "1d", "shape": [B, 256, H, D],
             "kernel_size": 7, "varlen": True,
             "seq_lens": [256, 192, 128, 64]},
            {"name": "varlen 1D B=4 L=256 K=7 dil2", "dim": "1d", "shape": [B, 256, H, D],
             "kernel_size": 7, "dilation": 2, "varlen": True,
             "seq_lens": [256, 192, 128, 64]},
        ]
    if "2d" in dims:
        B, H, D = 2, 4, 32
        configs += [
            {"name": "varlen 2D B=2 16×16 K=3", "dim": "2d", "shape": [B, 16, 16, H, D],
             "kernel_size": 3, "varlen": True,
             "spatial_sizes": [[16, 16], [12, 10]]},
            {"name": "varlen 2D B=2 32×32 K=7", "dim": "2d", "shape": [B, 32, 32, H, D],
             "kernel_size": 7, "varlen": True,
             "spatial_sizes": [[32, 32], [24, 20]]},
        ]
    if "3d" in dims:
        B, H, D = 2, 4, 32
        configs += [
            {"name": "varlen 3D B=2 8³ K=3", "dim": "3d", "shape": [B, 8, 8, 8, H, D],
             "kernel_size": 3, "varlen": True,
             "spatial_sizes": [[8, 8, 8], [6, 6, 4]]},
        ]
    return configs


# ---------------------------------------------------------------------------
# Worker scripts
# ---------------------------------------------------------------------------

WORKER_MPS = '''
import json, sys, time, torch

WARMUP = {warmup}
REPEATS = {repeats}

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
from natten_mps.functional import na1d_varlen, na2d_varlen, na3d_varlen
natten_mps.set_backend("metal")

fn_map = {{"1d": na1d, "2d": na2d, "3d": na3d}}
varlen_fn_map = {{"1d": na1d_varlen, "2d": na2d_varlen, "3d": na3d_varlen}}
results = []

for cfg in configs:
    name = cfg["name"]
    dim = cfg["dim"]
    shape = cfg["shape"]
    ks = cfg["kernel_size"]
    dil = cfg.get("dilation", 1)
    causal = cfg.get("is_causal", False)
    is_varlen = cfg.get("varlen", False)

    q = torch.randn(shape, device="mps", dtype=torch.float32)
    k = torch.randn(shape, device="mps", dtype=torch.float32)
    v = torch.randn(shape, device="mps", dtype=torch.float32)

    if is_varlen:
        na_fn = varlen_fn_map[dim]
        if dim == "1d":
            extra = torch.tensor(cfg["seq_lens"], device="mps", dtype=torch.int32)
            kwargs = {{"kernel_size": ks, "seq_lens": extra}}
        else:
            extra = torch.tensor(cfg["spatial_sizes"], device="mps", dtype=torch.int32)
            kwargs = {{"kernel_size": ks, "spatial_sizes": extra}}
        if dil != 1:
            kwargs["dilation"] = dil
    else:
        na_fn = fn_map[dim]
        kwargs = {{"kernel_size": ks}}
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
    if backward and not is_varlen:
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

    results.append({{"name": name, "fwd_ms": fwd_ms, "bwd_ms": bwd_ms}})
    bwd_str = f" bwd={{bwd_ms:.2f}}ms" if bwd_ms is not None else ""
    print(f"  {{name}}: fwd={{fwd_ms:.2f}}ms{{bwd_str}}", file=sys.stderr)

natten_mps.set_backend("auto")
print(json.dumps(results))
'''

WORKER_MLX = '''
import json, sys, time
import mlx.core as mx

WARMUP = {warmup}
REPEATS = {repeats}

def median(t):
    s = sorted(t)
    return s[len(s)//2]

configs = json.loads(sys.argv[1])
backward = sys.argv[2] == "true"

from natten_mlx import na1d, na2d, na3d, set_backend
from natten_mlx import na1d_varlen, na2d_varlen, na3d_varlen
set_backend("fast_metal")

fn_map = {{"1d": na1d, "2d": na2d, "3d": na3d}}
varlen_fn_map = {{"1d": na1d_varlen, "2d": na2d_varlen, "3d": na3d_varlen}}
results = []

for cfg in configs:
    name = cfg["name"]
    dim = cfg["dim"]
    shape = cfg["shape"]
    ks = cfg["kernel_size"]
    dil = cfg.get("dilation", 1)
    causal = cfg.get("is_causal", False)
    is_varlen = cfg.get("varlen", False)

    q = mx.random.normal(shape)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)
    mx.eval(q, k, v)

    if is_varlen:
        na_fn = varlen_fn_map[dim]
        if dim == "1d":
            extra = mx.array(cfg["seq_lens"], dtype=mx.int32)
            kwargs = {{"kernel_size": ks, "seq_lens": extra}}
        else:
            extra = mx.array(cfg["spatial_sizes"], dtype=mx.int32)
            kwargs = {{"kernel_size": ks, "spatial_sizes": extra}}
        if dil != 1:
            kwargs["dilation"] = dil
        mx.eval(extra)
    else:
        na_fn = fn_map[dim]
        kwargs = {{"kernel_size": ks}}
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
    if backward and not is_varlen:
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

    results.append({{"name": name, "fwd_ms": fwd_ms, "bwd_ms": bwd_ms}})
    bwd_str = f" bwd={{bwd_ms:.2f}}ms" if bwd_ms is not None else ""
    print(f"  {{name}}: fwd={{fwd_ms:.2f}}ms{{bwd_str}}", file=sys.stderr)

set_backend("auto")
print(json.dumps(results))
'''


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_worker(python: Path, script: str, configs: List[Dict], backward: bool,
               label: str, timeout: int = 600) -> Optional[List[Dict]]:
    if not python.exists():
        print(f"  [{label}] SKIP — {python} not found")
        return None

    configs_json = json.dumps(configs)
    bwd_str = "true" if backward else "false"
    rendered = script.format(warmup=WARMUP, repeats=REPEATS)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(rendered)
        script_path = f.name

    try:
        print(f"  [{label}] Running {len(configs)} configs...")
        proc = subprocess.run(
            [str(python), script_path, configs_json, bwd_str],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.stderr:
            for line in proc.stderr.strip().split("\n"):
                print(f"    {label}: {line.strip()}")

        if proc.returncode != 0:
            print(f"  [{label}] FAILED (exit {proc.returncode})")
            if proc.stderr:
                for line in proc.stderr.strip().split("\n")[-10:]:
                    print(f"    {line}")
            return None

        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        print(f"  [{label}] TIMEOUT after {timeout}s")
        return None
    except json.JSONDecodeError:
        print(f"  [{label}] Failed to parse JSON output")
        if proc.stdout:
            print(f"    stdout: {proc.stdout[:500]}")
        return None
    finally:
        os.unlink(script_path)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_ms(ms: Optional[float]) -> str:
    if ms is None:
        return "—"
    if ms < 0.1:
        return f"{ms*1000:.0f}µs"
    return f"{ms:.2f}ms"


def print_results(configs: List[Dict], mps_results: Optional[List[Dict]],
                  mlx_results: Optional[List[Dict]], backward: bool,
                  section_title: str):
    mps = {r["name"]: r for r in (mps_results or [])}
    mlx = {r["name"]: r for r in (mlx_results or [])}

    # Filter to only configs in this section
    relevant = [c for c in configs if c["name"] in mps or c["name"] in mlx]
    if not relevant:
        return

    print(f"\n### {section_title}")
    print()

    if backward:
        print(f"| {'Config':<30} | {'MPS fwd':>9} | {'MLX fwd':>9} | {'MPS bwd':>9} | {'MLX bwd':>9} | {'MLX fwd speedup':>15} |")
        print(f"|{'-'*32}|{'-'*11}|{'-'*11}|{'-'*11}|{'-'*11}|{'-'*17}|")
    else:
        print(f"| {'Config':<30} | {'MPS fwd':>9} | {'MLX fwd':>9} | {'MLX fwd speedup':>15} |")
        print(f"|{'-'*32}|{'-'*11}|{'-'*11}|{'-'*17}|")

    for cfg in relevant:
        name = cfg["name"]
        m = mps.get(name, {})
        x = mlx.get(name, {})
        mps_fwd = m.get("fwd_ms")
        mlx_fwd = x.get("fwd_ms")

        if mps_fwd and mlx_fwd and mlx_fwd > 0:
            ratio = mps_fwd / mlx_fwd
            speedup = f"{ratio:.1f}×"
        else:
            speedup = "—"

        line = f"| {name:<30} | {fmt_ms(mps_fwd):>9} | {fmt_ms(mlx_fwd):>9} |"

        if backward:
            mps_bwd = m.get("bwd_ms")
            mlx_bwd = x.get("bwd_ms")
            line += f" {fmt_ms(mps_bwd):>9} | {fmt_ms(mlx_bwd):>9} |"

        line += f" {speedup:>15} |"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Cross-project benchmark v3: natten-mps vs natten-mlx")
    parser.add_argument("--dim", nargs="+", choices=["1d", "2d", "3d"], default=["1d", "2d", "3d"])
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--varlen-only", action="store_true", help="Only run varlen benchmarks")
    parser.add_argument("--no-varlen", action="store_true", help="Skip varlen benchmarks")
    args = parser.parse_args()

    standard_configs = [] if args.varlen_only else get_standard_configs(args.dim)
    varlen_configs = [] if args.no_varlen else get_varlen_configs(args.dim)
    all_configs = standard_configs + varlen_configs

    print("Cross-project benchmark v3: natten-mps vs natten-mlx")
    print("=" * 55)
    print(f"Dims: {args.dim}")
    print(f"Backward: {args.backward}")
    print(f"Standard configs: {len(standard_configs)}")
    print(f"Varlen configs: {len(varlen_configs)}")
    print(f"Warmup: {WARMUP}, Repeats: {REPEATS}")
    print()

    mps_results = run_worker(NATTEN_MPS_PYTHON, WORKER_MPS, all_configs, args.backward, "MPS")
    mlx_results = run_worker(NATTEN_MLX_PYTHON, WORKER_MLX, all_configs, args.backward, "MLX")

    if standard_configs:
        print_results(standard_configs, mps_results, mlx_results, args.backward,
                      "Standard neighborhood attention (fp32, Metal-accelerated)")

    if varlen_configs:
        print_results(varlen_configs, mps_results, mlx_results, False,
                      "Variable-length neighborhood attention (fp32, Metal-accelerated)")

    # Save JSON
    output = {
        "mps": mps_results,
        "mlx": mlx_results,
        "warmup": WARMUP,
        "repeats": REPEATS,
    }
    out_path = Path(__file__).parent / "cross_bench_v3_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
