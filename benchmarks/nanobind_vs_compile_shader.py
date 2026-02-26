#!/usr/bin/env python3
"""
Benchmark: nanobind (precompiled .metallib) vs compile_shader (runtime MSL)
for 1D neighborhood attention QK and AV forward.

Measures three dispatch methods:
  A) metal backend   — torch.mps.compile_shader() + layout conversion
  B) nanobind backend — precompiled .metallib + layout conversion
  C) metal raw kernel — torch.mps.compile_shader(), no layout conversion
     (isolates kernel dispatch overhead from layout overhead)

Output: JSON file + table to stdout.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import torch

DEVICE = "mps"
DTYPE = torch.float32
WARMUP = 10
REPEATS = 50
OUTPUT_FILE = Path(__file__).parent / "nanobind_vs_compile_shader.json"


@dataclass
class BenchResult:
    config: str
    op: str
    method: str
    median_us: float
    min_us: float
    max_us: float


def median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def bench_fn(fn, warmup=WARMUP, repeats=REPEATS) -> list[float]:
    """Time a function with warmup, returning list of times in µs."""
    for _ in range(warmup):
        fn()
        torch.mps.synchronize()

    times = []
    for _ in range(repeats):
        torch.mps.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        torch.mps.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)  # ns → µs
    return times


def run_config(B: int, L: int, H: int, D: int, K: int, dil: int = 1) -> List[BenchResult]:
    """Run QK and AV benchmarks for all methods on one config."""
    config_name = f"B{B}_L{L}_H{H}_D{D}_K{K}"
    results = []

    # --- Create input tensors ---
    torch.manual_seed(42)
    q = torch.randn(B, L, H, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, L, H, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, L, H, D, device=DEVICE, dtype=DTYPE)

    # Pre-compute attention weights for AV benchmark
    from natten_mps._core import metal
    attn_weights = metal.na1d_qk_forward(q, k, (K,), (dil,))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    torch.mps.synchronize()

    # --- Method A: metal backend (compile_shader + layout conversion) ---
    def metal_qk():
        return metal.na1d_qk_forward(q, k, (K,), (dil,))

    def metal_av():
        return metal.na1d_av_forward(attn_weights, v, (K,), (dil,))

    qk_times = bench_fn(metal_qk)
    results.append(BenchResult(
        config_name, "qk", "metal",
        median(qk_times), min(qk_times), max(qk_times)))

    av_times = bench_fn(metal_av)
    results.append(BenchResult(
        config_name, "av", "metal",
        median(av_times), min(av_times), max(av_times)))

    # --- Method B: nanobind backend (precompiled metallib + layout conversion) ---
    from natten_mps._core import nanobind
    if nanobind.ext_loaded():
        def nb_qk():
            return nanobind.na1d_qk_forward(q, k, (K,), (dil,))

        def nb_av():
            return nanobind.na1d_av_forward(attn_weights, v, (K,), (dil,))

        qk_times = bench_fn(nb_qk)
        results.append(BenchResult(
            config_name, "qk", "nanobind",
            median(qk_times), min(qk_times), max(qk_times)))

        av_times = bench_fn(nb_av)
        results.append(BenchResult(
            config_name, "av", "nanobind",
            median(av_times), min(av_times), max(av_times)))
    else:
        print("  [SKIP] nanobind backend not available")

    # --- Method C: metal raw kernel (no layout conversion) ---
    # Pre-convert to heads-first layout once
    q_hf = q.permute(0, 2, 1, 3).contiguous()
    k_hf = k.permute(0, 2, 1, 3).contiguous()
    v_hf = v.permute(0, 2, 1, 3).contiguous()
    attn_hf = attn_weights.permute(0, 2, 1, 3).contiguous()
    torch.mps.synchronize()

    lib = metal._get_library()

    def raw_qk():
        out = torch.empty(B, H, L, K, device=DEVICE, dtype=DTYPE)
        kernel = getattr(lib, "natten1d_qk_forward")
        kernel(q_hf, k_hf, out, B, H, L, D, K, dil, threads=(L, H, B))
        return out

    def raw_av():
        out = torch.empty(B, H, L, D, device=DEVICE, dtype=DTYPE)
        kernel = getattr(lib, "natten1d_av_forward")
        kernel(attn_hf, v_hf, out, B, H, L, D, K, dil, threads=(L, H, B))
        return out

    qk_times = bench_fn(raw_qk)
    results.append(BenchResult(
        config_name, "qk", "metal_raw",
        median(qk_times), min(qk_times), max(qk_times)))

    av_times = bench_fn(raw_av)
    results.append(BenchResult(
        config_name, "av", "metal_raw",
        median(av_times), min(av_times), max(av_times)))

    # --- Method D: nanobind raw kernel (no layout conversion) ---
    if nanobind.ext_loaded():
        from natten_mps._core._nanobind_ext import dispatch_1d

        def nb_raw_qk():
            out = torch.empty(B, H, L, K, device=DEVICE, dtype=DTYPE)
            torch.mps.synchronize()
            dispatch_1d(
                "natten1d_qk_forward",
                q_hf.data_ptr(), k_hf.data_ptr(), out.data_ptr(),
                B, H, L, D, K, dil,
            )
            return out

        def nb_raw_av():
            out = torch.empty(B, H, L, D, device=DEVICE, dtype=DTYPE)
            torch.mps.synchronize()
            dispatch_1d(
                "natten1d_av_forward",
                attn_hf.data_ptr(), v_hf.data_ptr(), out.data_ptr(),
                B, H, L, D, K, dil,
            )
            return out

        qk_times = bench_fn(nb_raw_qk)
        results.append(BenchResult(
            config_name, "qk", "nanobind_raw",
            median(qk_times), min(qk_times), max(qk_times)))

        av_times = bench_fn(nb_raw_av)
        results.append(BenchResult(
            config_name, "av", "nanobind_raw",
            median(av_times), min(av_times), max(av_times)))

    return results


def print_results(all_results: List[BenchResult]):
    """Print results as a formatted table."""
    # Group by config
    configs = {}
    for r in all_results:
        key = (r.config, r.op)
        if key not in configs:
            configs[key] = {}
        configs[key][r.method] = r.median_us

    print()
    print("=" * 100)
    print(f"{'Config':<25} {'Op':<4} {'metal':>12} {'nanobind':>12} "
          f"{'metal_raw':>12} {'nb_raw':>12} {'nb/metal':>10}")
    print("-" * 100)

    for (config, op), methods in sorted(configs.items()):
        metal_t = methods.get("metal", 0)
        nb_t = methods.get("nanobind", 0)
        raw_t = methods.get("metal_raw", 0)
        nb_raw_t = methods.get("nanobind_raw", 0)

        ratio = f"{nb_t / metal_t:.2f}x" if metal_t > 0 and nb_t > 0 else "N/A"

        print(f"{config:<25} {op:<4} "
              f"{metal_t:>10.0f}µs "
              f"{nb_t:>10.0f}µs "
              f"{raw_t:>10.0f}µs "
              f"{nb_raw_t:>10.0f}µs "
              f"{ratio:>10}")

    print("=" * 100)
    print()
    print("Methods:")
    print("  metal       = compile_shader + layout conversion (current default)")
    print("  nanobind    = precompiled metallib + layout conversion")
    print("  metal_raw   = compile_shader kernel only (no layout conversion)")
    print("  nb_raw      = precompiled metallib kernel only (no layout conversion)")
    print("  nb/metal    = nanobind / metal ratio (<1 = nanobind faster)")
    print()


def main():
    print("Nanobind vs compile_shader benchmark")
    print("=" * 50)

    from natten_mps._core import nanobind
    print(f"Nanobind available: {nanobind.ext_loaded()}")

    configs = [
        # (B, L, H, D, K)
        (1, 64, 4, 32, 3),
        (1, 64, 4, 32, 7),
        (1, 128, 4, 64, 7),
        (2, 256, 8, 64, 7),
        (2, 512, 8, 64, 13),
        (4, 1024, 16, 64, 13),
    ]

    all_results = []
    for B, L, H, D, K in configs:
        config_name = f"B{B}_L{L}_H{H}_D{D}_K{K}"
        print(f"\n--- {config_name} ---")
        results = run_config(B, L, H, D, K)
        all_results.extend(results)

    print_results(all_results)

    # Save JSON
    data = [asdict(r) for r in all_results]
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
