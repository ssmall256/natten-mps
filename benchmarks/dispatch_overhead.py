#!/usr/bin/env python3
"""MPS dispatch overhead profiling for natten-mps.

Breaks down where time is spent in a Metal kernel call:
  1. Layout conversion (permute + contiguous)
  2. Shader compilation (first call vs cached)
  3. Kernel dispatch (the kernel(...) call itself)
  4. Synchronization (torch.mps.synchronize)
  5. End-to-end (full metal.na1d_qk_forward)
  6. bf16 upcast overhead

Usage:
    python benchmarks/dispatch_overhead.py
    python benchmarks/dispatch_overhead.py --json benchmarks/dispatch_overhead.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch


@dataclass
class Config:
    name: str
    B: int
    L: int
    H: int
    D: int
    K: int


@dataclass
class TimingResult:
    config_name: str
    component: str
    median_us: float
    pct_of_total: Optional[float] = None


CONFIGS = [
    Config("tiny", B=1, L=16, H=1, D=32, K=3),
    Config("small", B=1, L=128, H=4, D=64, K=7),
    Config("medium", B=2, L=512, H=8, D=64, K=13),
    Config("large", B=4, L=1024, H=16, D=64, K=13),
]


def _sync():
    torch.mps.synchronize()


def _time_us(fn, warmup=10, repeats=50) -> float:
    """Time a function, returning median time in microseconds."""
    for _ in range(warmup):
        fn()
        _sync()

    times = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    times.sort()
    return times[len(times) // 2]


def profile_config(cfg: Config) -> List[TimingResult]:
    """Profile all dispatch components for a given config."""
    results = []
    device = "mps"

    # Create tensors (spatial-first layout)
    q = torch.randn(cfg.B, cfg.L, cfg.H, cfg.D, device=device, dtype=torch.float32)
    k = torch.randn(cfg.B, cfg.L, cfg.H, cfg.D, device=device, dtype=torch.float32)
    _sync()

    # --- 1. Layout conversion ---
    def layout_conv():
        q.permute(0, 2, 1, 3).contiguous()

    t_layout = _time_us(layout_conv)
    results.append(TimingResult(cfg.name, "layout_conversion", t_layout))

    # --- 2. Shader compilation (first vs cached) ---
    from natten_mps._core._metal_shaders import NATTEN_METAL_SOURCE

    # First compilation (cold)
    _sync()
    t0 = time.perf_counter()
    lib_fresh = torch.mps.compile_shader(NATTEN_METAL_SOURCE)
    _sync()
    t1 = time.perf_counter()
    t_compile_cold = (t1 - t0) * 1e6

    # Cached (subsequent calls)
    t_compile_cached = _time_us(lambda: torch.mps.compile_shader(NATTEN_METAL_SOURCE))

    results.append(TimingResult(cfg.name, "shader_compile_cold", t_compile_cold))
    results.append(TimingResult(cfg.name, "shader_compile_cached", t_compile_cached))

    # --- 3. Kernel dispatch (pre-converted tensors) ---
    from natten_mps._core.metal import _get_library, _kernel_suffix

    lib = _get_library()
    q_hf = q.permute(0, 2, 1, 3).contiguous()
    k_hf = k.permute(0, 2, 1, 3).contiguous()
    attn_hf = torch.zeros(cfg.B, cfg.H, cfg.L, cfg.K, device=device, dtype=torch.float32)
    _sync()

    kernel = getattr(lib, "natten1d_qk_forward" + _kernel_suffix(q.dtype))

    def kernel_dispatch():
        kernel(q_hf, k_hf, attn_hf, cfg.B, cfg.H, cfg.L, cfg.D, cfg.K, 1,
               threads=(cfg.L, cfg.H, cfg.B))

    t_kernel = _time_us(kernel_dispatch)
    results.append(TimingResult(cfg.name, "kernel_dispatch", t_kernel))

    # --- 4. Synchronization overhead ---
    # Time just the sync call when GPU is idle
    _sync()
    t_sync_idle = _time_us(lambda: torch.mps.synchronize(), warmup=5, repeats=50)
    results.append(TimingResult(cfg.name, "sync_idle", t_sync_idle))

    # Time sync after a kernel (measures kernel + sync)
    def kernel_then_sync():
        kernel(q_hf, k_hf, attn_hf, cfg.B, cfg.H, cfg.L, cfg.D, cfg.K, 1,
               threads=(cfg.L, cfg.H, cfg.B))
        torch.mps.synchronize()

    t_kernel_sync = _time_us(kernel_then_sync, warmup=10, repeats=50)
    results.append(TimingResult(cfg.name, "kernel+sync", t_kernel_sync))

    # --- 5. End-to-end (full metal path) ---
    from natten_mps._core import metal

    def e2e():
        metal.na1d_qk_forward(q, k, (cfg.K,), (1,))

    t_e2e = _time_us(e2e)
    results.append(TimingResult(cfg.name, "end_to_end", t_e2e))

    # --- 6. bf16 upcast overhead ---
    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k.to(torch.bfloat16)
    _sync()

    def bf16_upcast():
        q_bf16.float()
        k_bf16.float()

    t_bf16 = _time_us(bf16_upcast)
    results.append(TimingResult(cfg.name, "bf16_upcast", t_bf16))

    def e2e_bf16():
        metal.na1d_qk_forward(q_bf16, k_bf16, (cfg.K,), (1,))

    t_e2e_bf16 = _time_us(e2e_bf16)
    results.append(TimingResult(cfg.name, "end_to_end_bf16", t_e2e_bf16))

    # --- Compute percentages ---
    for r in results:
        if r.component not in ("shader_compile_cold", "shader_compile_cached", "sync_idle"):
            r.pct_of_total = (r.median_us / t_e2e * 100) if t_e2e > 0 else 0

    return results


def print_results(all_results: List[TimingResult]):
    """Print results as a formatted table."""
    # Group by config
    configs_seen = []
    by_config = {}
    for r in all_results:
        if r.config_name not in by_config:
            configs_seen.append(r.config_name)
            by_config[r.config_name] = []
        by_config[r.config_name].append(r)

    for cfg_name in configs_seen:
        results = by_config[cfg_name]
        print(f"\n{'='*70}")
        print(f"  Config: {cfg_name}")
        print(f"{'='*70}")
        print(f"  {'Component':<28} {'Time (µs)':>12} {'% of E2E':>10}")
        print(f"  {'-'*28} {'-'*12} {'-'*10}")
        for r in results:
            pct = f"{r.pct_of_total:>9.1f}%" if r.pct_of_total is not None else "     n/a"
            print(f"  {r.component:<28} {r.median_us:>12.1f} {pct}")


def main():
    parser = argparse.ArgumentParser(description="MPS dispatch overhead profiling")
    parser.add_argument("--json", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available")
        return

    print("MPS Dispatch Overhead Profiling")
    print(f"PyTorch {torch.__version__}")
    print(f"Device: {torch.mps.current_allocated_memory() / 1024**2:.0f} MB allocated")

    all_results = []
    for cfg in CONFIGS:
        print(f"\nProfiling {cfg.name}: B={cfg.B} L={cfg.L} H={cfg.H} D={cfg.D} K={cfg.K}...")
        results = profile_config(cfg)
        all_results.extend(results)

    print_results(all_results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
