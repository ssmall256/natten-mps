#!/usr/bin/env python3
"""Benchmark Metal vs Pure backend for natten-mps.

Measures wall-clock time for forward and backward passes across 1D, 2D, and 3D
neighborhood attention with varying spatial sizes, kernel sizes, and features
(causal, strided, non-uniform kernel).

Usage:
    python benchmarks/bench.py                  # default suite
    python benchmarks/bench.py --dim 1d         # only 1D benchmarks
    python benchmarks/bench.py --dim 2d 3d      # 2D and 3D
    python benchmarks/bench.py --csv results.csv  # save to CSV
    python benchmarks/bench.py --backward       # include backward pass
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

import natten_mps
from natten_mps.functional import na1d, na2d, na3d


@dataclass
class BenchConfig:
    name: str
    dim: str  # "1d", "2d", "3d"
    shape_q: Tuple[int, ...]
    kernel_size: Any
    dilation: Any = 1
    stride: Any = 1
    is_causal: Any = False


@dataclass
class BenchResult:
    config: BenchConfig
    backend: str
    fwd_ms: float
    bwd_ms: Optional[float] = None


def _sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _time_fn(fn, warmup: int = 5, repeats: int = 20) -> float:
    """Time a function, returning median time in ms."""
    # warmup
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
        times.append((t1 - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def _make_tensors(shape, device, requires_grad=False):
    q = torch.randn(shape, device=device, dtype=torch.float32)
    k = torch.randn(shape, device=device, dtype=torch.float32)
    v = torch.randn(shape, device=device, dtype=torch.float32)
    if requires_grad:
        q = q.requires_grad_(True)
        k = k.requires_grad_(True)
        v = v.requires_grad_(True)
    return q, k, v


def _get_na_fn(dim: str):
    return {"1d": na1d, "2d": na2d, "3d": na3d}[dim]


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------


def get_configs_1d() -> List[BenchConfig]:
    B, H, D = 4, 8, 64
    configs = []
    # Vary spatial size
    for L in [64, 256, 1024]:
        configs.append(BenchConfig(
            name=f"1d_L{L}_K7",
            dim="1d", shape_q=(B, L, H, D),
            kernel_size=7,
        ))
    # Vary kernel size
    for K in [3, 7, 13]:
        configs.append(BenchConfig(
            name=f"1d_L256_K{K}",
            dim="1d", shape_q=(B, 256, H, D),
            kernel_size=K,
        ))
    # Causal
    configs.append(BenchConfig(
        name="1d_L256_K7_causal",
        dim="1d", shape_q=(B, 256, H, D),
        kernel_size=7, is_causal=True,
    ))
    # Strided
    configs.append(BenchConfig(
        name="1d_L256_K7_stride2",
        dim="1d", shape_q=(B, 256, H, D),
        kernel_size=7, stride=2,
    ))
    # Causal + strided
    configs.append(BenchConfig(
        name="1d_L256_K7_causal_stride2",
        dim="1d", shape_q=(B, 256, H, D),
        kernel_size=7, stride=2, is_causal=True,
    ))
    # Dilation
    configs.append(BenchConfig(
        name="1d_L256_K7_dil2",
        dim="1d", shape_q=(B, 256, H, D),
        kernel_size=7, dilation=2,
    ))
    return configs


def get_configs_2d() -> List[BenchConfig]:
    B, H, D = 2, 4, 64
    configs = []
    # Vary spatial size
    for S in [16, 32, 64]:
        configs.append(BenchConfig(
            name=f"2d_{S}x{S}_K7",
            dim="2d", shape_q=(B, S, S, H, D),
            kernel_size=7,
        ))
    # Vary kernel size
    for K in [3, 7, 13]:
        configs.append(BenchConfig(
            name=f"2d_32x32_K{K}",
            dim="2d", shape_q=(B, 32, 32, H, D),
            kernel_size=K,
        ))
    # Causal
    configs.append(BenchConfig(
        name="2d_32x32_K7_causal",
        dim="2d", shape_q=(B, 32, 32, H, D),
        kernel_size=7, is_causal=(True, True),
    ))
    # Strided
    configs.append(BenchConfig(
        name="2d_32x32_K7_stride2",
        dim="2d", shape_q=(B, 32, 32, H, D),
        kernel_size=7, stride=2,
    ))
    # Causal + strided
    configs.append(BenchConfig(
        name="2d_32x32_K7_causal_stride2",
        dim="2d", shape_q=(B, 32, 32, H, D),
        kernel_size=7, stride=2, is_causal=(True, True),
    ))
    # Non-uniform kernel
    configs.append(BenchConfig(
        name="2d_32x32_K3x7",
        dim="2d", shape_q=(B, 32, 32, H, D),
        kernel_size=(3, 7),
    ))
    # Non-uniform dilation
    configs.append(BenchConfig(
        name="2d_32x32_K7_dil1x2",
        dim="2d", shape_q=(B, 32, 32, H, D),
        kernel_size=7, dilation=(1, 2),
    ))
    return configs


def get_configs_3d() -> List[BenchConfig]:
    B, H, D = 1, 2, 32
    configs = []
    # Vary spatial size
    for S in [8, 16]:
        configs.append(BenchConfig(
            name=f"3d_{S}x{S}x{S}_K3",
            dim="3d", shape_q=(B, S, S, S, H, D),
            kernel_size=3,
        ))
    # Kernel sizes
    for K in [3, 5]:
        configs.append(BenchConfig(
            name=f"3d_8x8x8_K{K}",
            dim="3d", shape_q=(B, 8, 8, 8, H, D),
            kernel_size=K,
        ))
    # Causal
    configs.append(BenchConfig(
        name="3d_8x8x8_K3_causal",
        dim="3d", shape_q=(B, 8, 8, 8, H, D),
        kernel_size=3, is_causal=(True, True, True),
    ))
    # Strided
    configs.append(BenchConfig(
        name="3d_8x8x8_K3_stride2",
        dim="3d", shape_q=(B, 8, 8, 8, H, D),
        kernel_size=3, stride=2,
    ))
    # Causal + strided
    configs.append(BenchConfig(
        name="3d_8x8x8_K3_causal_stride2",
        dim="3d", shape_q=(B, 8, 8, 8, H, D),
        kernel_size=3, stride=2, is_causal=(True, True, True),
    ))
    # Non-uniform kernel
    configs.append(BenchConfig(
        name="3d_8x8x8_K3x5x3",
        dim="3d", shape_q=(B, 8, 8, 8, H, D),
        kernel_size=(3, 5, 3),
    ))
    return configs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_one(cfg: BenchConfig, backend: str, backward: bool = False) -> BenchResult:
    natten_mps.set_backend(backend)
    device = "mps" if backend == "metal" else "cpu"
    na_fn = _get_na_fn(cfg.dim)

    q, k, v = _make_tensors(cfg.shape_q, device, requires_grad=backward)

    kwargs: Dict[str, Any] = {"kernel_size": cfg.kernel_size}
    if cfg.dilation != 1:
        kwargs["dilation"] = cfg.dilation
    if cfg.stride != 1:
        kwargs["stride"] = cfg.stride
    if cfg.is_causal:
        kwargs["is_causal"] = cfg.is_causal

    # Forward timing
    def fwd():
        return na_fn(q, k, v, **kwargs)

    fwd_ms = _time_fn(fwd)

    bwd_ms = None
    if backward:
        def fwd_bwd():
            q_ = q.detach().requires_grad_(True)
            k_ = k.detach().requires_grad_(True)
            v_ = v.detach().requires_grad_(True)
            out = na_fn(q_, k_, v_, **kwargs)
            out.sum().backward()

        total_ms = _time_fn(fwd_bwd)
        bwd_ms = total_ms - fwd_ms

    return BenchResult(config=cfg, backend=backend, fwd_ms=fwd_ms, bwd_ms=bwd_ms)


def format_table(results: List[Tuple[BenchResult, BenchResult]], backward: bool) -> str:
    """Format results as a table. Each tuple is (metal_result, pure_result)."""
    buf = io.StringIO()

    if backward:
        header = f"{'Benchmark':<35} {'Metal fwd':>10} {'Pure fwd':>10} {'Speedup':>8} {'Metal bwd':>10} {'Pure bwd':>10} {'Speedup':>8}"
        sep = "-" * len(header)
        buf.write(f"\n{sep}\n{header}\n{sep}\n")
        for metal, pure in results:
            fwd_speedup = pure.fwd_ms / metal.fwd_ms if metal.fwd_ms > 0 else float("inf")
            bwd_metal = f"{metal.bwd_ms:.2f}ms" if metal.bwd_ms is not None else "n/a"
            bwd_pure = f"{pure.bwd_ms:.2f}ms" if pure.bwd_ms is not None else "n/a"
            if metal.bwd_ms and pure.bwd_ms and metal.bwd_ms > 0:
                bwd_speedup = f"{pure.bwd_ms / metal.bwd_ms:.1f}x"
            else:
                bwd_speedup = "n/a"
            buf.write(
                f"{metal.config.name:<35} {metal.fwd_ms:>9.2f}ms {pure.fwd_ms:>9.2f}ms {fwd_speedup:>7.1f}x"
                f" {bwd_metal:>10} {bwd_pure:>10} {bwd_speedup:>8}\n"
            )
    else:
        header = f"{'Benchmark':<35} {'Metal fwd':>10} {'Pure fwd':>10} {'Speedup':>8}"
        sep = "-" * len(header)
        buf.write(f"\n{sep}\n{header}\n{sep}\n")
        for metal, pure in results:
            speedup = pure.fwd_ms / metal.fwd_ms if metal.fwd_ms > 0 else float("inf")
            buf.write(
                f"{metal.config.name:<35} {metal.fwd_ms:>9.2f}ms {pure.fwd_ms:>9.2f}ms {speedup:>7.1f}x\n"
            )

    buf.write(f"{sep}\n")
    return buf.getvalue()


def write_csv(path: str, results: List[Tuple[BenchResult, BenchResult]], backward: bool):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        cols = ["benchmark", "dim", "metal_fwd_ms", "pure_fwd_ms", "fwd_speedup"]
        if backward:
            cols += ["metal_bwd_ms", "pure_bwd_ms", "bwd_speedup"]
        writer.writerow(cols)
        for metal, pure in results:
            fwd_speedup = pure.fwd_ms / metal.fwd_ms if metal.fwd_ms > 0 else 0
            row = [metal.config.name, metal.config.dim, f"{metal.fwd_ms:.3f}", f"{pure.fwd_ms:.3f}", f"{fwd_speedup:.2f}"]
            if backward:
                bwd_speedup = (pure.bwd_ms / metal.bwd_ms) if (metal.bwd_ms and pure.bwd_ms and metal.bwd_ms > 0) else 0
                row += [
                    f"{metal.bwd_ms:.3f}" if metal.bwd_ms else "",
                    f"{pure.bwd_ms:.3f}" if pure.bwd_ms else "",
                    f"{bwd_speedup:.2f}" if bwd_speedup else "",
                ]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Benchmark natten-mps Metal vs Pure")
    parser.add_argument("--dim", nargs="+", choices=["1d", "2d", "3d"], default=["1d", "2d", "3d"])
    parser.add_argument("--backward", action="store_true", help="Include backward pass timing")
    parser.add_argument("--csv", type=str, default=None, help="Write results to CSV file")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    if not natten_mps.has_metal():
        print("Metal backend not available, cannot benchmark.", file=sys.stderr)
        sys.exit(1)

    configs: List[BenchConfig] = []
    if "1d" in args.dim:
        configs += get_configs_1d()
    if "2d" in args.dim:
        configs += get_configs_2d()
    if "3d" in args.dim:
        configs += get_configs_3d()

    print(f"Running {len(configs)} benchmarks (warmup={args.warmup}, repeats={args.repeats}, backward={args.backward})")
    print(f"Device: {torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else 'MPS'}")

    results: List[Tuple[BenchResult, BenchResult]] = []
    for i, cfg in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] {cfg.name} ...", end="", flush=True)
        metal_res = run_one(cfg, "metal", backward=args.backward)
        pure_res = run_one(cfg, "pure", backward=args.backward)
        speedup = pure_res.fwd_ms / metal_res.fwd_ms if metal_res.fwd_ms > 0 else 0
        print(f" {speedup:.1f}x")
        results.append((metal_res, pure_res))

    # Restore default
    natten_mps.set_backend("auto")

    print(format_table(results, args.backward))

    if args.csv:
        write_csv(args.csv, results, args.backward)
        print(f"Results written to {args.csv}")


if __name__ == "__main__":
    main()
