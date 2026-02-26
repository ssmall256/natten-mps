#!/usr/bin/env python3
"""GQA overhead profiling for natten-mps.

Quantifies the cost of repeat_interleave for GQA to decide if native
kernel-level GQA support is worth implementing.

Measures:
  1. repeat_interleave cost alone
  2. Full forward time with GQA
  3. Full forward time with MHA (same Q heads)
  4. Overhead percentage

Usage:
    python benchmarks/gqa_overhead.py
    python benchmarks/gqa_overhead.py --json benchmarks/gqa_overhead.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch

from natten_mps.functional import na1d, na2d, _repeat_kv


@dataclass
class GQAConfig:
    name: str
    dim: str  # "1d" or "2d"
    B: int
    spatial: tuple  # (L,) for 1D, (H, W) for 2D
    H_q: int
    H_kv: int
    D: int
    K: int

    @property
    def n_rep(self):
        return self.H_q // self.H_kv


@dataclass
class GQAResult:
    config_name: str
    dim: str
    H_q: int
    H_kv: int
    n_rep: int
    component: str
    median_us: float
    notes: str = ""


CONFIGS = [
    # 1D configs
    GQAConfig("1D_MHA", "1d", B=2, spatial=(256,), H_q=8, H_kv=8, D=64, K=7),
    GQAConfig("1D_GQA2", "1d", B=2, spatial=(256,), H_q=8, H_kv=4, D=64, K=7),
    GQAConfig("1D_GQA4", "1d", B=2, spatial=(256,), H_q=8, H_kv=2, D=64, K=7),
    GQAConfig("1D_GQA8", "1d", B=2, spatial=(256,), H_q=8, H_kv=1, D=64, K=7),
    # 2D configs
    GQAConfig("2D_MHA", "2d", B=2, spatial=(32, 32), H_q=8, H_kv=8, D=64, K=7),
    GQAConfig("2D_GQA2", "2d", B=2, spatial=(32, 32), H_q=8, H_kv=4, D=64, K=7),
    GQAConfig("2D_GQA4", "2d", B=2, spatial=(32, 32), H_q=8, H_kv=2, D=64, K=7),
    GQAConfig("2D_GQA8", "2d", B=2, spatial=(32, 32), H_q=8, H_kv=1, D=64, K=7),
]


def _sync():
    torch.mps.synchronize()


def _time_us(fn, warmup=10, repeats=30) -> float:
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


def _make_tensors_1d(cfg: GQAConfig, device="mps"):
    q = torch.randn(cfg.B, cfg.spatial[0], cfg.H_q, cfg.D, device=device)
    k = torch.randn(cfg.B, cfg.spatial[0], cfg.H_kv, cfg.D, device=device)
    v = torch.randn(cfg.B, cfg.spatial[0], cfg.H_kv, cfg.D, device=device)
    return q, k, v


def _make_tensors_2d(cfg: GQAConfig, device="mps"):
    Hh, Hw = cfg.spatial
    q = torch.randn(cfg.B, Hh, Hw, cfg.H_q, cfg.D, device=device)
    k = torch.randn(cfg.B, Hh, Hw, cfg.H_kv, cfg.D, device=device)
    v = torch.randn(cfg.B, Hh, Hw, cfg.H_kv, cfg.D, device=device)
    return q, k, v


def profile_config(cfg: GQAConfig) -> List[GQAResult]:
    results = []
    device = "mps"

    if cfg.dim == "1d":
        q, k, v = _make_tensors_1d(cfg, device)
    else:
        q, k, v = _make_tensors_2d(cfg, device)
    _sync()

    # 1. repeat_interleave cost
    if cfg.n_rep > 1:
        def repeat_kv():
            _repeat_kv(k, cfg.n_rep)
            _repeat_kv(v, cfg.n_rep)

        t_repeat = _time_us(repeat_kv)
        kv_bytes = k.nelement() * k.element_size()
        expanded_bytes = kv_bytes * cfg.n_rep
        results.append(GQAResult(
            cfg.name, cfg.dim, cfg.H_q, cfg.H_kv, cfg.n_rep,
            "repeat_interleave", t_repeat,
            f"KV: {kv_bytes/1024:.0f}KB → {expanded_bytes/1024:.0f}KB"
        ))
    else:
        results.append(GQAResult(
            cfg.name, cfg.dim, cfg.H_q, cfg.H_kv, cfg.n_rep,
            "repeat_interleave", 0.0, "n_rep=1 (no-op)"
        ))

    # 2. Full forward time
    if cfg.dim == "1d":
        def forward():
            na1d(q, k, v, (cfg.K,), (1,), (1,), (False,))
    else:
        def forward():
            na2d(q, k, v, (cfg.K, cfg.K), (1, 1), (1, 1), (False, False))

    t_fwd = _time_us(forward)
    results.append(GQAResult(
        cfg.name, cfg.dim, cfg.H_q, cfg.H_kv, cfg.n_rep,
        "forward", t_fwd
    ))

    return results


def print_results(all_results: List[GQAResult]):
    """Print results grouped by dimension."""
    for dim in ("1d", "2d"):
        dim_results = [r for r in all_results if r.dim == dim]
        if not dim_results:
            continue

        print(f"\n{'='*80}")
        print(f"  {dim.upper()} GQA Overhead Analysis")
        print(f"{'='*80}")
        print(f"  {'Config':<12} {'H_q':>4} {'H_kv':>5} {'rep':>4} {'Component':<20} {'Time (µs)':>12} {'Notes'}")
        print(f"  {'-'*12} {'-'*4} {'-'*5} {'-'*4} {'-'*20} {'-'*12} {'-'*20}")

        for r in dim_results:
            print(f"  {r.config_name:<12} {r.H_q:>4} {r.H_kv:>5} {r.n_rep:>4} {r.component:<20} {r.median_us:>12.1f} {r.notes}")

    # Summary: GQA overhead percentage vs MHA
    print(f"\n{'='*80}")
    print(f"  GQA Overhead Summary (vs MHA baseline)")
    print(f"{'='*80}")
    print(f"  {'Config':<12} {'Ratio':>6} {'Forward (µs)':>14} {'Overhead':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*14} {'-'*10}")

    for dim in ("1d", "2d"):
        mha_result = next((r for r in all_results if r.dim == dim and r.n_rep == 1 and r.component == "forward"), None)
        if not mha_result:
            continue
        mha_time = mha_result.median_us

        for r in all_results:
            if r.dim == dim and r.component == "forward":
                overhead = (r.median_us - mha_time) / mha_time * 100 if mha_time > 0 else 0
                label = "baseline" if r.n_rep == 1 else f"+{overhead:.1f}%"
                print(f"  {r.config_name:<12} {r.n_rep:>4}:1 {r.median_us:>14.1f} {label:>10}")


def main():
    parser = argparse.ArgumentParser(description="GQA overhead profiling")
    parser.add_argument("--json", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available")
        return

    print("GQA Overhead Profiling")
    print(f"PyTorch {torch.__version__}")

    all_results = []
    for cfg in CONFIGS:
        label = f"{cfg.dim.upper()} H_q={cfg.H_q} H_kv={cfg.H_kv} (rep={cfg.n_rep})"
        print(f"\nProfiling {label}...")
        results = profile_config(cfg)
        all_results.extend(results)

    print_results(all_results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
