#!/usr/bin/env python3
"""Three-way benchmark: natten-mps vs NATTEN PR #312 vs natten-mlx.

Runs identical workloads on all available implementations and compares
wall-clock time for forward and backward passes.

Usage:
    python benchmarks/cross_bench.py
    python benchmarks/cross_bench.py --backward
    python benchmarks/cross_bench.py --dim 2d
    python benchmarks/cross_bench.py --dim 1d 2d 3d --backward
"""
from __future__ import annotations

import argparse
import io
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

HAS_NATTEN_MPS = False
HAS_NATTEN_PR = False
HAS_NATTEN_MLX = False

try:
    import natten_mps
    from natten_mps.functional import na1d as mps_na1d, na2d as mps_na2d, na3d as mps_na3d
    HAS_NATTEN_MPS = True
except ImportError:
    pass

try:
    import natten
    from natten.backends.metal import HAS_METAL_NATTEN
    if HAS_METAL_NATTEN:
        HAS_NATTEN_PR = True
except ImportError:
    pass

try:
    import mlx.core as mx
    from natten_mlx import na1d as mlx_na1d, na2d as mlx_na2d, na3d as mlx_na3d
    from natten_mlx import set_backend as mlx_set_backend
    HAS_NATTEN_MLX = True
except ImportError:
    pass


@dataclass
class BenchConfig:
    name: str
    dim: str  # "1d", "2d", "3d"
    # Shape in spatial-first layout: [B, *spatial, H, D]
    shape_1d: Optional[Tuple[int, ...]] = None  # (B, L, H, D)
    shape_2d: Optional[Tuple[int, ...]] = None  # (B, Hsp, Wsp, H, D)
    shape_3d: Optional[Tuple[int, ...]] = None  # (B, Dp, Hsp, Wsp, H, D)
    kernel_size: Any = 7
    dilation: Any = 1
    stride: Any = 1
    is_causal: Any = False


@dataclass
class BenchResult:
    backend: str
    fwd_ms: float
    bwd_ms: Optional[float] = None


WARMUP = 5
REPEATS = 20


def _sync_mps():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _median(times: List[float]) -> float:
    s = sorted(times)
    return s[len(s) // 2]


PER_CONFIG_TIMEOUT = 30  # seconds — if a single config takes longer, it's hung


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout()


def _run_with_timeout(fn, *args, timeout=PER_CONFIG_TIMEOUT):
    """Run fn(*args) with a wall-clock timeout. Returns None on timeout."""
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        result = fn(*args)
    except _Timeout:
        result = None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return result


# ---------------------------------------------------------------------------
# natten-mps runner
# ---------------------------------------------------------------------------

def run_mps(cfg: BenchConfig, backward: bool) -> Optional[BenchResult]:
    if not HAS_NATTEN_MPS:
        return None
    natten_mps.set_backend("metal")

    fn_map = {"1d": mps_na1d, "2d": mps_na2d, "3d": mps_na3d}
    na_fn = fn_map[cfg.dim]
    shape = {"1d": cfg.shape_1d, "2d": cfg.shape_2d, "3d": cfg.shape_3d}[cfg.dim]

    q = torch.randn(shape, device="mps", dtype=torch.float32)
    k = torch.randn(shape, device="mps", dtype=torch.float32)
    v = torch.randn(shape, device="mps", dtype=torch.float32)

    kwargs: Dict[str, Any] = {"kernel_size": cfg.kernel_size}
    if cfg.dilation != 1:
        kwargs["dilation"] = cfg.dilation
    if cfg.stride != 1:
        kwargs["stride"] = cfg.stride
    if cfg.is_causal:
        kwargs["is_causal"] = cfg.is_causal

    # Forward
    for _ in range(WARMUP):
        na_fn(q, k, v, **kwargs)
        _sync_mps()
    fwd_times = []
    for _ in range(REPEATS):
        _sync_mps()
        t0 = time.perf_counter()
        na_fn(q, k, v, **kwargs)
        _sync_mps()
        fwd_times.append((time.perf_counter() - t0) * 1000)
    fwd_ms = _median(fwd_times)

    bwd_ms = None
    if backward:
        def fwd_bwd():
            q_ = q.detach().requires_grad_(True)
            k_ = k.detach().requires_grad_(True)
            v_ = v.detach().requires_grad_(True)
            out = na_fn(q_, k_, v_, **kwargs)
            out.sum().backward()

        for _ in range(WARMUP):
            fwd_bwd()
            _sync_mps()
        total_times = []
        for _ in range(REPEATS):
            _sync_mps()
            t0 = time.perf_counter()
            fwd_bwd()
            _sync_mps()
            total_times.append((time.perf_counter() - t0) * 1000)
        bwd_ms = max(0.01, _median(total_times) - fwd_ms)

    natten_mps.set_backend("auto")
    return BenchResult(backend="mps", fwd_ms=fwd_ms, bwd_ms=bwd_ms)


# ---------------------------------------------------------------------------
# NATTEN PR #312 runner
# ---------------------------------------------------------------------------

def run_pr(cfg: BenchConfig, backward: bool) -> Optional[BenchResult]:
    if not HAS_NATTEN_PR:
        return None

    fn_map = {"1d": natten.na1d, "2d": natten.na2d, "3d": natten.na3d}
    na_fn = fn_map[cfg.dim]
    shape = {"1d": cfg.shape_1d, "2d": cfg.shape_2d, "3d": cfg.shape_3d}[cfg.dim]

    q = torch.randn(shape, device="mps", dtype=torch.float32)
    k = torch.randn(shape, device="mps", dtype=torch.float32)
    v = torch.randn(shape, device="mps", dtype=torch.float32)

    kwargs: Dict[str, Any] = {"kernel_size": cfg.kernel_size}
    if cfg.dilation != 1:
        kwargs["dilation"] = cfg.dilation
    if cfg.stride != 1:
        kwargs["stride"] = cfg.stride
    if cfg.is_causal:
        kwargs["is_causal"] = cfg.is_causal

    # Forward
    for _ in range(WARMUP):
        na_fn(q, k, v, **kwargs)
        _sync_mps()
    fwd_times = []
    for _ in range(REPEATS):
        _sync_mps()
        t0 = time.perf_counter()
        na_fn(q, k, v, **kwargs)
        _sync_mps()
        fwd_times.append((time.perf_counter() - t0) * 1000)
    fwd_ms = _median(fwd_times)

    bwd_ms = None
    if backward:
        def fwd_bwd():
            q_ = q.detach().requires_grad_(True)
            k_ = k.detach().requires_grad_(True)
            v_ = v.detach().requires_grad_(True)
            out = na_fn(q_, k_, v_, **kwargs)
            out.sum().backward()

        for _ in range(WARMUP):
            fwd_bwd()
            _sync_mps()
        total_times = []
        for _ in range(REPEATS):
            _sync_mps()
            t0 = time.perf_counter()
            fwd_bwd()
            _sync_mps()
            total_times.append((time.perf_counter() - t0) * 1000)
        bwd_ms = max(0.01, _median(total_times) - fwd_ms)

    return BenchResult(backend="pr", fwd_ms=fwd_ms, bwd_ms=bwd_ms)


# ---------------------------------------------------------------------------
# natten-mlx runner
# ---------------------------------------------------------------------------

def run_mlx(cfg: BenchConfig, backward: bool) -> Optional[BenchResult]:
    if not HAS_NATTEN_MLX:
        return None

    mlx_set_backend("fast_metal")
    fn_map = {"1d": mlx_na1d, "2d": mlx_na2d, "3d": mlx_na3d}
    na_fn = fn_map[cfg.dim]
    shape = {"1d": cfg.shape_1d, "2d": cfg.shape_2d, "3d": cfg.shape_3d}[cfg.dim]

    q = mx.random.normal(shape)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)
    mx.eval(q, k, v)

    kwargs: Dict[str, Any] = {"kernel_size": cfg.kernel_size}
    if cfg.dilation != 1:
        kwargs["dilation"] = cfg.dilation
    if cfg.stride != 1:
        kwargs["stride"] = cfg.stride
    if cfg.is_causal:
        kwargs["is_causal"] = cfg.is_causal

    # Forward
    for _ in range(WARMUP):
        out = na_fn(q, k, v, **kwargs)
        mx.eval(out)
    fwd_times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = na_fn(q, k, v, **kwargs)
        mx.eval(out)
        fwd_times.append((time.perf_counter() - t0) * 1000)
    fwd_ms = _median(fwd_times)

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
        # MLX grad includes the forward pass, so total = fwd+bwd
        bwd_ms = max(0.01, _median(bwd_times) - fwd_ms)

    mlx_set_backend("auto")
    return BenchResult(backend="mlx", fwd_ms=fwd_ms, bwd_ms=bwd_ms)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def get_configs_1d() -> List[BenchConfig]:
    B, H, D = 4, 8, 64
    return [
        BenchConfig("1d_L64_K7", "1d", shape_1d=(B, 64, H, D), kernel_size=7),
        BenchConfig("1d_L128_K7", "1d", shape_1d=(B, 128, H, D), kernel_size=7),
        BenchConfig("1d_L256_K7", "1d", shape_1d=(B, 256, H, D), kernel_size=7),
        BenchConfig("1d_L512_K7", "1d", shape_1d=(B, 512, H, D), kernel_size=7),
        BenchConfig("1d_L1024_K7", "1d", shape_1d=(B, 1024, H, D), kernel_size=7),
        BenchConfig("1d_L256_K3", "1d", shape_1d=(B, 256, H, D), kernel_size=3),
        BenchConfig("1d_L256_K13", "1d", shape_1d=(B, 256, H, D), kernel_size=13),
        BenchConfig("1d_L256_K7_causal", "1d", shape_1d=(B, 256, H, D), kernel_size=7, is_causal=True),
        BenchConfig("1d_L256_K7_dil2", "1d", shape_1d=(B, 256, H, D), kernel_size=7, dilation=2),
    ]


def get_configs_2d() -> List[BenchConfig]:
    B, H, D = 2, 4, 64
    return [
        BenchConfig("2d_8x8_K3", "2d", shape_2d=(B, 8, 8, H, D), kernel_size=3),
        BenchConfig("2d_16x16_K7", "2d", shape_2d=(B, 16, 16, H, D), kernel_size=7),
        BenchConfig("2d_32x32_K3", "2d", shape_2d=(B, 32, 32, H, D), kernel_size=3),
        BenchConfig("2d_32x32_K7", "2d", shape_2d=(B, 32, 32, H, D), kernel_size=7),
        BenchConfig("2d_32x32_K13", "2d", shape_2d=(B, 32, 32, H, D), kernel_size=13),
        BenchConfig("2d_64x64_K7", "2d", shape_2d=(B, 64, 64, H, D), kernel_size=7),
        BenchConfig("2d_32x32_K7_causal", "2d", shape_2d=(B, 32, 32, H, D), kernel_size=7, is_causal=(True, True)),
        BenchConfig("2d_32x32_K7_dil2", "2d", shape_2d=(B, 32, 32, H, D), kernel_size=7, dilation=2),
    ]


def get_configs_3d() -> List[BenchConfig]:
    B, H, D = 1, 2, 32
    return [
        BenchConfig("3d_8x8x8_K3", "3d", shape_3d=(B, 8, 8, 8, H, D), kernel_size=3),
        BenchConfig("3d_8x8x8_K5", "3d", shape_3d=(B, 8, 8, 8, H, D), kernel_size=5),
        BenchConfig("3d_16x16x16_K3", "3d", shape_3d=(B, 16, 16, 16, H, D), kernel_size=3),
        BenchConfig("3d_8x8x8_K3_causal", "3d", shape_3d=(B, 8, 8, 8, H, D), kernel_size=3, is_causal=(True, True, True)),
    ]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_ms(ms: Optional[float]) -> str:
    if ms is None:
        return "n/a"
    if ms < 0.1:
        return f"{ms*1000:.0f}us"
    return f"{ms:.2f}ms"


def _ratio(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None or b <= 0:
        return ""
    r = a / b
    if r >= 1:
        return f"{r:.1f}x"
    return f"{1/r:.1f}x*"  # * = first is faster


def format_table(
    results: List[Tuple[BenchConfig, Optional[BenchResult], Optional[BenchResult], Optional[BenchResult]]],
    backward: bool,
) -> str:
    buf = io.StringIO()

    if backward:
        buf.write(f"\n{'='*120}\n")
        buf.write(f"{'Config':<25} {'MPS fwd':>9} {'PR fwd':>9} {'MLX fwd':>9} {'MPS bwd':>9} {'PR bwd':>9} {'MLX bwd':>9}  Notes\n")
        buf.write(f"{'-'*120}\n")
    else:
        buf.write(f"\n{'='*90}\n")
        buf.write(f"{'Config':<25} {'MPS fwd':>9} {'PR fwd':>9} {'MLX fwd':>9}  Notes\n")
        buf.write(f"{'-'*90}\n")

    for cfg, mps_r, pr_r, mlx_r in results:
        mps_fwd = _fmt_ms(mps_r.fwd_ms) if mps_r else "n/a"
        pr_fwd = _fmt_ms(pr_r.fwd_ms) if pr_r else ("TIMEOUT" if HAS_NATTEN_PR else "n/a")
        mlx_fwd = _fmt_ms(mlx_r.fwd_ms) if mlx_r else "n/a"

        # Find fastest forward
        fwd_vals = {}
        if mps_r: fwd_vals["MPS"] = mps_r.fwd_ms
        if pr_r: fwd_vals["PR"] = pr_r.fwd_ms
        if mlx_r: fwd_vals["MLX"] = mlx_r.fwd_ms
        if fwd_vals:
            best_fwd = min(fwd_vals, key=fwd_vals.get)
            worst_fwd = max(fwd_vals, key=fwd_vals.get)
            fwd_note = f"{best_fwd} wins fwd ({fwd_vals[worst_fwd]/fwd_vals[best_fwd]:.1f}x vs {worst_fwd})"
        else:
            fwd_note = ""

        line = f"{cfg.name:<25} {mps_fwd:>9} {pr_fwd:>9} {mlx_fwd:>9}"

        if backward:
            mps_bwd = _fmt_ms(mps_r.bwd_ms) if (mps_r and mps_r.bwd_ms is not None) else "n/a"
            pr_bwd = _fmt_ms(pr_r.bwd_ms) if (pr_r and pr_r.bwd_ms is not None) else "n/a"
            mlx_bwd = _fmt_ms(mlx_r.bwd_ms) if (mlx_r and mlx_r.bwd_ms is not None) else "n/a"

            bwd_vals = {}
            if mps_r and mps_r.bwd_ms is not None: bwd_vals["MPS"] = mps_r.bwd_ms
            if pr_r and pr_r.bwd_ms is not None: bwd_vals["PR"] = pr_r.bwd_ms
            if mlx_r and mlx_r.bwd_ms is not None: bwd_vals["MLX"] = mlx_r.bwd_ms
            if bwd_vals:
                best_bwd = min(bwd_vals, key=bwd_vals.get)
                worst_bwd = max(bwd_vals, key=bwd_vals.get)
                bwd_note = f"{best_bwd} wins bwd ({bwd_vals[worst_bwd]/bwd_vals[best_bwd]:.1f}x)"
            else:
                bwd_note = ""

            line += f" {mps_bwd:>9} {pr_bwd:>9} {mlx_bwd:>9}"
            line += f"  {fwd_note} | {bwd_note}"
        else:
            line += f"  {fwd_note}"

        buf.write(line + "\n")

    if backward:
        buf.write(f"{'='*120}\n")
    else:
        buf.write(f"{'='*90}\n")

    buf.write("\nMPS = natten-mps (PyTorch Metal, split QK+AV)\n")
    buf.write("PR  = NATTEN PR #312 (PyTorch Metal, fused flash-attn)\n")
    buf.write("MLX = natten-mlx (MLX Metal, fused + split)\n")

    # Summary counts
    buf.write("\n--- Winner Summary ---\n")
    fwd_wins = {"MPS": 0, "PR": 0, "MLX": 0}
    bwd_wins = {"MPS": 0, "PR": 0, "MLX": 0}
    for cfg, mps_r, pr_r, mlx_r in results:
        fwd_vals = {}
        if mps_r: fwd_vals["MPS"] = mps_r.fwd_ms
        if pr_r: fwd_vals["PR"] = pr_r.fwd_ms
        if mlx_r: fwd_vals["MLX"] = mlx_r.fwd_ms
        if fwd_vals:
            best = min(fwd_vals, key=fwd_vals.get)
            fwd_wins[best] += 1
        if backward:
            bwd_vals = {}
            if mps_r and mps_r.bwd_ms is not None: bwd_vals["MPS"] = mps_r.bwd_ms
            if pr_r and pr_r.bwd_ms is not None: bwd_vals["PR"] = pr_r.bwd_ms
            if mlx_r and mlx_r.bwd_ms is not None: bwd_vals["MLX"] = mlx_r.bwd_ms
            if bwd_vals:
                best = min(bwd_vals, key=bwd_vals.get)
                bwd_wins[best] += 1

    total = sum(fwd_wins.values())
    buf.write(f"Forward:  MPS={fwd_wins['MPS']}/{total}  PR={fwd_wins['PR']}/{total}  MLX={fwd_wins['MLX']}/{total}\n")
    if backward:
        total_b = sum(bwd_wins.values())
        buf.write(f"Backward: MPS={bwd_wins['MPS']}/{total_b}  PR={bwd_wins['PR']}/{total_b}  MLX={bwd_wins['MLX']}/{total_b}\n")

    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Three-way benchmark: natten-mps vs NATTEN PR #312 vs natten-mlx")
    parser.add_argument("--dim", nargs="+", choices=["1d", "2d", "3d"], default=["1d", "2d", "3d"])
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    print(f"natten-mps available:  {HAS_NATTEN_MPS}")
    print(f"NATTEN PR #312:       {HAS_NATTEN_PR}")
    print(f"natten-mlx available: {HAS_NATTEN_MLX}")
    print(f"Backward: {args.backward}")
    print(f"Warmup: {WARMUP}, Repeats: {REPEATS}")

    configs: List[BenchConfig] = []
    if "1d" in args.dim:
        configs += get_configs_1d()
    if "2d" in args.dim:
        configs += get_configs_2d()
    if "3d" in args.dim:
        configs += get_configs_3d()

    print(f"\nRunning {len(configs)} benchmarks\n")

    results = []
    for i, cfg in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] {cfg.name} ...", end="", flush=True)

        mps_r = run_mps(cfg, args.backward)
        mlx_r = run_mlx(cfg, args.backward)
        # PR gets a timeout — it hangs on larger configs
        pr_r = _run_with_timeout(run_pr, cfg, args.backward)

        parts = []
        if mps_r: parts.append(f"MPS={_fmt_ms(mps_r.fwd_ms)}")
        if pr_r:
            parts.append(f"PR={_fmt_ms(pr_r.fwd_ms)}")
        elif HAS_NATTEN_PR:
            parts.append("PR=TIMEOUT")
        if mlx_r: parts.append(f"MLX={_fmt_ms(mlx_r.fwd_ms)}")
        print(f" {' '.join(parts)}")

        results.append((cfg, mps_r, pr_r, mlx_r))

    print(format_table(results, args.backward))


if __name__ == "__main__":
    main()
