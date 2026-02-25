#!/usr/bin/env python3
"""Cross-project benchmark for new features: GQA, return_lse, additional_kv, bf16, FMHA.

Measures natten-mps (PyTorch/MPS) and natten-mlx (MLX) side by side on
identical workloads covering all six features ported from NATTEN PR #312.

Usage:
    python benchmarks/bench_new_features.py
    python benchmarks/bench_new_features.py --features gqa fmha
    python benchmarks/bench_new_features.py --json results.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

HAS_MPS = False
HAS_MLX = False

try:
    import natten_mps
    from natten_mps.functional import na1d as mps_na1d, na2d as mps_na2d
    from natten_mps.merge import merge_attentions as mps_merge
    HAS_MPS = True
except ImportError:
    pass

try:
    import mlx.core as mx
    from natten_mlx import na1d as mlx_na1d, na2d as mlx_na2d
    from natten_mlx import set_backend as mlx_set_backend
    from natten_mlx.merge import merge_attentions as mlx_merge
    HAS_MLX = True
except ImportError:
    pass

WARMUP = 5
REPEATS = 30


def _median(times: List[float]) -> float:
    s = sorted(times)
    return s[len(s) // 2]


def _sync_mps():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------

@dataclass
class FeatureBench:
    name: str
    feature: str  # gqa, return_lse, additional_kv, bf16, fmha, merge
    dim: str      # 1d, 2d
    # Runner receives this config and builds tensors
    spatial: Tuple[int, ...]
    heads_q: int
    heads_kv: int
    head_dim: int
    kernel_size: Any
    batch: int = 1
    n_extra: int = 0         # additional KV tokens
    return_lse: bool = False
    dtype_name: str = "fp32"  # fp32, bf16
    stride: Any = 1
    is_causal: Any = False


def get_baseline_configs() -> List[FeatureBench]:
    """Baseline (no new features) for comparison."""
    return [
        FeatureBench("baseline_1d_L256_K7", "baseline", "1d", (256,), 8, 8, 32, 7, batch=2),
        FeatureBench("baseline_2d_32x32_K7", "baseline", "2d", (32, 32), 4, 4, 32, 7, batch=2),
    ]


def get_gqa_configs() -> List[FeatureBench]:
    return [
        FeatureBench("gqa4_1d_L256_K7", "gqa", "1d", (256,), 8, 2, 32, 7, batch=2),
        FeatureBench("mqa_1d_L256_K7", "gqa", "1d", (256,), 8, 1, 32, 7, batch=2),
        FeatureBench("gqa4_2d_32x32_K7", "gqa", "2d", (32, 32), 8, 2, 32, 7, batch=2),
        FeatureBench("mqa_2d_32x32_K7", "gqa", "2d", (32, 32), 8, 1, 32, 7, batch=2),
    ]


def get_return_lse_configs() -> List[FeatureBench]:
    return [
        FeatureBench("lse_1d_L256_K7", "return_lse", "1d", (256,), 8, 8, 32, 7, batch=2, return_lse=True),
        FeatureBench("lse_2d_32x32_K7", "return_lse", "2d", (32, 32), 4, 4, 32, 7, batch=2, return_lse=True),
    ]


def get_additional_kv_configs() -> List[FeatureBench]:
    return [
        FeatureBench("addkv1_1d_L256_K7", "additional_kv", "1d", (256,), 8, 8, 32, 7, batch=2, n_extra=1),
        FeatureBench("addkv4_1d_L256_K7", "additional_kv", "1d", (256,), 8, 8, 32, 7, batch=2, n_extra=4),
        FeatureBench("addkv1_2d_32x32_K7", "additional_kv", "2d", (32, 32), 4, 4, 32, 7, batch=2, n_extra=1),
        # GQA + additional_kv
        FeatureBench("addkv1_gqa4_1d_L256_K7", "additional_kv", "1d", (256,), 8, 2, 32, 7, batch=2, n_extra=1),
    ]


def get_bf16_configs() -> List[FeatureBench]:
    return [
        FeatureBench("bf16_1d_L256_K7", "bf16", "1d", (256,), 8, 8, 32, 7, batch=2, dtype_name="bf16"),
        FeatureBench("bf16_2d_32x32_K7", "bf16", "2d", (32, 32), 4, 4, 32, 7, batch=2, dtype_name="bf16"),
        FeatureBench("bf16_gqa4_1d_L256_K7", "bf16", "1d", (256,), 8, 2, 32, 7, batch=2, dtype_name="bf16"),
    ]


def get_fmha_configs() -> List[FeatureBench]:
    """FMHA fast path: kernel_size >= spatial extent."""
    return [
        FeatureBench("fmha_1d_L16_Kfull", "fmha", "1d", (16,), 8, 8, 32, 16, batch=4),
        FeatureBench("fmha_1d_L64_Kfull", "fmha", "1d", (64,), 8, 8, 32, 64, batch=4),
        FeatureBench("fmha_2d_8x8_Kfull", "fmha", "2d", (8, 8), 4, 4, 32, (8, 8), batch=4),
        FeatureBench("fmha_2d_16x16_Kfull", "fmha", "2d", (16, 16), 4, 4, 32, (16, 16), batch=2),
    ]


def get_merge_configs() -> List[FeatureBench]:
    """merge_attentions benchmark (2-way merge)."""
    return [
        FeatureBench("merge2_1d_L256", "merge", "1d", (256,), 8, 8, 32, 7, batch=2),
        FeatureBench("merge2_2d_32x32", "merge", "2d", (32, 32), 4, 4, 32, 7, batch=2),
    ]


ALL_FEATURES = {
    "baseline": get_baseline_configs,
    "gqa": get_gqa_configs,
    "return_lse": get_return_lse_configs,
    "additional_kv": get_additional_kv_configs,
    "bf16": get_bf16_configs,
    "fmha": get_fmha_configs,
    "merge": get_merge_configs,
}


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    backend: str
    ms: float


def _make_mps_tensors(cfg: FeatureBench):
    dtype = torch.bfloat16 if cfg.dtype_name == "bf16" else torch.float32
    shape_q = (cfg.batch,) + cfg.spatial + (cfg.heads_q, cfg.head_dim)
    shape_kv = (cfg.batch,) + cfg.spatial + (cfg.heads_kv, cfg.head_dim)
    q = torch.randn(shape_q, device="mps", dtype=dtype)
    k = torch.randn(shape_kv, device="mps", dtype=dtype)
    v = torch.randn(shape_kv, device="mps", dtype=dtype)
    add_k = add_v = None
    if cfg.n_extra > 0:
        add_k = torch.randn(cfg.batch, cfg.n_extra, cfg.heads_kv, cfg.head_dim, device="mps", dtype=dtype)
        add_v = torch.randn(cfg.batch, cfg.n_extra, cfg.heads_kv, cfg.head_dim, device="mps", dtype=dtype)
    return q, k, v, add_k, add_v


def run_mps(cfg: FeatureBench) -> Optional[BenchResult]:
    if not HAS_MPS:
        return None
    try:
        natten_mps.set_backend("metal")
    except (NotImplementedError, Exception):
        natten_mps.set_backend("auto")

    na_fn = mps_na1d if cfg.dim == "1d" else mps_na2d
    q, k, v, add_k, add_v = _make_mps_tensors(cfg)

    kwargs: Dict[str, Any] = {
        "kernel_size": cfg.kernel_size,
        "return_lse": cfg.return_lse,
    }
    if cfg.stride != 1:
        kwargs["stride"] = cfg.stride
    if cfg.is_causal:
        kwargs["is_causal"] = cfg.is_causal
    if add_k is not None:
        kwargs["additional_keys"] = add_k
        kwargs["additional_values"] = add_v

    if cfg.feature == "merge":
        # Special: run two attentions + merge
        def bench_fn():
            out1, lse1 = na_fn(q, k, v, return_lse=True, kernel_size=cfg.kernel_size)
            out2, lse2 = na_fn(q, k, v, return_lse=True, kernel_size=cfg.kernel_size)
            merged_out, merged_lse = mps_merge([out1, out2], [lse1, lse2], use_autograd_fix=False)
            return merged_out
    else:
        def bench_fn():
            return na_fn(q, k, v, **kwargs)

    for _ in range(WARMUP):
        bench_fn()
        _sync_mps()

    times = []
    for _ in range(REPEATS):
        _sync_mps()
        t0 = time.perf_counter()
        bench_fn()
        _sync_mps()
        times.append((time.perf_counter() - t0) * 1000)

    natten_mps.set_backend("auto")
    return BenchResult(backend="mps", ms=_median(times))


def _make_mlx_tensors(cfg: FeatureBench):
    dtype = mx.bfloat16 if cfg.dtype_name == "bf16" else mx.float32
    shape_q = (cfg.batch,) + cfg.spatial + (cfg.heads_q, cfg.head_dim)
    shape_kv = (cfg.batch,) + cfg.spatial + (cfg.heads_kv, cfg.head_dim)
    q = mx.random.normal(shape_q).astype(dtype)
    k = mx.random.normal(shape_kv).astype(dtype)
    v = mx.random.normal(shape_kv).astype(dtype)
    mx.eval(q, k, v)
    add_k = add_v = None
    if cfg.n_extra > 0:
        add_k = mx.random.normal((cfg.batch, cfg.n_extra, cfg.heads_kv, cfg.head_dim)).astype(dtype)
        add_v = mx.random.normal((cfg.batch, cfg.n_extra, cfg.heads_kv, cfg.head_dim)).astype(dtype)
        mx.eval(add_k, add_v)
    return q, k, v, add_k, add_v


def run_mlx(cfg: FeatureBench) -> Optional[BenchResult]:
    if not HAS_MLX:
        return None
    mlx_set_backend("fast_metal")

    na_fn = mlx_na1d if cfg.dim == "1d" else mlx_na2d
    q, k, v, add_k, add_v = _make_mlx_tensors(cfg)

    kwargs: Dict[str, Any] = {
        "kernel_size": cfg.kernel_size,
        "return_lse": cfg.return_lse,
    }
    if cfg.stride != 1:
        kwargs["stride"] = cfg.stride
    if cfg.is_causal:
        kwargs["is_causal"] = cfg.is_causal
    if add_k is not None:
        kwargs["additional_keys"] = add_k
        kwargs["additional_values"] = add_v

    if cfg.feature == "merge":
        def bench_fn():
            out1, lse1 = na_fn(q, k, v, return_lse=True, kernel_size=cfg.kernel_size)
            out2, lse2 = na_fn(q, k, v, return_lse=True, kernel_size=cfg.kernel_size)
            merged_out, merged_lse = mlx_merge([out1, out2], [lse1, lse2])
            mx.eval(merged_out, merged_lse)
    else:
        def bench_fn():
            result = na_fn(q, k, v, **kwargs)
            if isinstance(result, tuple):
                mx.eval(*result)
            else:
                mx.eval(result)

    for _ in range(WARMUP):
        bench_fn()

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        bench_fn()
        times.append((time.perf_counter() - t0) * 1000)

    mlx_set_backend("auto")
    return BenchResult(backend="mlx", ms=_median(times))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt_ms(ms: Optional[float]) -> str:
    if ms is None:
        return "n/a"
    if ms < 0.1:
        return f"{ms*1000:.0f}us"
    return f"{ms:.2f}ms"


def format_results(
    results: List[Tuple[FeatureBench, Optional[BenchResult], Optional[BenchResult]]],
) -> str:
    buf = io.StringIO()
    buf.write(f"\n{'='*85}\n")
    buf.write(f"{'Feature':<14} {'Config':<30} {'MPS':>9} {'MLX':>9} {'Ratio':>8}  Note\n")
    buf.write(f"{'-'*85}\n")

    current_feature = None
    for cfg, mps_r, mlx_r in results:
        if cfg.feature != current_feature:
            if current_feature is not None:
                buf.write(f"{'-'*85}\n")
            current_feature = cfg.feature

        mps_ms = mps_r.ms if mps_r else None
        mlx_ms = mlx_r.ms if mlx_r else None

        ratio = ""
        note = ""
        if mps_ms is not None and mlx_ms is not None:
            if mlx_ms < mps_ms:
                r = mps_ms / mlx_ms
                ratio = f"{r:.1f}x"
                note = "MLX wins"
            else:
                r = mlx_ms / mps_ms
                ratio = f"{r:.1f}x"
                note = "MPS wins"

        buf.write(
            f"{cfg.feature:<14} {cfg.name:<30} "
            f"{_fmt_ms(mps_ms):>9} {_fmt_ms(mlx_ms):>9} {ratio:>8}  {note}\n"
        )

    buf.write(f"{'='*85}\n")
    buf.write(f"\nMPS = natten-mps (PyTorch Metal)  |  MLX = natten-mlx (MLX Metal)\n")
    buf.write(f"Warmup={WARMUP}  Repeats={REPEATS}\n")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Benchmark new features: GQA, LSE, additional_kv, bf16, FMHA, merge")
    parser.add_argument("--features", nargs="+", choices=list(ALL_FEATURES.keys()), default=list(ALL_FEATURES.keys()))
    parser.add_argument("--json", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    print(f"natten-mps:  {HAS_MPS}")
    print(f"natten-mlx:  {HAS_MLX}")

    configs: List[FeatureBench] = []
    for feat in args.features:
        configs += ALL_FEATURES[feat]()

    print(f"\nRunning {len(configs)} benchmarks across {len(args.features)} features\n")

    results = []
    for i, cfg in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] {cfg.name} ...", end="", flush=True)
        mps_r = run_mps(cfg)
        mlx_r = run_mlx(cfg)

        parts = []
        if mps_r:
            parts.append(f"MPS={_fmt_ms(mps_r.ms)}")
        if mlx_r:
            parts.append(f"MLX={_fmt_ms(mlx_r.ms)}")
        print(f" {' '.join(parts)}")

        results.append((cfg, mps_r, mlx_r))

    print(format_results(results))

    if args.json:
        json_data = []
        for cfg, mps_r, mlx_r in results:
            entry = {
                "name": cfg.name,
                "feature": cfg.feature,
                "dim": cfg.dim,
                "mps_ms": mps_r.ms if mps_r else None,
                "mlx_ms": mlx_r.ms if mlx_r else None,
            }
            json_data.append(entry)
        with open(args.json, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
