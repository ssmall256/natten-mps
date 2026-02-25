# natten-mps

GPU-accelerated Neighborhood Attention for Apple Silicon, built on PyTorch MPS.

## Why this exists

Upstream NATTEN provides CUDA kernels but dropped macOS support. `natten-mps` delivers Metal compute shader acceleration for 1D, 2D, and 3D neighborhood attention on Apple Silicon, with a pure-PyTorch CPU fallback.

## Installation

```bash
pip install natten-mps
```

## Quick start

```python
import torch
from natten_mps import na1d, na2d, na3d, NeighborhoodAttention2D, NeighborhoodAttention3D

# 1D neighborhood attention
q = torch.randn(2, 128, 4, 32, device="mps")
k = torch.randn(2, 128, 4, 32, device="mps")
v = torch.randn(2, 128, 4, 32, device="mps")
out = na1d(q, k, v, kernel_size=7)

# 2D module
layer2d = NeighborhoodAttention2D(embed_dim=128, num_heads=4, kernel_size=(7, 7)).to("mps")
x = torch.randn(2, 32, 32, 128, device="mps")
y = layer2d(x)

# 3D neighborhood attention
q3d = torch.randn(1, 8, 8, 8, 4, 32, device="mps")
k3d = torch.randn(1, 8, 8, 8, 4, 32, device="mps")
v3d = torch.randn(1, 8, 8, 8, 4, 32, device="mps")
out3d = na3d(q3d, k3d, v3d, kernel_size=3)
```

## Features

- **1D, 2D, 3D** neighborhood attention (fused and split QK/AV ops)
- **Causal masking** with per-axis control (e.g. `is_causal=(True, False)` for 2D)
- **Strided output** for downsampling (e.g. `stride=2`)
- **Combined causal + strided** in a single kernel
- **Non-uniform kernels** — per-axis kernel sizes and dilations for 2D/3D (e.g. `kernel_size=(3, 7)`)
- **Autograd** — forward and backward through Metal kernels
- **float32, float16, and bfloat16**
- **GQA / MQA** — grouped-query and multi-query attention via `num_kv_heads` (nn modules) or mismatched Q/KV head counts (functional API)
- **`return_lse`** — return log-sum-exp alongside output for gradient checkpointing and attention merging
- **`additional_keys` / `additional_values`** — prepend extra global tokens that every query attends to
- **`merge_attentions`** — numerically stable sigmoid-based merge of multiple attention outputs (for ring attention, sliding window + global, etc.)
- **FMHA fast path** — auto-dispatches to `F.scaled_dot_product_attention` when kernel covers the full spatial extent
- **Compatibility shims** for upstream NATTEN v0.14, v0.17, and v0.20

### New features usage

```python
import torch
from natten_mps import na1d, na2d, merge_attentions

# GQA: 8 query heads, 2 KV heads
q = torch.randn(1, 128, 8, 32, device="mps")
k = torch.randn(1, 128, 2, 32, device="mps")
v = torch.randn(1, 128, 2, 32, device="mps")
out = na1d(q, k, v, kernel_size=7)

# return_lse for merging
out1, lse1 = na1d(q, k, v, kernel_size=7, return_lse=True)
out2, lse2 = na1d(q, k, v, kernel_size=7, return_lse=True)
merged, merged_lse = merge_attentions([out1, out2], [lse1, lse2])

# Additional global tokens
add_k = torch.randn(1, 4, 2, 32, device="mps")
add_v = torch.randn(1, 4, 2, 32, device="mps")
out = na1d(q, k, v, kernel_size=7, additional_keys=add_k, additional_values=add_v)

# GQA via nn module
from natten_mps import NeighborhoodAttention1D
layer = NeighborhoodAttention1D(embed_dim=256, num_heads=8, kernel_size=7, num_kv_heads=2)
```

## Performance

Metal kernels vs pure-PyTorch backend on Apple Silicon (M-series), forward pass:

| Benchmark | Metal | Pure | Speedup |
|---|---|---|---|
| 1D, L=256, K=7 | 0.9 ms | 9.8 ms | **11x** |
| 1D, L=1024, K=7 | 1.1 ms | 37 ms | **34x** |
| 2D, 32x32, K=7 | 1.3 ms | 20 ms | **15–17x** |
| 2D, 64x64, K=7 | 2.9 ms | 84 ms | **29x** |
| 2D, 32x32, K=7, causal | 1.1 ms | 21 ms | **19x** |
| 3D, 16x16x16, K=3 | 1.7 ms | 12 ms | **7x** |

Run the full suite: `python benchmarks/bench.py` (add `--backward` for backward pass timing).

### Cross-framework: natten-mps vs natten-mlx

Apple Silicon (M-series), fp32, B=1 H=4 D=32:

| Config | natten-mps (MPS) | natten-mlx (MLX) |
|---|---|---|
| 1D L=256 K=7 fwd | 0.42 ms | 0.24 ms |
| 1D L=1024 K=7 fwd | 1.12 ms | 0.45 ms |
| 2D 32×32 K=7 fwd | 0.96 ms | 0.42 ms |
| 2D 64×64 K=7 fwd | 3.23 ms | 1.52 ms |
| 3D 16³ K=3 fwd | 0.56 ms | 0.22 ms |
| 1D L=256 K=7 bwd | 0.56 ms | 0.19 ms |
| 2D 32×32 K=7 bwd | 1.73 ms | 0.48 ms |

MLX's compiled Metal primitives are generally 2–3× faster than PyTorch MPS dispatch. Both are orders of magnitude faster than pure-framework baselines.

### Apple Silicon vs CUDA GPUs — backward pass

NATTEN's CUDA backward pass has known performance issues for 3D and large 2D workloads. Apple Silicon backward passes are competitive with — and sometimes faster than — datacenter GPUs:

| Config | natten-mps bwd | natten-mlx bwd | A100 CUDA bwd | A40 CUDA bwd |
|---|---|---|---|---|
| 3D 32³ K=3 | 12.4 ms | 5.7 ms | 458 ms (default) / 11.8 ms (KV-parallel) | — |
| 2D 1024² K=9 | — | — | — | 800–1041 ms |
| 3D 16³ K=3 | — | — | — | 3856 ms |

CUDA numbers sourced from NATTEN GitHub issues: [#157](https://github.com/SHI-Labs/NATTEN/issues/157) (A100/H100 3D backward) and [#161](https://github.com/SHI-Labs/NATTEN/issues/161) (A40 2D/3D backward). Our CSR inverse-map backward design avoids the scaling problems that affect NATTEN's default CUDA backward kernels.

## Backend tiers

| Backend | Status | Description |
|---|---|---|
| `pure` | Complete | Pure PyTorch, CPU and MPS |
| `metal` | Complete | 72 Metal compute shaders via `torch.mps.compile_shader` |
| `nanobind` | Stub | Reserved for future C++/nanobind acceleration |
| `auto` | Default | Selects the best available backend |

```python
import natten_mps
natten_mps.set_backend("metal")  # or "pure", "auto"
print(natten_mps.get_backend())
```

## Compatibility shims

```python
import natten_mps.compat.v014 as natten014
import natten_mps.compat.v017 as natten017
import natten_mps.compat.v020 as natten020
```

Drop-in replacements for downstream code that depends on upstream `natten` APIs.

## Extras: fused DiNAT ops

```python
from natten_mps.extras.allin1 import na1d_qk_rpb, na1d_av_fused, na2d_qk_rpb, na2d_av_fused
```

Fused QK+RPB and AV operations for DiNAT-style models with relative position bias.

## Differences from upstream NATTEN

- No CUDA backend — targets MPS/CPU on Apple Silicon
- Metal compute shaders instead of CUDA kernels
- Native non-uniform per-axis kernel sizes and dilations

## License

MIT
