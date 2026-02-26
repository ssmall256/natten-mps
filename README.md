# natten-mps

GPU-accelerated Neighborhood Attention for Apple Silicon, built on PyTorch MPS.

## Why this exists

Upstream NATTEN provides CUDA kernels but dropped macOS support after v0.14. If you train or fine-tune models that use neighborhood attention on a Mac — or want to run inference on Apple Silicon without a CUDA GPU — there was no GPU-accelerated option.

`natten-mps` fills that gap with Metal compute shaders for PyTorch MPS, covering 1D, 2D, and 3D neighborhood attention with full autograd support. For MLX-based workflows, see the sibling project [natten-mlx](https://github.com/ssmall256/natten-mlx).

[Installation](#installation) | [Quick start](#quick-start) | [Features](#features) | [Performance](#performance) | [Backend tiers](#backend-tiers) | [Limitations](#limitations) | [Acknowledgments](#acknowledgments) | [License](#license)

## Installation

```bash
pip install natten-mps
```

Requires Python 3.10+ and PyTorch 2.8+ with MPS support (macOS 12.3+).

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
- **Variable-length (varlen)** attention — padded batches with per-sample spatial sizes, Metal-accelerated for all ranks
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

### Variable-length attention

```python
import torch
from natten_mps import na1d_varlen, na2d_varlen

# 1D: padded batch with per-sample lengths
q = torch.randn(3, 128, 4, 32, device="mps")  # B=3, L_max=128
k = torch.randn(3, 128, 4, 32, device="mps")
v = torch.randn(3, 128, 4, 32, device="mps")
seq_lens = torch.tensor([128, 96, 64], device="mps")
out = na1d_varlen(q, k, v, kernel_size=7, seq_lens=seq_lens)

# 2D: padded batch with per-sample (H, W)
q2d = torch.randn(2, 32, 32, 4, 32, device="mps")  # B=2, H_max=32, W_max=32
k2d = torch.randn(2, 32, 32, 4, 32, device="mps")
v2d = torch.randn(2, 32, 32, 4, 32, device="mps")
spatial_sizes = torch.tensor([[32, 32], [24, 20]], device="mps")
out2d = na2d_varlen(q2d, k2d, v2d, kernel_size=7, spatial_sizes=spatial_sizes)
```

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

Apple Silicon (M-series), fp32, B=1 H=4 D=32, Metal-accelerated:

| Config | natten-mps fwd | natten-mlx fwd | natten-mps bwd | natten-mlx bwd |
|---|---|---|---|---|
| 1D L=256 K=7 | 0.25 ms | 0.21 ms | 0.39 ms | 0.14 ms |
| 1D L=1024 K=7 | 0.40 ms | 0.27 ms | 0.63 ms | 0.26 ms |
| 2D 32×32 K=7 | 0.88 ms | 0.65 ms | 1.62 ms | 1.02 ms |
| 2D 64×64 K=7 | 1.32 ms | 1.13 ms | 1.55 ms | 0.97 ms |
| 2D 32×32 K=7 causal | 0.37 ms | 0.29 ms | 0.49 ms | 0.31 ms |
| 3D 16³ K=3 | 0.55 ms | 0.43 ms | 0.89 ms | 0.50 ms |

MLX's compiled Metal primitives have lower dispatch overhead than PyTorch MPS, giving a consistent 1.2–1.5× forward advantage. Both are orders of magnitude faster than pure-framework baselines.

### Variable-length (varlen) attention

Metal-accelerated varlen forward, fp32:

| Config | natten-mps | natten-mlx | MLX speedup |
|---|---|---|---|
| varlen 1D B=4 L=128 K=7 | 1.74 ms | 0.53 ms | 3.3× |
| varlen 1D B=4 L=256 K=7 | 1.74 ms | 0.51 ms | 3.4× |
| varlen 2D B=2 16×16 K=3 | 2.39 ms | 0.82 ms | 2.9× |
| varlen 2D B=2 32×32 K=7 | 3.79 ms | 1.23 ms | 3.1× |
| varlen 3D B=2 8³ K=3 | 3.82 ms | 1.55 ms | 2.5× |

Both projects now support GPU-accelerated varlen for all ranks (1D/2D/3D). Backward pass uses per-sample autograd re-differentiation through the standard Metal-accelerated `na*d` kernels.

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
| `metal` | Complete | 108 Metal compute shaders via `torch.mps.compile_shader` |
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

## Limitations

- Metal kernel acceleration requires odd kernel sizes (1D: K≤63, 2D: K≤13, 3D: K≤7).
- Unsupported kernel sizes or configurations fall back to pure PyTorch.
- macOS only (Apple Silicon required for Metal backend; CPU fallback works anywhere PyTorch runs).

## Differences from NATTEN

- No CUDA backend — targets MPS/CPU on Apple Silicon
- Metal compute shaders instead of CUDA kernels
- Native non-uniform per-axis kernel sizes and dilations

## Acknowledgments

This project implements the neighborhood attention mechanism introduced by [NATTEN](https://github.com/SHI-Labs/NATTEN) (SHI-Labs), ported to PyTorch MPS with custom Metal kernels. The original NATTEN library and the research behind it are by Ali Hassani, Steven Walton, Humphrey Shi, and collaborators.

If you use neighborhood attention in research, please cite the original papers:

- Hassani et al., "Neighborhood Attention Transformer" (CVPR 2023)
- Hassani & Shi, "Dilated Neighborhood Attention Transformer" (2022)
- Hassani et al., "Faster Neighborhood Attention" (NeurIPS 2024)

## License

MIT — see [LICENSE](LICENSE) for details.

NATTEN (the original PyTorch library) is also MIT-licensed.
