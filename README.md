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
- **float32 and float16**
- **Compatibility shims** for upstream NATTEN v0.14, v0.17, and v0.20

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
