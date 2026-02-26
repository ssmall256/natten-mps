# natten-mps

GPU-accelerated Neighborhood Attention for Apple Silicon — built on **PyTorch MPS**.

> **Disclaimer (unofficial):** This is an independent, unofficial implementation/port for Apple Silicon.  
> **Not affiliated with** SHI-Labs or the upstream [NATTEN](https://github.com/SHI-Labs/NATTEN) project.

This is a focused, Apple-Silicon-first implementation intended to be useful, correct, and easy to install — not a replacement for upstream NATTEN on CUDA.

Neighborhood Attention was introduced by the NATTEN authors. If you use Neighborhood Attention in research, please cite the original papers (see [Acknowledgments](#acknowledgments)).

> **v0.x** — API may change between minor versions. Pin your dependency for production use.

---

## Why this exists

Upstream NATTEN is CUDA-focused and targets NVIDIA GPUs. On Apple Silicon, PyTorch users often want a **GPU-accelerated** neighborhood attention option without requiring CUDA.

**natten-mps** provides:
- **Metal-backed kernels** for PyTorch MPS using `torch.mps.compile_shader`
- **1D / 2D / 3D** neighborhood attention with **full autograd support**
- A deployment story that is intentionally simple: **no native extension build step** — install from PyPI and go. Metal shaders are compiled at runtime via `torch.mps.compile_shader` and cached by PyTorch for the process (best effort).

For MLX-based workflows, see the sibling project: **[natten-mlx](https://github.com/ssmall256/natten-mlx)**.

**Jump to:** [Installation](#installation) | [Quick start](#quick-start) | [Features](#features) | [Backends](#backends) | [Performance](#performance) | [Limitations](#limitations) | [Acknowledgments](#acknowledgments)

---

## Use natten-mps if…

- You’re using **PyTorch**
- You run on **Apple Silicon** and want **MPS (Metal) acceleration**
- You want a drop-in-ish API (plus optional compatibility shims for historical NATTEN versions)

---

## Installation

```bash
pip install natten-mps
```

Requirements:
- Python 3.10+
- PyTorch 2.8+ with MPS support
- macOS 12.3+ for MPS (CPU fallback works anywhere PyTorch runs)

---

## Quick start

### Functional API

```python
import torch
from natten_mps import na1d, na2d, na3d

# 1D: [B, L, heads, head_dim]
q = torch.randn(2, 128, 4, 32, device="mps")
k = torch.randn(2, 128, 4, 32, device="mps")
v = torch.randn(2, 128, 4, 32, device="mps")
out = na1d(q, k, v, kernel_size=7)

# 2D: [B, H, W, heads, head_dim]
q2d = torch.randn(2, 32, 32, 4, 32, device="mps")
k2d = torch.randn(2, 32, 32, 4, 32, device="mps")
v2d = torch.randn(2, 32, 32, 4, 32, device="mps")
out2d = na2d(q2d, k2d, v2d, kernel_size=7)

# 3D: [B, D, H, W, heads, head_dim]
q3d = torch.randn(1, 8, 8, 8, 4, 32, device="mps")
k3d = torch.randn(1, 8, 8, 8, 4, 32, device="mps")
v3d = torch.randn(1, 8, 8, 8, 4, 32, device="mps")
out3d = na3d(q3d, k3d, v3d, kernel_size=3)
```

### Module API

```python
import torch
from natten_mps import NeighborhoodAttention2D

layer = NeighborhoodAttention2D(embed_dim=128, num_heads=4, kernel_size=(7, 7)).to("mps")
x = torch.randn(2, 32, 32, 128, device="mps")  # [B, H, W, C]
y = layer(x)
```

### Split QK / AV (access attention weights)

```python
import torch
from natten_mps import na1d_qk, na1d_av

B, L, H, D = 2, 128, 4, 32
q = torch.randn(B, L, H, D, device="mps")
k = torch.randn(B, L, H, D, device="mps")
v = torch.randn(B, L, H, D, device="mps")

logits = na1d_qk(q, k, kernel_size=7, scale=D ** -0.5)  # [B, L, H, K]
attn = torch.softmax(logits, dim=-1)
out = na1d_av(attn, v, kernel_size=7)                   # [B, L, H, D]
```

---

## Features

Core:
- **1D / 2D / 3D** neighborhood attention (fused and split QK/AV ops)
- **Causal masking**, including per-axis control (e.g. `is_causal=(True, False)` for 2D)
- **Strided output** for downsampling (e.g. `stride=2`)
- **Combined causal + stride** in one kernel
- **Non-uniform kernels** for 2D/3D (per-axis kernel sizes and dilations)

Batching / advanced:
- **Variable-length (varlen) attention** — padded batches with per-sample spatial sizes, Metal-accelerated for all ranks
- **GQA / MQA** (`num_kv_heads`) for grouped-query attention patterns
- **additional_keys / additional_values** — prepend extra global tokens that every query attends to
- **merge_attentions** — numerically stable sigmoid-based merge of multiple attention outputs
- **FMHA fast path** — when the kernel covers the full spatial extent, can dispatch to efficient full attention

Extras:
- **`extras/`** namespace for model-specific fused kernels (e.g., DiNAT-style fused QK+RPB paths)

Compatibility:
- Optional **compat shims** for historical upstream API versions (see [Compatibility shims](#compatibility-shims))

---

## Backends

Backend dispatch is controlled at runtime and does not require a native extension.

| Backend | Status | Description |
|---|---|---|
| `pure`  | Complete | Pure PyTorch fallback (CPU/MPS) |
| `metal` | Complete | Metal compute shaders via `torch.mps.compile_shader` |
| `auto`  | Default  | Select best available backend for the configuration |

```python
import natten_mps

natten_mps.set_backend("metal")  # "auto" (default), "metal", or "pure"
print(natten_mps.get_backend())
```

Or via environment variable:

```bash
NATTEN_BACKEND=metal python my_script.py   # "auto" (default), "metal", or "pure"
```

---

## Performance

Metal kernels vs pure-PyTorch backend on Apple Silicon (M-series), forward pass:

| Benchmark | Metal | Pure | Speedup |
|---|---:|---:|---:|
| 1D, L=256, K=7 | 0.9 ms | 9.8 ms | **11×** |
| 1D, L=1024, K=7 | 1.1 ms | 37 ms | **34×** |
| 2D, 32×32, K=7 | 1.3 ms | 20 ms | **15–17×** |
| 2D, 64×64, K=7 | 2.9 ms | 84 ms | **29×** |
| 2D, 32×32, K=7, causal | 1.1 ms | 21 ms | **19×** |
| 3D, 16³, K=3 | 1.7 ms | 12 ms | **7×** |

Run the full suite:
```bash
python benchmarks/bench.py
# add --backward to time backward pass
```

### Cross-framework: natten-mps vs natten-mlx

Apple Silicon (M-series), fp32, B=1 H=4 D=32, Metal-accelerated:

| Config | natten-mps fwd | natten-mlx fwd | natten-mps bwd | natten-mlx bwd |
|---|---:|---:|---:|---:|
| 1D L=256 K=7 | 0.25 ms | 0.21 ms | 0.39 ms | 0.14 ms |
| 1D L=1024 K=7 | 0.40 ms | 0.27 ms | 0.63 ms | 0.26 ms |
| 2D 32×32 K=7 | 0.88 ms | 0.65 ms | 1.62 ms | 1.02 ms |
| 2D 64×64 K=7 | 1.32 ms | 1.13 ms | 1.55 ms | 0.97 ms |
| 2D 32×32 K=7 causal | 0.37 ms | 0.29 ms | 0.49 ms | 0.31 ms |
| 3D 16³ K=3 | 0.55 ms | 0.43 ms | 0.89 ms | 0.50 ms |

MLX’s compiled primitives tend to have lower dispatch overhead than PyTorch MPS, so natten-mlx is often faster for the same shapes. Both are dramatically faster than pure-framework baselines.

### Variable-length (varlen) attention

Metal-accelerated varlen forward, fp32:

| Config | natten-mps | natten-mlx | MLX speedup |
|---|---:|---:|---:|
| varlen 1D B=4 L=128 K=7 | 1.74 ms | 0.53 ms | 3.3× |
| varlen 1D B=4 L=256 K=7 | 1.74 ms | 0.51 ms | 3.4× |
| varlen 2D B=2 16×16 K=3 | 2.39 ms | 0.82 ms | 2.9× |
| varlen 2D B=2 32×32 K=7 | 3.79 ms | 1.23 ms | 3.1× |
| varlen 3D B=2 8³ K=3 | 3.82 ms | 1.55 ms | 2.5× |

Backward pass uses per-sample autograd re-differentiation through the standard Metal-accelerated `na*d` kernels.

### Methodology

All timings on **Apple M4 Max**, macOS 26.3, Python 3.11, PyTorch 2.10, float32. Each kernel is warmed up for 5 iterations, then timed for 20 repetitions with `torch.mps.synchronize()` gating; the reported value is the **median**. Reproduce with `python benchmarks/bench.py`.

---

## Compatibility shims

If you have downstream code written against historical upstream APIs, natten-mps includes optional shims:

```python
import natten_mps.compat.v014 as natten014
import natten_mps.compat.v017 as natten017
import natten_mps.compat.v020 as natten020
```

These are best-effort drop-in replacements for common upstream `natten` entry points.

---

## Extras: model-specific fused kernels

Example: fused DiNAT-style ops with relative position bias:

```python
from natten_mps.extras.allin1 import (
    na1d_qk_rpb, na1d_av_fused,
    na2d_qk_rpb, na2d_av_fused,
)
```

---

## Limitations

- **Odd kernel sizes only** for accelerated Neighborhood Attention (this matches upstream NATTEN’s neighborhood half-width formulation).  
- Metal kernel acceleration has size caps tuned for performance:
  - 1D: K ≤ 63
  - 2D: K ≤ 13
  - 3D: K ≤ 7
- Unsupported kernel sizes or configurations automatically fall back to `pure`.
- **Supported dtypes:** Metal kernels run in float32 and float16. Bfloat16 inputs are accepted but upcast to float32 internally. Other dtypes fall back to `pure`.
- MPS acceleration is **macOS-only** (CPU fallback works anywhere PyTorch runs).

---

## Differences from upstream NATTEN (high level)

- Targets **Apple Silicon** (PyTorch **MPS** + CPU fallback); no CUDA backend
- Uses **Metal compute shaders** instead of CUDA kernels
- Includes Apple-Silicon-focused extras (and optional compatibility shims)

---

## Acknowledgments

This project implements Neighborhood Attention as introduced by the upstream [NATTEN](https://github.com/SHI-Labs/NATTEN) project (SHI-Labs). The original NATTEN library and research are by Ali Hassani, Steven Walton, Humphrey Shi, and collaborators.

If you use Neighborhood Attention in research, please cite the original papers:

- Hassani et al., **Neighborhood Attention Transformer** (CVPR 2023)
- Hassani & Shi, **Dilated Neighborhood Attention Transformer** (2022)
- Hassani et al., **Faster Neighborhood Attention** (NeurIPS 2024)

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{hassani2023neighborhood,
  title   = {Neighborhood Attention Transformer},
  author  = {Hassani, Ali and Walton, Steven and Li, Jiachen and Li, Shen and Shi, Humphrey},
  booktitle = {CVPR},
  year    = {2023}
}

@article{hassani2022dilated,
  title   = {Dilated Neighborhood Attention Transformer},
  author  = {Hassani, Ali and Shi, Humphrey},
  journal = {arXiv preprint arXiv:2209.15001},
  year    = {2022}
}

@inproceedings{hassani2024faster,
  title   = {Faster Neighborhood Attention: Reducing the O(n^2) Cost of Self Attention at the Threadblock Level},
  author  = {Hassani, Ali and Ke, Wen-Mei and Gong, Jiaming and Walton, Steven and Shi, Humphrey},
  booktitle = {NeurIPS},
  year    = {2024}
}
```
</details>

---

## License

MIT — see [LICENSE](LICENSE) for details.  
Upstream NATTEN is also MIT-licensed.
