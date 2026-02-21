# natten-mps

`natten-mps` is an independent implementation of Neighborhood Attention for Apple Silicon, focused on PyTorch MPS and CPU fallback.

## Why this exists

Upstream NATTEN provides CUDA-accelerated kernels, but modern releases dropped macOS CPU support. `natten-mps` targets Apple Silicon with a backend architecture designed for progressive acceleration.

## Installation

```bash
pip install natten-mps
```

## Quick start (modern API)

```python
import torch
from natten_mps import na1d, NeighborhoodAttention2D

q = torch.randn(2, 128, 4, 32)
k = torch.randn(2, 128, 4, 32)
v = torch.randn(2, 128, 4, 32)
out = na1d(q, k, v, kernel_size=7)

layer = NeighborhoodAttention2D(embed_dim=128, num_heads=4, kernel_size=(7, 7))
x = torch.randn(2, 32, 32, 128)
y = layer(x)
```

## Compatibility shims

```python
import natten_mps.compat.v014 as natten014
import natten_mps.compat.v017 as natten017
import natten_mps.compat.v020 as natten020
```

These modules provide API-era compatibility for downstream code that historically depended on `natten`.

## Backend tiers

- `pure`: full working Tier-0 backend in pure PyTorch
- `metal`: Tier-1 placeholder (stub)
- `nanobind`: Tier-2 placeholder (stub)
- `auto`: choose the best available backend

```python
import natten_mps
natten_mps.set_backend("pure")
```

## Differences from upstream NATTEN

- No CUDA backend
- No 3D neighborhood attention
- Focused on MPS/CPU compatibility
- Tier-1 and Tier-2 acceleration backends are scaffolding stubs in this release

## License

MIT
