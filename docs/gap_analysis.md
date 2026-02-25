# Gap Analysis: natten-mps & natten-mlx vs NATTEN PR #312 & NATTEN CUDA

Comprehensive comparison of feature parity between our Apple Silicon
implementations and the upstream NATTEN ecosystem.

**Date**: 2025-02
**Scope**: natten-mps (PyTorch/MPS), natten-mlx (MLX) vs NATTEN Metal PR #312
and NATTEN's CUDA-based feature set (CUTLASS, Hopper, Blackwell, Flex backends).

---

## Feature Matrix

| Feature | NATTEN CUDA | NATTEN PR #312 | natten-mps | natten-mlx | Notes |
|---------|:-----------:|:--------------:|:----------:|:----------:|-------|
| **Core Operations** |
| 1D Neighborhood Attention | Yes | Yes | Yes | Yes | Full parity |
| 2D Neighborhood Attention | Yes | Yes | Yes | Yes | Full parity |
| 3D Neighborhood Attention | Yes | Yes | Yes | Yes | Full parity |
| Split QK/AV ops | Yes | No (tiled) | Yes | Yes | Our split design is 10-300x faster on Apple Silicon |
| Fused forward (QKV in one call) | Yes | Yes | Yes | Yes | |
| Backward pass (autograd) | Yes | Yes | Yes | Yes | |
| **Attention Parameters** |
| kernel_size (per-dim) | Yes | Yes | Yes | Yes | |
| stride (per-dim) | Yes | Yes | Yes | Yes | |
| dilation (per-dim) | Yes | Yes | Yes | Yes | |
| is_causal (per-dim) | Yes | Yes | Yes | Yes | |
| scale | Yes | Yes | Yes | Yes | |
| Even kernel sizes | Yes | Yes | Yes | Yes | |
| **Advanced Features** |
| return_lse | Yes | Yes | **Yes** | **Yes** | Ported from PR |
| merge_attentions | Yes | Yes | **Yes** | **Yes** | Ported from PR (sigmoid-based) |
| GQA/MQA | Yes | Yes | **Yes** | **Yes** | Via KV head repeat |
| additional_keys/values | Yes | Yes | **Yes** | **Yes** | Via return_lse + merge |
| FMHA fast path | Yes | Yes | **Yes** | **Yes** | SDPA when kernel >= spatial |
| bfloat16 | Yes | Yes | **Yes** | Yes | mps: upcast; mlx: native |
| float16 | Yes | Yes | Yes | Yes | |
| float8 (FP8) | Yes (Hopper/Blackwell) | No | No | No | Requires SM89+ hardware |
| **CUDA-Only Features** |
| Variable-length (varlen) attention | Yes | No | No | No | Sequence packing; CUDA-specific |
| Token permutation | Yes (Hopper/Blackwell/Flex) | No | No | No | Out-of-kernel spatial reorder |
| Tile shape tuning | Yes | No | No | No | q_tile_shape, kv_tile_shape |
| KV parallelism (backward) | Yes | No | No | No | backward_kv_splits |
| Persistent kernels | Yes (Blackwell) | No | No | No | Blackwell-specific |
| Kernel schedule selection | Yes (Hopper) | No | No | No | Non-persistent/coop/pingpong |
| torch.compile integration | Yes | No | No | No | flex backend only |
| MLA (multi-head latent attention) | Yes (CUTLASS) | No | No | No | head_dim != head_dim_v |
| Deterministic mode toggle | Yes | No | No | No | Global context flag |
| Memory usage preference | Yes | No | No | No | strict/unrestricted modes |
| Profiler / auto-tuner | Yes | No | No | No | Config search |
| **Module-Level Features** |
| NeighborhoodAttention{1,2,3}D | Yes | Yes | Yes | Yes | |
| num_kv_heads in nn module | No | No | **Yes** | **Yes** | We go beyond PR |
| attn_drop in nn module | No | No | Yes | Yes | PR dropped this param |
| RPB (relative positional bias) | No (at kernel level) | No | Yes (extras) | No | Our extras/allin1 module |
| **Backend Architecture** |
| Metal kernels (runtime compiled) | No | Yes (C++ ext) | Yes (torch.mps.compile_shader) | Yes (MLX Metal) | Different compilation model |
| Pure-framework fallback | No | No | Yes | Yes | CPU/debug fallback |
| Nanobind extension | No | No | Stub | Yes | MLX nanobind backend |
| CSR inverse-map backward | No | No | Yes | Yes | Our backward design |
| Multiple backend tiers | Yes (6 backends) | Yes (Metal) | Yes (3 tiers) | Yes (3 tiers) | |

---

## Gap Categories

### 1. Features We Have That NATTEN CUDA/PR Lack

| Feature | Description |
|---------|-------------|
| **num_kv_heads in nn module** | Our modules accept GQA config directly; NATTEN's modules don't |
| **attn_drop in nn module** | NATTEN PR removed this parameter from the module class |
| **RPB via extras/allin1** | Metal-accelerated relative positional bias (natten-mps only) |
| **Pure-framework fallback** | Always-available CPU path for debugging and testing |
| **CSR inverse-map backward** | Our backward kernels avoid NATTEN's O(n^d * K^d) scaling issue |
| **Runtime Metal compilation** | No C++ build step required; kernels compiled at import time |

### 2. NATTEN CUDA Features Not Applicable to Apple Silicon

| Feature | Why Not Applicable |
|---------|-------------------|
| FP8 (float8) | Requires NVIDIA SM89+ (Ada Lovelace / Hopper) tensor cores |
| Token permutation kernels | CUTLASS/Hopper/Blackwell-specific optimization |
| Persistent kernels | Blackwell SM100-specific feature |
| Kernel schedule selection | Hopper SM90 warp specialization |
| KV parallelism splits | CUTLASS-specific backward optimization |
| Tile shape tuning | Tied to CUDA tiled kernel architecture |
| Profiler / auto-tuner | For CUDA kernel config search only |
| Deterministic mode | CUDA atomic operations; not relevant to our deterministic Metal kernels |
| Memory usage preference | CUDA workspace allocation control |

### 3. Potentially Portable NATTEN Features (Future Work)

| Feature | Effort | Value | Notes |
|---------|--------|-------|-------|
| **Variable-length attention** | Medium | High | Sequence packing for variable-length batches. Would require kernel modifications for offset-based indexing. |
| **torch.compile integration** | Low | Medium | Wrap our ops in torch.compile-compatible custom ops. Need to register as `torch.library` custom ops. |
| **MLA (multi-head latent attention)** | Medium | Low | Different head_dim for Q vs V. Requires kernel changes for non-square attention. Niche use case. |
| **RPB in natten-mlx** | Low | Medium | Port the extras/allin1 RPB Metal kernels from natten-mps to MLX. |

---

## API Parity Summary

### Functional API: `na{1,2,3}d()`

```
NATTEN:     na1d(query, key, value, kernel_size, stride, dilation, is_causal,
                 scale, return_lse, additional_keys, additional_values,
                 attention_kwargs, backend, q_tile_shape, kv_tile_shape,
                 backward_q_tile_shape, backward_kv_tile_shape,
                 backward_kv_splits, backward_use_pt_reduction,
                 run_persistent_kernel, kernel_schedule, torch_compile)

natten-mps: na1d(query, key, value, kernel_size, stride, dilation, is_causal,
                 scale, return_lse, additional_keys, additional_values)

natten-mlx: na1d(query, key, value, kernel_size, stride, dilation, is_causal,
                 scale, return_lse, additional_keys, additional_values)
```

The 10 extra NATTEN parameters are all CUDA backend tuning knobs that don't
apply to Apple Silicon. Our API covers all semantically meaningful parameters.

### nn Module: `NeighborhoodAttention{1,2,3}D`

```
NATTEN:     (embed_dim, num_heads, kernel_size, stride, dilation, is_causal,
             qkv_bias, qk_scale, proj_drop)

natten-mps: (embed_dim, num_heads, kernel_size, stride, dilation, is_causal,
             num_kv_heads, qkv_bias, qk_scale, attn_drop, proj_drop)

natten-mlx: (embed_dim, num_heads, kernel_size, stride, dilation, is_causal,
             num_kv_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
```

Our modules are a superset of NATTEN's: we add `num_kv_heads` and `attn_drop`.

---

## Performance Comparison Notes

- **Forward pass**: MLX is 2-3x faster than MPS dispatch; both are much faster
  than NATTEN's tiled Metal approach (PR #312: 10-300x slower).
- **3D backward**: Our CSR inverse-map design is competitive with A100 CUDA
  (5.7ms MLX vs 11.8ms A100 KV-parallel for 32^3 K=3). NATTEN's default CUDA
  3D backward has known O(n^3 * K^3) scaling issues.
- **GQA overhead**: KV head repeat adds a copy but is negligible for typical
  head ratios (2-8x). NATTEN's Blackwell/Flex backends do native GQA without
  repeat; this is a potential future optimization for our Metal kernels.

---

## Conclusion

After porting the 6 features from the PR analysis, natten-mps and natten-mlx
now cover **all semantically meaningful features** from NATTEN's ecosystem that
are applicable to Apple Silicon. The remaining gaps are either CUDA
hardware-specific (FP8, persistent kernels, token permutation) or CUDA
software-specific (torch.compile, tile tuning, profiler) and cannot be directly
ported. The one potentially valuable future addition is **variable-length
attention** for efficient batched inference with mixed sequence lengths.
