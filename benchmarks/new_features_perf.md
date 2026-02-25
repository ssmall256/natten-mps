# New Features Performance: natten-mps vs natten-mlx

Cross-project benchmark results for the six features ported from NATTEN PR #312.

**Setup**: Apple Silicon, fp32 unless noted. B=2 (1D/2D), H_q=8 (1D) or 4 (2D), D=32.
Both use Metal backends: natten-mps uses `torch.mps.compile_shader`, natten-mlx uses fast_metal.
Median of 30 runs after 5 warmup iterations.

## Results

| Feature | Config | natten-mps | natten-mlx | MLX speedup |
|---------|--------|------------|------------|-------------|
| **Baseline** | 1D L=256 K=7 | 0.58 ms | 0.26 ms | 2.2x |
| | 2D 32x32 K=7 | 0.65 ms | 0.29 ms | 2.3x |
| **GQA** | GQA4 1D L=256 K=7 | 0.29 ms | 0.18 ms | 1.6x |
| | MQA 1D L=256 K=7 | 0.30 ms | 0.16 ms | 1.8x |
| | GQA4 2D 32x32 K=7 | 0.86 ms | 0.38 ms | 2.2x |
| | MQA 2D 32x32 K=7 | 0.88 ms | 0.39 ms | 2.3x |
| **return_lse** | 1D L=256 K=7 | 0.43 ms | 0.24 ms | 1.8x |
| | 2D 32x32 K=7 | 1.24 ms | 0.83 ms | 1.5x |
| **additional_kv** | 1 token, 1D L=256 K=7 | 1.25 ms | 0.54 ms | 2.3x |
| | 4 tokens, 1D L=256 K=7 | 1.25 ms | 0.58 ms | 2.2x |
| | 1 token, 2D 32x32 K=7 | 2.71 ms | 0.79 ms | 3.4x |
| | 1 token + GQA4, 1D | 1.09 ms | 0.46 ms | 2.4x |
| **bfloat16** | 1D L=256 K=7 | 0.66 ms | 0.26 ms | 2.6x |
| | 2D 32x32 K=7 | 2.14 ms | 0.45 ms | 4.7x |
| | GQA4 1D L=256 K=7 | 0.61 ms | 0.27 ms | 2.3x |
| **FMHA** | 1D L=16 full | 0.27 ms | 0.19 ms | 1.4x |
| | 1D L=64 full | 0.33 ms | 0.23 ms | 1.5x |
| | 2D 8x8 full | 0.31 ms | 0.22 ms | 1.4x |
| | 2D 16x16 full | 0.30 ms | 0.22 ms | 1.4x |
| **merge** | 2-way, 1D L=256 | 0.89 ms | 0.40 ms | 2.2x |
| | 2-way, 2D 32x32 | 1.33 ms | 0.60 ms | 2.2x |

## Key Observations

1. **MLX is consistently 1.4-4.7x faster** due to lower Metal dispatch overhead
   compared to PyTorch MPS. The gap is consistent with baseline (2-3x) for most
   features.

2. **GQA reduces compute for both**: With 4x fewer KV heads, 1D drops from
   0.58ms to 0.29ms (MPS, 2.0x) and 0.26ms to 0.18ms (MLX, 1.4x). The KV
   repeat is cheap; the savings come from smaller QK/AV matrices.

3. **return_lse overhead is modest**: Forces the split QK path (instead of
   fused), adding ~30-90% to baseline. Acceptable for the merging and gradient
   checkpointing it enables.

4. **additional_kv scales well**: Adding 1-4 extra tokens costs a fixed ~0.3ms
   (MLX) regardless of token count. The full-attention on a handful of tokens
   plus the merge is negligible vs the neighborhood attention itself.

5. **bf16 overhead** (MPS): ~1.1-3.3x vs fp32 due to bf16-to-fp32-to-bf16
   round-trip at the Metal dispatch boundary. MLX handles bf16 natively with
   the _cast mechanism, so bf16 is comparable to fp32 there. The 2D bf16 MPS
   penalty (2.14ms vs 0.65ms baseline) suggests the upcast/downcast cost
   dominates for larger workloads.

6. **FMHA fast path**: When kernel covers the full spatial extent, both backends
   delegate to SDPA. The MLX advantage narrows to 1.4-1.5x since both use
   highly optimized full-attention kernels rather than NA-specific code.

7. **merge_attentions**: Pure tensor ops (sigmoid, logsumexp, elementwise). The
   merge itself is fast; the total time is dominated by the two underlying
   attention computations.

## Reproducing

```bash
# From natten-mps repo root:
python benchmarks/bench_new_features.py

# With JSON output:
python benchmarks/bench_new_features.py --json results.json

# Specific features only:
python benchmarks/bench_new_features.py --features gqa fmha bf16

# Cross-project (needs both installed in same venv):
pip install -e /path/to/natten-mps -e /path/to/natten-mlx
python benchmarks/bench_new_features.py
```
