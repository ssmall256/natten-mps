# Changelog

All notable changes to natten-mps are documented here.

## [0.3.0] — 2026-02-25

### Added
- Variable-length (varlen) attention for 1D, 2D, and 3D with Metal acceleration.
- Fused SIMD backward kernels for improved backward pass performance.
- `torch.library` custom op registration for `torch.compile` compatibility.
- GitHub Actions CI pipeline.
- Nanobind extension stubs (reserved for future use).

## [0.2.0] — 2026-02-24

### Added
- `return_lse` parameter for log-sum-exp output.
- `merge_attentions` for numerically stable attention merging.
- GQA / MQA support via mismatched head counts and `num_kv_heads` in nn modules.
- `additional_keys` / `additional_values` for global token prepending.
- bfloat16 support.
- FMHA fast path (auto-dispatch to `F.scaled_dot_product_attention`).
- `extras.allin1` module for DiNAT fused Metal kernels.
- Comprehensive test suite (254+ tests).

## [0.1.0] — 2026-02-24

### Added
- Initial release with pure PyTorch and Metal compute shader backends.
- 1D, 2D, and 3D neighborhood attention (fused and split QK/AV).
- Metal backward kernels with CSR inverse-map dispatch.
- Causal masking with per-axis control.
- Strided output for downsampling.
- Non-uniform per-axis kernel sizes and dilations.
- Compatibility shims for NATTEN v0.14, v0.17, and v0.20.
- Automatic backend selection (metal > pure).
