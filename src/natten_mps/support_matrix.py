"""Backend capability matrix for natten-mps."""

from __future__ import annotations

from natten_mps._core import metal, nanobind, pure


def get_support_matrix() -> dict[str, dict]:
    """Return capability matrix for each backend tier."""

    return {
        "pure": {
            "available": pure.is_available(),
            "forward": {"na1d": True, "na2d": True, "na3d": True, "split_qk_av": True},
            "backward": {"na1d": True, "na2d": True, "na3d": True, "split_qk_av": True},
            "varlen": {"na1d": True, "na2d": True, "na3d": True},
            "fusion": {"na1d": False, "na2d": False, "na3d": False},
            "constraints": [],
        },
        "metal": {
            "available": metal.is_available(),
            "forward": {"na1d": True, "na2d": True, "na3d": True, "split_qk_av": True},
            "backward": {"na1d": True, "na2d": True, "na3d": True, "split_qk_av": True},
            "varlen": {"na1d": True, "na2d": True, "na3d": True},
            "fusion": {"na1d": False, "na2d": False, "na3d": False},
            "constraints": [
                "Requires MPS device and torch.mps.compile_shader support.",
                "1D/2D/3D forward: fully GPU-accelerated via 108 Metal compute shaders.",
                "Supports causal masking, strided output, combined causal+strided, "
                "and non-uniform per-axis kernel sizes and dilations.",
                "Forward: base, causal, strided, and causal+strided all use Metal kernels.",
                "Backward: fully GPU-accelerated Metal kernels for all configs "
                "(base, causal, strided, causal+strided, non-uniform).",
                "Supports float32 and float16.",
            ],
        },
        "nanobind": {
            "available": nanobind.is_available(),
            "forward": {"na1d": False, "na2d": False, "na3d": False, "split_qk_av": False},
            "backward": {"na1d": False, "na2d": False, "na3d": False, "split_qk_av": False},
            "varlen": {"na1d": False, "na2d": False, "na3d": False},
            "fusion": {"na1d": False, "na2d": False, "na3d": False},
            "constraints": ["Nanobind backend is not yet available."],
        },
    }
