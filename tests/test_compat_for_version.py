"""Test the for_version() compat dispatcher."""

import pytest

from natten_mps.compat import for_version


class TestForVersion:
    def test_v014_returns_v014_module(self):
        mod = for_version("0.14.0")
        assert hasattr(mod, "NeighborhoodAttention1D")
        assert hasattr(mod, "NeighborhoodAttention2D")

    def test_v014_6_returns_v014_module(self):
        mod = for_version("0.14.6")
        assert hasattr(mod, "NeighborhoodAttention1D")

    def test_v015_returns_v017_module(self):
        mod = for_version("0.15.0")
        assert hasattr(mod, "NeighborhoodAttention1D")
        assert hasattr(mod, "NeighborhoodAttention2D")

    def test_v017_returns_v017_module(self):
        mod = for_version("0.17.0")
        assert hasattr(mod, "NeighborhoodAttention1D")

    def test_v020_returns_v020_module(self):
        mod = for_version("0.20.0")
        assert hasattr(mod, "na1d")
        assert hasattr(mod, "na2d")
        assert hasattr(mod, "na3d")

    def test_v020_has_3d(self):
        mod = for_version("0.20.0")
        assert hasattr(mod, "na3d")
        assert hasattr(mod, "na3d_qk")
        assert hasattr(mod, "na3d_av")

    def test_different_versions_different_modules(self):
        m014 = for_version("0.14.0")
        m020 = for_version("0.20.0")
        assert m014 is not m020
