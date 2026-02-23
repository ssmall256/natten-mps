import importlib

import pytest

from natten_mps import get_backend, set_backend


def test_set_and_get_backend_pure():
    set_backend("pure")
    assert get_backend() == "pure"


def test_set_backend_metal():
    set_backend("metal")
    assert get_backend() == "metal"


def test_auto_backend_selects_metal():
    set_backend("auto")
    assert get_backend() == "metal"


def test_env_backend_override_respected_on_ops_reload(monkeypatch):
    import natten_mps._core.ops as ops

    monkeypatch.setenv("NATTEN_BACKEND", "pure")
    ops = importlib.reload(ops)
    assert ops.get_backend() == "pure"

    monkeypatch.delenv("NATTEN_BACKEND", raising=False)
    importlib.reload(ops)
