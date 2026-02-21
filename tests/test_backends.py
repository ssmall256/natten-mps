import pytest

from natten_mps import get_backend, set_backend


def test_set_and_get_backend_pure():
    set_backend("pure")
    assert get_backend() == "pure"


def test_set_backend_metal_raises_when_unavailable():
    with pytest.raises(NotImplementedError):
        set_backend("metal")


def test_auto_backend_falls_back_to_pure():
    set_backend("auto")
    assert get_backend() == "pure"
