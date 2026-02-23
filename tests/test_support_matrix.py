from natten_mps.support_matrix import get_support_matrix


def test_get_support_matrix_returns_dict_with_expected_keys():
    matrix = get_support_matrix()
    assert isinstance(matrix, dict)
    assert "pure" in matrix
    assert "metal" in matrix
    assert "nanobind" in matrix


def test_pure_backend_reports_available():
    matrix = get_support_matrix()
    assert matrix["pure"]["available"] is True


def test_all_backends_have_3d_entries():
    matrix = get_support_matrix()
    for name, info in matrix.items():
        assert "na3d" in info["forward"], f"{name} missing na3d in forward"
        assert "na3d" in info["backward"], f"{name} missing na3d in backward"
        assert "na3d" in info["fusion"], f"{name} missing na3d in fusion"


def test_metal_available_nanobind_not():
    matrix = get_support_matrix()
    assert matrix["metal"]["available"] is True
    assert matrix["nanobind"]["available"] is False


def test_metal_reports_forward_backward_support():
    matrix = get_support_matrix()
    metal = matrix["metal"]
    assert metal["forward"]["na1d"] is True
    assert metal["forward"]["na2d"] is True
    assert metal["forward"]["na3d"] is True
    assert metal["forward"]["split_qk_av"] is True
    assert metal["backward"]["na1d"] is True
