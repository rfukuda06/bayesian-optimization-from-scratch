import numpy as np

from gpbo.model_selection import decode_parameters


def test_decode_parameters_maps_names_in_order():
    params = decode_parameters(np.array([0.5, -1.5]), ("log10_C", "log10_gamma"))
    assert params == {"log10_C": 0.5, "log10_gamma": -1.5}
    assert all(type(v) is float for v in params.values())
    assert list(params) == ["log10_C", "log10_gamma"]   # dimension order preserved
