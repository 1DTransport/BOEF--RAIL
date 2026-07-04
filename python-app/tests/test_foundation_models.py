import pytest

from core.foundation.base import equivalent_series_stiffness


def test_equivalent_series_stiffness_limits() -> None:
    k_pad = 1000.0
    k_bed = 1.0e9
    k_eq = equivalent_series_stiffness(k_pad, k_bed)
    assert k_eq == pytest.approx(k_pad, rel=1e-6)

    k_pad = 1.0e9
    k_bed = 2500.0
    k_eq = equivalent_series_stiffness(k_pad, k_bed)
    assert k_eq == pytest.approx(k_bed, rel=1e-5)
