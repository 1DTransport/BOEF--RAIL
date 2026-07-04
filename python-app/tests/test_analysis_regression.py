import math

from core.analysis import AnalysisInputs, compute_track_response
from core.model import PointLoad


def test_closed_form_regression_snapshot() -> None:
    inputs = AnalysisInputs(
        loads=[PointLoad(position_m=0.0, load_newtons=100_000.0)],
        foundation_modulus_n_per_m2=40_000_000.0,
        elastic_modulus_pa=210_000_000_000.0,
        moment_inertia_m4=3.05e-5,
        section_modulus_m3=3.2e-5,
        sleeper_spacing_m=0.6,
        sleeper_length_m=2.6,
        sleeper_width_m=0.25,
        sample_count=401,
    )

    result = compute_track_response(inputs)

    def max_abs(values: list[float]) -> tuple[float, float]:
        idx = max(range(len(values)), key=lambda i: abs(values[i]))
        return values[idx], result.x_m[idx]

    max_deflection, max_deflection_x = max_abs(result.deflection_m)
    max_moment, max_moment_x = max_abs(result.moment_nm)
    max_shear, max_shear_x = max_abs(result.shear_n)
    max_reaction, max_reaction_x = max_abs(result.reaction_n_per_m)

    assert math.isclose(max_deflection, 0.0013972696616225775, rel_tol=1e-12)
    assert math.isclose(max_deflection_x, 0.0, abs_tol=1e-12)
    assert math.isclose(max_moment, 22365.04581636088, rel_tol=1e-12)
    assert math.isclose(max_moment_x, 0.0, abs_tol=1e-12)
    assert math.isclose(max_shear, -50000.0, rel_tol=1e-12)
    assert math.isclose(max_shear_x, 0.0, abs_tol=1e-12)
    assert math.isclose(max_reaction, 55890.7864649031, rel_tol=1e-12)
    assert math.isclose(max_reaction_x, 0.0, abs_tol=1e-12)

    samples = {
        -2.0: (
            2.207588099876326e-05,
            -3314.8872494088396,
            -3310.4506362596912,
            883.0352399505304,
        ),
        -1.0: (
            0.0006254824435627555,
            -3257.874339777138,
            7549.451625008884,
            25019.29774251022,
        ),
        0.0: (
            0.0013972696616225775,
            22365.04581636088,
            -50000.0,
            55890.7864649031,
        ),
        1.0: (
            0.0006254824435627555,
            -3257.874339777138,
            -7549.451625008884,
            25019.29774251022,
        ),
        2.0: (
            2.207588099876326e-05,
            -3314.8872494088396,
            3310.4506362596912,
            883.0352399505304,
        ),
    }

    for target_x, (defl, moment, shear, reaction) in samples.items():
        index = min(range(len(result.x_m)), key=lambda i: abs(result.x_m[i] - target_x))
        assert math.isclose(result.x_m[index], target_x, abs_tol=0.02)
        assert math.isclose(result.deflection_m[index], defl, rel_tol=1e-12)
        assert math.isclose(result.moment_nm[index], moment, rel_tol=1e-12)
        assert math.isclose(result.shear_n[index], shear, rel_tol=1e-12)
        assert math.isclose(result.reaction_n_per_m[index], reaction, rel_tol=1e-12)
