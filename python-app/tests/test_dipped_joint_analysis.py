import logging
import math

import pytest

from core.dynamic.config import DippedJointConfig, DynamicMode
from core.dynamic.engine import run_dynamic_analysis


def test_dipped_joint_zero_speed_and_zero_angle_match_static_load() -> None:
    config = DippedJointConfig(
        static_wheel_load_n=120_000.0,
        total_dip_angle_rad=0.02,
        speed_m_per_s=0.0,
        hertzian_stiffness_n_per_m=1.4e9,
        track_mass_p1_kg=500.0,
        unsprung_mass_kg=350.0,
        track_mass_p2_kg=800.0,
        track_stiffness_p2_n_per_m=4.0e7,
        track_damping_p2_n_s_per_m=20_000.0,
    )

    result = run_dynamic_analysis(config, mode=DynamicMode.DIPPED_JOINT)
    assert result.p1_n == config.static_wheel_load_n
    assert result.p2_n == config.static_wheel_load_n

    config = DippedJointConfig(
        static_wheel_load_n=120_000.0,
        total_dip_angle_rad=0.0,
        speed_m_per_s=30.0,
        hertzian_stiffness_n_per_m=1.4e9,
        track_mass_p1_kg=500.0,
        unsprung_mass_kg=350.0,
        track_mass_p2_kg=800.0,
        track_stiffness_p2_n_per_m=4.0e7,
        track_damping_p2_n_s_per_m=20_000.0,
    )

    result = run_dynamic_analysis(config, mode=DynamicMode.DIPPED_JOINT)
    assert result.p1_n == config.static_wheel_load_n
    assert result.p2_n == config.static_wheel_load_n


def test_dipped_joint_negative_damping_clamps_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    config = DippedJointConfig(
        static_wheel_load_n=100_000.0,
        total_dip_angle_rad=0.02,
        speed_m_per_s=30.0,
        hertzian_stiffness_n_per_m=1.4e9,
        track_mass_p1_kg=500.0,
        unsprung_mass_kg=350.0,
        track_mass_p2_kg=800.0,
        track_stiffness_p2_n_per_m=4.0e7,
        track_damping_p2_n_s_per_m=90_000.0,
    )

    with caplog.at_level(logging.WARNING):
        result = run_dynamic_analysis(config, mode=DynamicMode.DIPPED_JOINT)

    assert any("damping factor is below zero" in record.message for record in caplog.records)
    assert result.p2_n == config.static_wheel_load_n


def test_dipped_joint_reference_values_match_equations() -> None:
    # Verified against Eq. (6.22) and (6.23) using the stated inputs and manual arithmetic.
    config = DippedJointConfig(
        static_wheel_load_n=100_000.0,
        total_dip_angle_rad=0.02,
        speed_m_per_s=30.0,
        hertzian_stiffness_n_per_m=1.4e9,
        track_mass_p1_kg=500.0,
        unsprung_mass_kg=350.0,
        track_mass_p2_kg=800.0,
        track_stiffness_p2_n_per_m=4.0e7,
        track_damping_p2_n_s_per_m=20_000.0,
    )

    result = run_dynamic_analysis(config, mode=DynamicMode.DIPPED_JOINT)

    expected_p1_n = 422_125.2953158956
    expected_p2_n = 125_408.78756733816

    assert math.isclose(result.p1_n, expected_p1_n, rel_tol=1e-6)
    assert math.isclose(result.p2_n, expected_p2_n, rel_tol=1e-6)


def test_dipped_joint_rejects_zero_static_load() -> None:
    config = DippedJointConfig(
        static_wheel_load_n=0.0,
        total_dip_angle_rad=0.02,
        speed_m_per_s=30.0,
        hertzian_stiffness_n_per_m=1.4e9,
        track_mass_p1_kg=500.0,
        unsprung_mass_kg=350.0,
        track_mass_p2_kg=800.0,
        track_stiffness_p2_n_per_m=4.0e7,
        track_damping_p2_n_s_per_m=20_000.0,
    )

    with pytest.raises(ValueError, match="static_wheel_load_n"):
        run_dynamic_analysis(config, mode=DynamicMode.DIPPED_JOINT)
