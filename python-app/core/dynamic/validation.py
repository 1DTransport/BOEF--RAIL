"""Validation helpers for dynamic analysis inputs."""

from __future__ import annotations

import logging
from typing import Iterable

from core.dynamic.config import (
    DynamicBoundaryMode,
    DynamicConfig,
    DynamicExcitationMode,
    DynamicTransitionConfig,
    DynamicTransitionProfileType,
    DynamicTransitionRunMode,
    IrregularityMode,
)
from core.foundation.base import DampingModel
from core.model import beam_parameter_beta

LOGGER = logging.getLogger(__name__)


def require_positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def require_non_negative(value: float, name: str) -> float:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def require_non_empty(values: Iterable[object], name: str) -> None:
    if not list(values):
        raise ValueError(f"{name} must include at least one entry")


def require_single_entry(values: Iterable[object], name: str) -> None:
    values_list = list(values)
    if len(values_list) != 1:
        raise ValueError(f"{name} must include exactly one entry")


def require_domain_length(
    domain_length_m: float,
    *,
    foundation_modulus_n_per_m2: float,
    elastic_modulus_pa: float,
    moment_inertia_m4: float,
    multiplier: float = 4.0,
) -> None:
    if domain_length_m <= 0:
        raise ValueError("domain_length_m must be positive")
    beta = beam_parameter_beta(
        foundation_modulus_n_per_m2,
        elastic_modulus_pa,
        moment_inertia_m4,
    )
    characteristic_length = 1.0 / beta
    min_length = multiplier * characteristic_length
    if domain_length_m < min_length:
        raise ValueError(
            f"domain_length_m must be at least {min_length:.3f} m "
            f"(≈{multiplier:.1f}×1/β) to avoid wrap-around."
        )


def validate_dynamic_advanced_options(config: DynamicConfig) -> None:
    if config.excitation_mode == DynamicExcitationMode.MOVING_OSCILLATOR:
        if config.oscillator_unsprung_mass_kg is None:
            raise ValueError("oscillator_unsprung_mass_kg is required for moving_oscillator mode.")
        if config.oscillator_suspension_stiffness_n_per_m is None:
            raise ValueError(
                "oscillator_suspension_stiffness_n_per_m is required for moving_oscillator mode."
            )
        require_positive(config.oscillator_unsprung_mass_kg, "oscillator_unsprung_mass_kg")
        require_positive(
            config.oscillator_suspension_stiffness_n_per_m,
            "oscillator_suspension_stiffness_n_per_m",
        )
        require_non_negative(
            config.oscillator_suspension_damping_n_s_per_m,
            "oscillator_suspension_damping_n_s_per_m",
        )

    irregularity = config.irregularity_input
    if irregularity is None:
        return

    if irregularity.mode == IrregularityMode.PROFILE:
        if not irregularity.profile_x_m or not irregularity.profile_z_m:
            raise ValueError("Irregularity profile mode requires profile_x_m and profile_z_m.")
        if len(irregularity.profile_x_m) != len(irregularity.profile_z_m):
            raise ValueError("Irregularity profile_x_m and profile_z_m lengths must match.")
        if len(irregularity.profile_x_m) < 2:
            raise ValueError("Irregularity profile_x_m must contain at least two entries.")
        for i in range(len(irregularity.profile_x_m) - 1):
            if irregularity.profile_x_m[i + 1] <= irregularity.profile_x_m[i]:
                raise ValueError("Irregularity profile_x_m must be strictly increasing.")
    elif irregularity.mode == IrregularityMode.SYNTHETIC_PSD:
        if irregularity.psd_level_m3 is None:
            raise ValueError("Synthetic irregularity mode requires psd_level_m3.")
        require_positive(irregularity.psd_level_m3, "psd_level_m3")


def validate_time_window_coverage(config: DynamicConfig) -> None:
    if config.boundary_mode != DynamicBoundaryMode.ZERO_PAD:
        return
    if config.speed_m_per_s <= 0:
        return
    travel_distance_m = config.speed_m_per_s * config.time_window_s
    if travel_distance_m > config.domain_length_m and LOGGER.hasHandlers():
        LOGGER.warning(
            "Dynamic time window spans %.2f m which exceeds domain length %.2f m in zero-pad mode; "
            "tail-zeroing may bias FFT/PSD. Consider periodic_wrap or shorter time window.",
            travel_distance_m,
            config.domain_length_m,
        )


def validate_dynamic_transition_config(config: DynamicTransitionConfig) -> None:
    require_non_empty(config.loads, "loads")
    require_positive(config.elastic_modulus_pa, "elastic_modulus_pa")
    require_positive(config.moment_inertia_m4, "moment_inertia_m4")
    require_positive(config.section_modulus_m3, "section_modulus_m3")
    require_positive(config.mass_kg_per_m, "mass_kg_per_m")
    require_positive(config.foundation_modulus_n_per_m2, "foundation_modulus_n_per_m2")
    require_non_negative(config.foundation_damping_n_s_per_m2, "foundation_damping_n_s_per_m2")
    require_non_negative(config.foundation_loss_factor, "foundation_loss_factor")
    require_non_negative(config.speed_m_per_s, "speed_m_per_s")
    require_positive(config.domain_length_m, "domain_length_m")
    require_positive(config.spatial_step_m, "spatial_step_m")
    require_non_empty(config.probe_positions_m, "probe_positions_m")
    require_positive(config.time_window_s, "time_window_s")
    require_positive(config.sample_rate_hz, "sample_rate_hz")
    require_positive(config.k1_n_per_m2, "k1_n_per_m2")
    require_domain_length(
        config.domain_length_m,
        foundation_modulus_n_per_m2=config.k1_n_per_m2,
        elastic_modulus_pa=config.elastic_modulus_pa,
        moment_inertia_m4=config.moment_inertia_m4,
    )
    if config.profile_type != DynamicTransitionProfileType.UNIFORM:
        if config.k2_n_per_m2 is None:
            raise ValueError("k2_n_per_m2 is required for non-uniform transition profiles.")
        require_positive(config.k2_n_per_m2, "k2_n_per_m2")
    if config.profile_type in (
        DynamicTransitionProfileType.RAMP,
        DynamicTransitionProfileType.EXPONENTIAL,
    ):
        if config.transition_length_m is None:
            raise ValueError("transition_length_m is required for ramp/exponential profiles.")
        require_positive(config.transition_length_m, "transition_length_m")
    if config.profile_type == DynamicTransitionProfileType.SEGMENT:
        if config.segment_length_m is None:
            raise ValueError("segment_length_m is required for segment profile.")
        require_positive(config.segment_length_m, "segment_length_m")
    if config.run_mode == DynamicTransitionRunMode.ENVELOPE:
        if config.x_ref_start_m is None or config.x_ref_end_m is None or config.x_ref_step_m is None:
            raise ValueError("Envelope transition mode requires x_ref_start_m, x_ref_end_m, and x_ref_step_m.")
        if config.x_ref_end_m <= config.x_ref_start_m:
            raise ValueError("x_ref_end_m must be greater than x_ref_start_m for envelope mode.")
        require_positive(config.x_ref_step_m, "x_ref_step_m")
    if config.solver_fidelity not in {"screening", "full_profile"}:
        raise ValueError("solver_fidelity must be either 'screening' or 'full_profile'.")
    if (
        config.solver_fidelity == "full_profile"
        and config.boundary_mode == DynamicBoundaryMode.PERIODIC_WRAP
    ):
        raise ValueError("Periodic boundary mode is not supported with full-profile dynamic transition solving.")
    if (
        config.solver_fidelity == "full_profile"
        and config.irregularity_input is not None
    ):
        raise ValueError(
            "Irregularity excitation is not supported with full-profile dynamic transition solving."
        )
    if (
        config.solver_fidelity == "full_profile"
        and config.foundation_damping_model == DampingModel.HYSTERETIC
    ):
        raise ValueError(
            "Hysteretic damping is not supported with full-profile dynamic transition solving."
        )
    if config.excitation_mode == DynamicExcitationMode.MOVING_OSCILLATOR:
        raise ValueError("Moving oscillator excitation is not supported for dynamic transition mode in this release.")
