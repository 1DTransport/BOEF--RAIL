"""Dynamic analysis outputs (SI units)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Extremum:
    value: float
    position_m: float


@dataclass(frozen=True)
class DynamicSummary:
    max_deflection: Extremum
    max_moment: Extremum
    max_shear: Extremum
    max_reaction: Extremum


@dataclass(frozen=True)
class DynamicParameterTrace:
    """Derived parameters needed to independently check a dynamic run."""

    flexural_rigidity_nm2: float
    foundation_modulus_n_per_m2: float
    foundation_damping_n_s_per_m2: float
    damping_ratio: float
    mass_kg_per_m: float
    beta_per_m: float
    characteristic_length_m: float
    spatial_step_m: float
    critical_speed_m_per_s: float
    critical_speed_ratio: float
    dynamic_amplification: float


@dataclass(frozen=True)
class DynamicSpatialResult:
    xi_m: list[float]
    deflection_m: list[float]
    moment_nm: list[float]
    shear_n: list[float]
    reaction_n_per_m: list[float]
    damping_force_n_per_m: list[float]


@dataclass(frozen=True)
class DynamicTimeSeries:
    position_m: float
    time_s: list[float]
    deflection_m: list[float]
    moment_nm: list[float]
    shear_n: list[float]
    reaction_n_per_m: list[float]
    damping_force_n_per_m: list[float]
    fft_frequency_hz: list[float]
    fft_amplitude: list[float]
    psd_frequency_hz: list[float]
    psd: list[float]
    psd_ci_lower: list[float]
    psd_ci_upper: list[float]
    impedance_frequency_hz: list[float]
    impedance_magnitude_n_per_m2: list[float]
    impedance_phase_deg: list[float]


@dataclass(frozen=True)
class WavelengthBandMetric:
    band_name: str
    min_wavelength_m: float
    max_wavelength_m: float
    rms_amplitude_m: float


@dataclass(frozen=True)
class TransitionRiskPoint:
    speed_ratio: float
    stiffness_ratio: float
    risk_index: float


@dataclass(frozen=True)
class TransitionRiskMetrics:
    critical_speed_m_per_s: float
    critical_speed_ratio: float
    dynamic_amplification: float
    transition_stiffness_ratio: float | None
    risk_index: float
    risk_map: list[TransitionRiskPoint]


@dataclass(frozen=True)
class DynamicResult:
    spatial: DynamicSpatialResult
    probes: Sequence[DynamicTimeSeries]
    summary: DynamicSummary
    wavelength_band_metrics: Sequence[WavelengthBandMetric] | None = None
    transition_risk_metrics: TransitionRiskMetrics | None = None
    parameter_trace: DynamicParameterTrace | None = None


@dataclass(frozen=True)
class DynamicTransitionMetrics:
    max_deflection_m: float
    max_moment_nm: float
    max_shear_n: float
    max_reaction_n_per_m: float
    governing_x_ref_m: float
    risk_index: float
    critical_speed_ratio: float
    dynamic_amplification: float
    transition_stiffness_ratio: float | None


@dataclass(frozen=True)
class DynamicTransitionSeries:
    x_m: list[float]
    k_profile_n_per_m2: list[float]
    deflection_m: list[float] | None = None
    moment_nm: list[float] | None = None
    shear_n: list[float] | None = None
    reaction_n_per_m: list[float] | None = None
    deflection_max_m: list[float] | None = None
    deflection_min_m: list[float] | None = None
    moment_max_nm: list[float] | None = None
    moment_min_nm: list[float] | None = None
    shear_max_n: list[float] | None = None
    shear_min_n: list[float] | None = None
    reaction_max_n_per_m: list[float] | None = None
    reaction_min_n_per_m: list[float] | None = None


@dataclass(frozen=True)
class DynamicTransitionResult:
    solver_fidelity: str
    profile_type: str
    run_mode: str
    k1_n_per_m2: float
    k2_n_per_m2: float | None
    transition_length_m: float | None
    segment_length_m: float | None
    x_ref_m: float | None
    x_ref_start_m: float | None
    x_ref_end_m: float | None
    x_ref_step_m: float | None
    metrics: DynamicTransitionMetrics
    series: DynamicTransitionSeries
    representative: DynamicResult
    envelope_count: int = 1


@dataclass(frozen=True)
class DippedJointResult:
    static_load_n: float
    p1_n: float
    p2_n: float
    p1_dynamic_amplification: float
    p2_dynamic_amplification: float
