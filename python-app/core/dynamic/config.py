"""Dynamic analysis configuration inputs (SI units)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from core.foundation.base import DampingModel
from core.model import PointLoad


class DynamicMode(str, Enum):
    STEADY_STATE = "steady_state"
    TIME_HISTORY = "time_history"
    DIPPED_JOINT = "dipped_joint"
    TRANSITION = "transition"


class DynamicExcitationMode(str, Enum):
    MOVING_LOAD = "moving_load"
    MOVING_OSCILLATOR = "moving_oscillator"


class DynamicBoundaryMode(str, Enum):
    ZERO_PAD = "zero_pad"
    PERIODIC_WRAP = "periodic_wrap"


class IrregularityMode(str, Enum):
    PROFILE = "profile"
    SYNTHETIC_PSD = "synthetic_psd"


class DynamicTransitionProfileType(str, Enum):
    UNIFORM = "uniform"
    STEP = "step"
    RAMP = "ramp"
    EXPONENTIAL = "exponential"
    SEGMENT = "segment"


class DynamicTransitionRunMode(str, Enum):
    SINGLE = "single"
    ENVELOPE = "envelope"


@dataclass(frozen=True)
class IrregularityInput:
    """Optional track irregularity excitation input (SI units)."""

    mode: IrregularityMode
    profile_x_m: Sequence[float] | None = None
    profile_z_m: Sequence[float] | None = None
    psd_level_m3: float | None = None
    seed: int = 0


@dataclass(frozen=True)
class DynamicConfig:
    """Configuration for analytical dynamic analysis (SI units)."""

    loads: Sequence[PointLoad]
    elastic_modulus_pa: float
    moment_inertia_m4: float
    section_modulus_m3: float
    mass_kg_per_m: float
    foundation_modulus_n_per_m2: float
    foundation_damping_n_s_per_m2: float
    speed_m_per_s: float
    domain_length_m: float
    spatial_step_m: float
    probe_positions_m: Sequence[float]
    time_window_s: float
    sample_rate_hz: float
    foundation_damping_model: DampingModel = DampingModel.VISCOUS
    foundation_loss_factor: float = 0.0
    pasternak_shear_n: float = 0.0
    psd_segment_length: int = 256
    psd_overlap: float = 0.5
    excitation_mode: DynamicExcitationMode = DynamicExcitationMode.MOVING_LOAD
    boundary_mode: DynamicBoundaryMode = DynamicBoundaryMode.ZERO_PAD
    oscillator_unsprung_mass_kg: float | None = None
    oscillator_suspension_stiffness_n_per_m: float | None = None
    oscillator_suspension_damping_n_s_per_m: float = 0.0
    irregularity_input: IrregularityInput | None = None
    transition_stiffness_ratio: float | None = None


@dataclass(frozen=True)
class DynamicTransitionConfig:
    """Configuration for dynamic transition analysis (SI units)."""

    loads: Sequence[PointLoad]
    elastic_modulus_pa: float
    moment_inertia_m4: float
    section_modulus_m3: float
    mass_kg_per_m: float
    foundation_modulus_n_per_m2: float
    foundation_damping_n_s_per_m2: float
    speed_m_per_s: float
    domain_length_m: float
    spatial_step_m: float
    probe_positions_m: Sequence[float]
    time_window_s: float
    sample_rate_hz: float
    foundation_damping_model: DampingModel = DampingModel.VISCOUS
    foundation_loss_factor: float = 0.0
    pasternak_shear_n: float = 0.0
    psd_segment_length: int = 256
    psd_overlap: float = 0.5
    excitation_mode: DynamicExcitationMode = DynamicExcitationMode.MOVING_LOAD
    boundary_mode: DynamicBoundaryMode = DynamicBoundaryMode.ZERO_PAD
    oscillator_unsprung_mass_kg: float | None = None
    oscillator_suspension_stiffness_n_per_m: float | None = None
    oscillator_suspension_damping_n_s_per_m: float = 0.0
    irregularity_input: IrregularityInput | None = None

    profile_type: DynamicTransitionProfileType = DynamicTransitionProfileType.UNIFORM
    run_mode: DynamicTransitionRunMode = DynamicTransitionRunMode.SINGLE
    solver_fidelity: str = "screening"
    k1_n_per_m2: float = 0.0
    k2_n_per_m2: float | None = None
    transition_length_m: float | None = None
    segment_length_m: float | None = None
    x_ref_m: float = 0.0
    x_ref_start_m: float | None = None
    x_ref_end_m: float | None = None
    x_ref_step_m: float | None = None
    transition_stiffness_ratio: float | None = None


@dataclass(frozen=True)
class DippedJointConfig:
    """Configuration for dipped joint wheel/rail force analysis (SI units, 2α input)."""

    static_wheel_load_n: float
    total_dip_angle_rad: float
    speed_m_per_s: float
    hertzian_stiffness_n_per_m: float
    track_mass_p1_kg: float
    unsprung_mass_kg: float
    track_mass_p2_kg: float
    track_stiffness_p2_n_per_m: float
    track_damping_p2_n_s_per_m: float
