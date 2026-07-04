"""Analytical dynamic solver for a beam on elastic foundation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Sequence

import numpy as np

from core.dynamic.config import (
    DippedJointConfig,
    DynamicBoundaryMode,
    DynamicConfig,
    DynamicExcitationMode,
    DynamicTransitionConfig,
    DynamicTransitionProfileType,
    IrregularityMode,
)
from core.foundation.base import DampingModel
from core.dynamic.results import DippedJointResult, DynamicSpatialResult, DynamicTimeSeries
from core.dynamic.validation import require_non_negative, require_positive
from core.model import PointLoad

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpatialGrid:
    xi_m: np.ndarray
    wavenumber_rad_per_m: np.ndarray
    step_m: float


def build_spatial_grid(domain_length_m: float, spatial_step_m: float) -> SpatialGrid:
    require_positive(domain_length_m, "domain_length_m")
    require_positive(spatial_step_m, "spatial_step_m")
    count = int(math.ceil(domain_length_m / spatial_step_m))
    if count % 2 != 0:
        count += 1
    step = domain_length_m / count
    xi = (np.arange(count) - count / 2) * step
    wavenumber = 2.0 * math.pi * np.fft.fftfreq(count, d=step)
    if LOGGER.hasHandlers():
        LOGGER.info("Dynamic FFT grid: %s points, Δξ=%s m", count, step)
    return SpatialGrid(xi_m=xi, wavenumber_rad_per_m=wavenumber, step_m=step)


def _load_spectrum(loads: Sequence[PointLoad], wavenumber: np.ndarray) -> np.ndarray:
    spectrum = np.zeros_like(wavenumber, dtype=np.complex128)
    for load in loads:
        spectrum += load.load_newtons * np.exp(-1j * wavenumber * load.position_m)
    return spectrum


def _moving_oscillator_transfer(config: DynamicConfig, omega: np.ndarray) -> np.ndarray:
    mass = config.oscillator_unsprung_mass_kg
    stiffness = config.oscillator_suspension_stiffness_n_per_m
    damping = config.oscillator_suspension_damping_n_s_per_m
    if mass is None or stiffness is None:
        return np.ones_like(omega)

    numerator = np.sqrt(stiffness**2 + (damping * omega) ** 2)
    denominator = np.sqrt((stiffness - mass * omega**2) ** 2 + (damping * omega) ** 2)
    safe_denominator = np.maximum(denominator, 1.0e-12)
    transfer = numerator / safe_denominator
    return np.clip(transfer, 0.0, 5.0)


def _synthetic_irregularity_profile(
    *,
    xi_m: np.ndarray,
    psd_level_m3: float,
    seed: int,
) -> np.ndarray:
    count = xi_m.size
    if count < 2:
        return np.zeros_like(xi_m)
    step = float(xi_m[1] - xi_m[0])
    wavenumber = 2.0 * math.pi * np.fft.rfftfreq(count, d=step)
    wavenumber_safe = np.maximum(wavenumber, 2.0 * math.pi / 100.0)

    # Simple synthetic irregularity spectrum S(k) ~ C / k^2 (deterministic seed).
    spectral_density = psd_level_m3 / (wavenumber_safe**2)
    amplitude = np.sqrt(np.maximum(spectral_density, 0.0) * (wavenumber[1] if count > 2 else 1.0))
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=amplitude.shape)
    complex_spec = amplitude * np.exp(1j * phase)
    complex_spec[0] = 0.0
    profile = np.fft.irfft(complex_spec, n=count)
    return profile - float(np.mean(profile))


def _irregularity_load_spectrum(config: DynamicConfig, grid: SpatialGrid) -> np.ndarray | None:
    irregularity = config.irregularity_input
    if irregularity is None:
        return None

    if irregularity.mode == IrregularityMode.PROFILE:
        assert irregularity.profile_x_m is not None
        assert irregularity.profile_z_m is not None
        profile = np.interp(
            grid.xi_m,
            np.asarray(irregularity.profile_x_m, dtype=float),
            np.asarray(irregularity.profile_z_m, dtype=float),
            left=0.0,
            right=0.0,
        )
    else:
        assert irregularity.psd_level_m3 is not None
        profile = _synthetic_irregularity_profile(
            xi_m=grid.xi_m,
            psd_level_m3=float(irregularity.psd_level_m3),
            seed=int(irregularity.seed),
        )

    equivalent_load = config.foundation_modulus_n_per_m2 * profile
    return np.fft.fft(np.fft.ifftshift(equivalent_load))


def build_transition_stiffness_profile(
    *,
    x_values: Sequence[float],
    profile_type: DynamicTransitionProfileType,
    k1_n_per_m2: float,
    k2_n_per_m2: float | None,
    transition_length_m: float | None,
    segment_length_m: float | None,
) -> list[float]:
    if k1_n_per_m2 <= 0:
        raise ValueError("k1_n_per_m2 must be positive")
    if profile_type == DynamicTransitionProfileType.UNIFORM:
        return [k1_n_per_m2 for _ in x_values]
    if k2_n_per_m2 is None or k2_n_per_m2 <= 0:
        raise ValueError("k2_n_per_m2 must be positive for non-uniform profiles")
    if profile_type == DynamicTransitionProfileType.STEP:
        return [k1_n_per_m2 if x < 0 else k2_n_per_m2 for x in x_values]
    if profile_type == DynamicTransitionProfileType.RAMP:
        if transition_length_m is None or transition_length_m <= 0:
            raise ValueError("transition_length_m must be positive for ramp profile")
        values: list[float] = []
        for x in x_values:
            if x < 0:
                values.append(k1_n_per_m2)
            elif x <= transition_length_m:
                ratio = x / transition_length_m
                values.append(k1_n_per_m2 + (k2_n_per_m2 - k1_n_per_m2) * ratio)
            else:
                values.append(k2_n_per_m2)
        return values
    if profile_type == DynamicTransitionProfileType.EXPONENTIAL:
        if transition_length_m is None or transition_length_m <= 0:
            raise ValueError("transition_length_m must be positive for exponential profile")
        values = []
        for x in x_values:
            if x < 0:
                values.append(k1_n_per_m2)
            else:
                values.append(
                    k1_n_per_m2 + (k2_n_per_m2 - k1_n_per_m2) * (1.0 - math.exp(-x / transition_length_m))
                )
        return values
    if profile_type == DynamicTransitionProfileType.SEGMENT:
        if segment_length_m is None or segment_length_m <= 0:
            raise ValueError("segment_length_m must be positive for segment profile")
        half = segment_length_m / 2.0
        return [k2_n_per_m2 if abs(x) <= half else k1_n_per_m2 for x in x_values]
    raise ValueError(f"Unsupported profile type: {profile_type}")


def solve_transition_spatial_response(
    config: DynamicTransitionConfig,
    *,
    foundation_profile_n_per_m2: Sequence[float],
) -> DynamicSpatialResult:
    """Solve moving-load response with non-uniform support profile.

    This implementation uses a finite-difference discretization on the moving-frame
    domain and solves a sparse five-diagonal linear system in dense form.
    """
    require_positive(config.elastic_modulus_pa, "elastic_modulus_pa")
    require_positive(config.moment_inertia_m4, "moment_inertia_m4")
    require_non_negative(config.speed_m_per_s, "speed_m_per_s")

    grid = build_spatial_grid(config.domain_length_m, config.spatial_step_m)
    xi = grid.xi_m
    target_points = xi.size
    step = grid.step_m

    k_profile = np.asarray(foundation_profile_n_per_m2, dtype=float)
    if k_profile.size != xi.size:
        raise ValueError("foundation_profile_n_per_m2 length must match transition spatial grid.")

    A = np.zeros((target_points, target_points), dtype=float)
    b = np.zeros(target_points, dtype=float)

    ei = config.elastic_modulus_pa * config.moment_inertia_m4
    inertial = config.mass_kg_per_m * (config.speed_m_per_s**2)
    damping = config.foundation_damping_n_s_per_m2 * config.speed_m_per_s
    pasternak = config.pasternak_shear_n
    dx = step
    inv_dx = 1.0 / dx
    inv_dx2 = inv_dx * inv_dx
    inv_dx4 = inv_dx2 * inv_dx2

    for i in range(target_points):
        if i < 2 or i >= target_points - 2:
            A[i, i] = 1.0
            b[i] = 0.0
            continue
        k_local = float(k_profile[i])
        if config.foundation_damping_model == DampingModel.HYSTERETIC:
            k_local *= 1.0 + config.foundation_loss_factor

        c_im2 = ei * inv_dx4
        c_im1 = -4.0 * ei * inv_dx4 - inertial * inv_dx2 - damping * (0.5 * inv_dx) - pasternak * inv_dx2
        c_i = 6.0 * ei * inv_dx4 + 2.0 * inertial * inv_dx2 + k_local + 2.0 * pasternak * inv_dx2
        c_ip1 = -4.0 * ei * inv_dx4 - inertial * inv_dx2 + damping * (0.5 * inv_dx) - pasternak * inv_dx2
        c_ip2 = ei * inv_dx4

        A[i, i - 2] = c_im2
        A[i, i - 1] = c_im1
        A[i, i] = c_i
        A[i, i + 1] = c_ip1
        A[i, i + 2] = c_ip2

    for load in config.loads:
        pos = load.position_m
        if pos < float(xi[0]) or pos > float(xi[-1]):
            continue
        right = int(np.searchsorted(xi, pos))
        if right <= 0:
            b[0] += load.load_newtons / dx
            continue
        if right >= target_points:
            b[-1] += load.load_newtons / dx
            continue
        left = right - 1
        x_left = float(xi[left])
        x_right = float(xi[right])
        if math.isclose(x_right, x_left):
            b[left] += load.load_newtons / dx
            continue
        ratio = (pos - x_left) / (x_right - x_left)
        b[left] += load.load_newtons * (1.0 - ratio) / dx
        b[right] += load.load_newtons * ratio / dx

    try:
        deflection = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Dynamic transition solver matrix is singular; adjust settings.") from exc

    slope = np.gradient(deflection, xi)
    curvature = np.gradient(slope, xi)
    third_derivative = np.gradient(curvature, xi)

    moment = -ei * curvature
    shear = -ei * third_derivative
    if config.foundation_damping_model == DampingModel.HYSTERETIC:
        damping_force = k_profile * config.foundation_loss_factor * deflection
        reaction = k_profile * deflection + damping_force - pasternak * curvature
    else:
        damping_force = -config.foundation_damping_n_s_per_m2 * (config.speed_m_per_s * slope)
        reaction = k_profile * deflection + damping_force - pasternak * curvature

    return DynamicSpatialResult(
        xi_m=list(xi),
        deflection_m=list(deflection),
        moment_nm=list(moment),
        shear_n=list(shear),
        reaction_n_per_m=list(reaction),
        damping_force_n_per_m=list(damping_force),
    )


def solve_spatial_response(config: DynamicConfig) -> DynamicSpatialResult:
    require_positive(config.elastic_modulus_pa, "elastic_modulus_pa")
    require_positive(config.moment_inertia_m4, "moment_inertia_m4")
    require_positive(config.foundation_modulus_n_per_m2, "foundation_modulus_n_per_m2")
    require_non_negative(config.pasternak_shear_n, "pasternak_shear_n")

    grid = build_spatial_grid(config.domain_length_m, config.spatial_step_m)
    k = grid.wavenumber_rad_per_m

    load_spectrum = _load_spectrum(config.loads, k)
    if config.excitation_mode == DynamicExcitationMode.MOVING_OSCILLATOR:
        omega = np.abs(config.speed_m_per_s * k)
        load_spectrum = load_spectrum * _moving_oscillator_transfer(config, omega)
    irregularity_spectrum = _irregularity_load_spectrum(config, grid)
    if irregularity_spectrum is not None:
        load_spectrum = load_spectrum + irregularity_spectrum
    foundation_stiffness = config.foundation_modulus_n_per_m2
    if config.foundation_damping_model == DampingModel.HYSTERETIC:
        foundation_complex = foundation_stiffness * (1.0 + 1j * config.foundation_loss_factor)
        damping_term = 0.0
    else:
        foundation_complex = foundation_stiffness
        damping_term = 1j * config.foundation_damping_n_s_per_m2 * config.speed_m_per_s * k

    # k* = k - m ω^2 + i c ω with ω = v k (moving frame formulation).
    denom = (
        config.elastic_modulus_pa * config.moment_inertia_m4 * k**4
        - config.mass_kg_per_m * (config.speed_m_per_s**2) * k**2
        + damping_term
        + foundation_complex
        + config.pasternak_shear_n * k**2
    )

    denom_abs = np.abs(denom)
    min_abs = float(np.min(denom_abs))
    scale = float(np.median(denom_abs))
    if min_abs < 1e-10 * max(scale, 1.0):
        raise ValueError("Dynamic solver denominator is near-singular; adjust damping or domain settings.")

    spectrum = load_spectrum / denom
    scale = 1.0 / grid.step_m
    deflection = np.fft.ifft(spectrum) * scale
    slope = np.fft.ifft(1j * k * spectrum) * scale
    curvature = np.fft.ifft(-(k**2) * spectrum) * scale
    third_derivative = np.fft.ifft((1j * k) ** 3 * spectrum) * scale

    deflection = np.real(np.fft.fftshift(deflection))
    slope = np.real(np.fft.fftshift(slope))
    curvature = np.real(np.fft.fftshift(curvature))
    third_derivative = np.real(np.fft.fftshift(third_derivative))

    moment = -config.elastic_modulus_pa * config.moment_inertia_m4 * curvature
    shear = -config.elastic_modulus_pa * config.moment_inertia_m4 * third_derivative
    if config.foundation_damping_model == DampingModel.HYSTERETIC:
        damping_force = foundation_stiffness * config.foundation_loss_factor * deflection
        reaction = (
            foundation_stiffness * deflection
            + damping_force
            - config.pasternak_shear_n * curvature
        )
    else:
        damping_force = -config.foundation_damping_n_s_per_m2 * (config.speed_m_per_s * slope)
        reaction = (
            foundation_stiffness * deflection
            + damping_force
            - config.pasternak_shear_n * curvature
        )

    return DynamicSpatialResult(
        xi_m=list(grid.xi_m),
        deflection_m=list(deflection),
        moment_nm=list(moment),
        shear_n=list(shear),
        reaction_n_per_m=list(reaction),
        damping_force_n_per_m=list(damping_force),
    )


def build_time_series(
    config: DynamicConfig,
    spatial: DynamicSpatialResult,
) -> list[DynamicTimeSeries]:
    require_non_negative(config.speed_m_per_s, "speed_m_per_s")
    require_positive(config.sample_rate_hz, "sample_rate_hz")

    time_step = 1.0 / config.sample_rate_hz
    if config.time_window_s <= 0:
        raise ValueError("time_window_s must be positive")

    time = np.arange(0.0, config.time_window_s, time_step)
    if time.size == 0:
        raise ValueError("time_window_s is too small for the selected sample_rate_hz")

    xi = np.asarray(spatial.xi_m)
    deflection = np.asarray(spatial.deflection_m)
    moment = np.asarray(spatial.moment_nm)
    shear = np.asarray(spatial.shear_n)
    reaction = np.asarray(spatial.reaction_n_per_m)
    damping_force = np.asarray(spatial.damping_force_n_per_m)

    series: list[DynamicTimeSeries] = []
    period = float(xi[-1] - xi[0] + (xi[1] - xi[0])) if xi.size > 1 else 0.0
    left_bound = float(xi[0]) if xi.size else 0.0
    for position in config.probe_positions_m:
        if math.isclose(config.speed_m_per_s, 0.0):
            xi_t = np.full_like(time, position)
        else:
            xi_t = position - config.speed_m_per_s * time
        if config.boundary_mode == DynamicBoundaryMode.PERIODIC_WRAP and period > 0.0:
            xi_t = _wrap_positions(xi_t, left_bound, period)
            interp_kwargs = {}
        else:
            interp_kwargs = {"left": 0.0, "right": 0.0}

        deflection_t = np.interp(xi_t, xi, deflection, **interp_kwargs)
        moment_t = np.interp(xi_t, xi, moment, **interp_kwargs)
        shear_t = np.interp(xi_t, xi, shear, **interp_kwargs)
        reaction_t = np.interp(xi_t, xi, reaction, **interp_kwargs)
        damping_force_t = np.interp(xi_t, xi, damping_force, **interp_kwargs)

        fft_freq, fft_amp = _fft_amplitude(deflection_t, config.sample_rate_hz)
        psd_freq, psd, psd_ci_lower, psd_ci_upper = _welch_psd(
            deflection_t,
            config.sample_rate_hz,
            segment_length=config.psd_segment_length,
            overlap=config.psd_overlap,
        )
        impedance_freq, impedance_mag, impedance_phase = _support_impedance_spectrum(
            config=config, frequency_hz=fft_freq
        )

        series.append(
            DynamicTimeSeries(
                position_m=position,
                time_s=list(time),
                deflection_m=list(deflection_t),
                moment_nm=list(moment_t),
                shear_n=list(shear_t),
                reaction_n_per_m=list(reaction_t),
                damping_force_n_per_m=list(damping_force_t),
                fft_frequency_hz=list(fft_freq),
                fft_amplitude=list(fft_amp),
                psd_frequency_hz=list(psd_freq),
                psd=list(psd),
                psd_ci_lower=list(psd_ci_lower),
                psd_ci_upper=list(psd_ci_upper),
                impedance_frequency_hz=list(impedance_freq),
                impedance_magnitude_n_per_m2=list(impedance_mag),
                impedance_phase_deg=list(impedance_phase),
            )
        )

    return series


def solve_dipped_joint_forces(config: DippedJointConfig) -> DippedJointResult:
    """Compute wheel/rail forces due to a dipped joint using Jenkins/Cope equations."""
    require_positive(config.static_wheel_load_n, "static_wheel_load_n")
    require_non_negative(config.total_dip_angle_rad, "total_dip_angle_rad")
    require_non_negative(config.speed_m_per_s, "speed_m_per_s")
    require_positive(config.hertzian_stiffness_n_per_m, "hertzian_stiffness_n_per_m")
    require_positive(config.track_mass_p1_kg, "track_mass_p1_kg")
    require_positive(config.unsprung_mass_kg, "unsprung_mass_kg")
    require_positive(config.track_mass_p2_kg, "track_mass_p2_kg")
    require_positive(config.track_stiffness_p2_n_per_m, "track_stiffness_p2_n_per_m")
    require_non_negative(config.track_damping_p2_n_s_per_m, "track_damping_p2_n_s_per_m")

    load_term_p1 = math.sqrt(
        (config.hertzian_stiffness_n_per_m * config.track_mass_p1_kg)
        / (1.0 + config.track_mass_p1_kg / config.unsprung_mass_kg)
    )
    p1 = config.static_wheel_load_n + config.total_dip_angle_rad * config.speed_m_per_s * load_term_p1

    damping_factor = 1.0 - (
        config.track_damping_p2_n_s_per_m
        * math.pi
        / math.sqrt(config.track_stiffness_p2_n_per_m * config.track_mass_p2_kg)
    )
    damping_factor = _clamp_with_warning(
        damping_factor,
        min_value=0.0,
        max_value=None,
        message=(
            "Dipped-joint damping factor is below zero; cT is too high relative to kT2 "
            "and mT2. Clamping to 0."
        ),
    )
    if damping_factor > 1.0 and LOGGER.hasHandlers():
        LOGGER.warning(
            "Dipped-joint damping factor exceeds 1.0 (%.3f); verify cT, kT2, and masses.",
            damping_factor,
        )

    # Eq. 6.23 uses: P2 = P0 + 2α v · √(mu/(mu+mT2)) · (1 - cTπ/√(kT2 mT2)) · √(kT2·mu).
    mass_ratio = config.unsprung_mass_kg / (config.unsprung_mass_kg + config.track_mass_p2_kg)
    load_term_p2 = math.sqrt(mass_ratio) * damping_factor * math.sqrt(
        config.track_stiffness_p2_n_per_m * config.unsprung_mass_kg
    )
    p2 = config.static_wheel_load_n + config.total_dip_angle_rad * config.speed_m_per_s * load_term_p2

    if p1 < 0.0 or p2 < 0.0:
        raise ValueError("Dipped-joint wheel/rail forces must be non-negative.")

    if config.static_wheel_load_n > 0:
        p1_daf = p1 / config.static_wheel_load_n
        p2_daf = p2 / config.static_wheel_load_n
    else:
        p1_daf = 0.0
        p2_daf = 0.0

    return DippedJointResult(
        static_load_n=config.static_wheel_load_n,
        p1_n=p1,
        p2_n=p2,
        p1_dynamic_amplification=p1_daf,
        p2_dynamic_amplification=p2_daf,
    )


def _clamp_with_warning(
    value: float,
    *,
    min_value: float,
    max_value: float | None,
    message: str,
) -> float:
    if value < min_value:
        if LOGGER.hasHandlers():
            LOGGER.warning(message)
        return min_value
    if max_value is not None and value > max_value:
        if LOGGER.hasHandlers():
            LOGGER.warning(message)
        return max_value
    return value


def _fft_amplitude(signal: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    detrended = signal - np.mean(signal)
    spectrum = np.fft.rfft(detrended)
    count = max(len(detrended), 1)
    amplitude = np.abs(spectrum) / count
    if count > 1:
        amplitude[1:-1] *= 2.0
    frequency = np.fft.rfftfreq(len(detrended), d=1.0 / sample_rate_hz)
    return frequency, amplitude


def _wrap_positions(values: np.ndarray, start: float, period: float) -> np.ndarray:
    return (values - start) % period + start


def _support_impedance_spectrum(
    *,
    config: DynamicConfig,
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stiffness = config.foundation_modulus_n_per_m2
    if config.foundation_damping_model == DampingModel.HYSTERETIC:
        impedance = stiffness * (1.0 + 1j * config.foundation_loss_factor)
        impedance = np.full_like(frequency_hz, impedance, dtype=np.complex128)
    else:
        omega = 2.0 * math.pi * frequency_hz
        impedance = stiffness + 1j * omega * config.foundation_damping_n_s_per_m2
    magnitude = np.abs(impedance)
    phase = np.angle(impedance, deg=True)
    return frequency_hz, magnitude, phase


def _welch_psd(
    signal: np.ndarray,
    sample_rate_hz: float,
    *,
    segment_length: int,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if segment_length <= 0:
        raise ValueError("psd_segment_length must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("psd_overlap must be in [0, 1)")

    total_length = len(signal)
    if total_length == 0:
        raise ValueError("Signal must contain samples")

    segment_length = min(segment_length, total_length)
    step = max(int(segment_length * (1.0 - overlap)), 1)
    window = np.hanning(segment_length)
    scale = np.sum(window**2)

    psd_segments = []
    for start in range(0, total_length - segment_length + 1, step):
        segment = signal[start : start + segment_length]
        segment = segment - np.mean(segment)
        windowed = segment * window
        spectrum = np.fft.rfft(windowed)
        psd = (np.abs(spectrum) ** 2) / (sample_rate_hz * scale)
        psd_segments.append(psd)

    if not psd_segments:
        segment = signal - np.mean(signal)
        spectrum = np.fft.rfft(segment)
        psd_segments.append((np.abs(spectrum) ** 2) / (sample_rate_hz * total_length))

    segment_count = len(psd_segments)
    psd_average = np.mean(psd_segments, axis=0)
    frequency = np.fft.rfftfreq(segment_length, d=1.0 / sample_rate_hz)
    dof = max(2 * segment_count, 2)
    alpha = 0.05
    chi2_lower = _chi2_ppf(alpha / 2.0, dof)
    chi2_upper = _chi2_ppf(1.0 - alpha / 2.0, dof)
    psd_ci_lower = (dof * psd_average) / chi2_upper
    psd_ci_upper = (dof * psd_average) / chi2_lower
    return frequency, psd_average, psd_ci_lower, psd_ci_upper


def _chi2_ppf(probability: float, dof: int) -> float:
    probability = min(max(probability, 1.0e-6), 1.0 - 1.0e-6)
    z = _norm_ppf(probability)
    return dof * (1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))) ** 3


def _norm_ppf(probability: float) -> float:
    """Approximate inverse CDF for the standard normal distribution."""
    # Acklam's approximation
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow

    if probability < plow:
        q = math.sqrt(-2.0 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if probability > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    q = probability - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )
