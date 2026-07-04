"""Load builder utilities for axle/bogie train configurations (SI units)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.model import PointLoad


AS5100_STANDARD_ID = "AS5100.2:2017"
AS5100_LOAD_SOURCE_TYPE = "as5100_fixed_rail"
AS5100_MODEL_300LA = "300LA"
AS5100_MODEL_150LA = "150LA"
AS5100_MODEL_LABELS = (AS5100_MODEL_300LA, AS5100_MODEL_150LA)
AS5100_GROUP_SPACING_MIN_M = 12.0
AS5100_GROUP_SPACING_MAX_M = 20.0
AS5100_GROUP_INTERNAL_SPACINGS_M = (1.7, 1.1, 1.7)
AS5100_LEADING_TO_FIRST_GROUP_AXLE_M = 2.0
AXLE_TO_WHEEL_LOAD_FACTOR = 0.5

AS5100Model = Literal["300LA", "150LA"]


@dataclass(frozen=True)
class TrainLoadConfig:
    """Configuration for a repeated bogie/axle train load pattern (SI units)."""

    axle_load_n: float
    bogie_count: int
    bogie_spacing_m: float
    axles_per_bogie: int = 2
    axle_spacing_m: float = 0.0
    reference_bogie_center_m: float = 0.0


@dataclass(frozen=True)
class AS5100RailLoadConfig:
    """Configuration for a fixed AS5100 vertical rail load arrangement (SI units)."""

    model: AS5100Model
    group_count: int
    group_spacing_m: float
    reference_position_m: float = 0.0


def build_train_loads(config: TrainLoadConfig) -> list[PointLoad]:
    """Build per-rail wheel loads for a train consisting of repeated axles."""
    _require_positive(config.axle_load_n, "axle_load_n")
    _require_positive(config.bogie_count, "bogie_count")
    _require_positive(config.axles_per_bogie, "axles_per_bogie")
    if config.bogie_count > 1:
        _require_positive(config.bogie_spacing_m, "bogie_spacing_m")
    if config.axles_per_bogie > 1:
        _require_positive(config.axle_spacing_m, "axle_spacing_m")

    bogie_centers = [
        config.reference_bogie_center_m + i * config.bogie_spacing_m
        for i in range(config.bogie_count)
    ]
    axle_offsets = _axle_offsets(config.axles_per_bogie, config.axle_spacing_m)

    wheel_load_n = axle_load_to_wheel_load(config.axle_load_n)
    loads: list[PointLoad] = []
    for center in bogie_centers:
        for offset in axle_offsets:
            loads.append(
                PointLoad(position_m=center + offset, load_newtons=wheel_load_n)
            )
    return sorted(loads, key=lambda load: load.position_m)


def build_as5100_rail_loads(config: AS5100RailLoadConfig) -> list[PointLoad]:
    """Build fixed AS5100 vertical rail traffic loads as per-rail point loads."""
    model = _normalize_as5100_model(config.model)
    _require_positive(config.group_count, "group_count")
    _require_as5100_group_spacing(config.group_spacing_m)

    scale = 0.5 if model == AS5100_MODEL_150LA else 1.0
    leading_axle_load_n = 360_000.0 * scale
    group_axle_load_n = 300_000.0 * scale
    group_offsets = _as5100_group_offsets()

    loads = [
        PointLoad(
            position_m=config.reference_position_m,
            load_newtons=axle_load_to_wheel_load(leading_axle_load_n),
        )
    ]
    for group_index in range(config.group_count):
        group_start = (
            config.reference_position_m
            + AS5100_LEADING_TO_FIRST_GROUP_AXLE_M
            + group_index * config.group_spacing_m
        )
        for offset in group_offsets:
            loads.append(
                PointLoad(
                    position_m=group_start + offset,
                    load_newtons=axle_load_to_wheel_load(group_axle_load_n),
                )
            )
    return sorted(loads, key=lambda load: load.position_m)


def as5100_load_metadata(
    config: AS5100RailLoadConfig,
    *,
    loads: list[PointLoad] | None = None,
    arrangement: str = "fixed_user_selected",
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return JSON-safe traceability metadata for an AS5100 fixed load arrangement."""
    model = _normalize_as5100_model(config.model)
    resolved_loads = loads if loads is not None else build_as5100_rail_loads(config)
    axle_loads_n = [wheel_load_to_axle_load(load.load_newtons) for load in resolved_loads]
    metadata = {
        "source_type": AS5100_LOAD_SOURCE_TYPE,
        "standard": AS5100_STANDARD_ID,
        "model": model,
        "arrangement": arrangement,
        "vertical_loading_only": True,
        "load_basis": "axle_load_split_to_two_rails",
        "solver_load_basis": "wheel_load_per_rail",
        "group_count": config.group_count,
        "group_spacing_m": config.group_spacing_m,
        "group_internal_spacings_m": list(AS5100_GROUP_INTERNAL_SPACINGS_M),
        "leading_axle_to_first_group_axle_m": AS5100_LEADING_TO_FIRST_GROUP_AXLE_M,
        "reference_position_m": config.reference_position_m,
        "axle_count": len(resolved_loads),
        "max_axle_load_n": max((abs(load) for load in axle_loads_n), default=0.0),
        "max_wheel_load_n_per_rail": max(
            (abs(load.load_newtons) for load in resolved_loads),
            default=0.0,
        ),
        "axle_positions_m": [load.position_m for load in resolved_loads],
        "axle_loads_n": axle_loads_n,
        "wheel_loads_n_per_rail": [load.load_newtons for load in resolved_loads],
        "automatic_dla_applied": False,
    }
    if extra:
        metadata.update(extra)
    return metadata


def build_as5100_group_spacing_candidates(selected_spacing_m: float) -> list[float]:
    """Return a compact deterministic spacing sweep that preserves the selected spacing."""
    _require_as5100_group_spacing(selected_spacing_m)
    candidates = {
        AS5100_GROUP_SPACING_MIN_M,
        AS5100_GROUP_SPACING_MAX_M,
        float(selected_spacing_m),
    }
    return sorted(candidates)


def axle_load_to_wheel_load(axle_load_n: float) -> float:
    """Convert a total axle load to the load carried by one rail/wheel line."""
    return axle_load_n * AXLE_TO_WHEEL_LOAD_FACTOR


def wheel_load_to_axle_load(wheel_load_n: float) -> float:
    """Convert a per-rail wheel load back to the corresponding total axle load."""
    return wheel_load_n / AXLE_TO_WHEEL_LOAD_FACTOR


def _axle_offsets(axles_per_bogie: int, axle_spacing_m: float) -> list[float]:
    if axles_per_bogie == 1:
        return [0.0]
    half_span = 0.5 * (axles_per_bogie - 1) * axle_spacing_m
    return [
        -half_span + i * axle_spacing_m
        for i in range(axles_per_bogie)
    ]


def _as5100_group_offsets() -> list[float]:
    offsets = [0.0]
    current = 0.0
    for spacing in AS5100_GROUP_INTERNAL_SPACINGS_M:
        current += spacing
        offsets.append(current)
    return offsets


def _normalize_as5100_model(model: str) -> AS5100Model:
    normalized = model.strip().upper()
    if normalized not in AS5100_MODEL_LABELS:
        raise ValueError("AS5100 model must be 300LA or 150LA")
    return normalized  # type: ignore[return-value]


def _require_as5100_group_spacing(value: float) -> None:
    if value < AS5100_GROUP_SPACING_MIN_M or value > AS5100_GROUP_SPACING_MAX_M:
        raise ValueError("group_spacing_m must be between 12 m and 20 m for AS5100 rail loading")


def _require_positive(value: float | int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
