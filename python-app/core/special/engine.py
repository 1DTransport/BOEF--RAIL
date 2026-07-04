"""Engine dispatcher for special analysis modes."""

from __future__ import annotations

from core.special.config import FloatingSlabConfig, SpecialMode
from core.special.results import FloatingSlabResult
from core.special.solver import solve_floating_slab


def run_special_analysis(
    config: FloatingSlabConfig,
    *,
    mode: SpecialMode,
) -> FloatingSlabResult:
    if mode == SpecialMode.FLOATING_SLAB:
        return solve_floating_slab(config)
    raise ValueError(f"Unsupported special analysis mode: {mode}")
