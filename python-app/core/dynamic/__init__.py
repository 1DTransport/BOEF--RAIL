"""Dynamic analysis subsystem for BOEF."""

from core.dynamic.config import (
    DippedJointConfig,
    DynamicBoundaryMode,
    DynamicConfig,
    DynamicExcitationMode,
    DynamicMode,
    DynamicTransitionConfig,
    DynamicTransitionProfileType,
    DynamicTransitionRunMode,
    IrregularityInput,
    IrregularityMode,
)
from core.dynamic.engine import run_dynamic_analysis
from core.dynamic.results import DippedJointResult, DynamicResult, DynamicTransitionResult

__all__ = [
    "DippedJointConfig",
    "DippedJointResult",
    "DynamicBoundaryMode",
    "DynamicConfig",
    "DynamicExcitationMode",
    "DynamicMode",
    "DynamicResult",
    "DynamicTransitionConfig",
    "DynamicTransitionProfileType",
    "DynamicTransitionResult",
    "DynamicTransitionRunMode",
    "IrregularityInput",
    "IrregularityMode",
    "run_dynamic_analysis",
]
