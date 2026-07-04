"""Validation helpers for special analysis inputs."""

from __future__ import annotations


def require_positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def require_non_negative(value: float, name: str) -> float:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value
