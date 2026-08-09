"""Severity scoring for a single detected damage.

    score = W_AREA * area_factor + W_TYPE * type_factor + W_CONF * confidence

This is a designed heuristic for ranking detections consistently, not a calibrated
model of repair urgency. See the note in `constants.py` before treating its output as
authoritative.

Pure functions only - no I/O, no model, no global state.
"""

from __future__ import annotations

import math

from src.utils.constants import (
    AREA_SATURATION,
    AREA_USES_SQRT,
    CLASS_BY_ID,
    SEVERITY_LEVELS,
    SeverityLevel,
    W_AREA,
    W_CONF,
    W_TYPE,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def area_factor(relative_area: float) -> float:
    """Map a damage's frame coverage to 0-1, saturating at ``AREA_SATURATION``.

    Area is a squared quantity, so the square root makes the result track the damage's
    linear extent instead of its pixel count. That spreads scores usefully across the
    small boxes that dominate real road imagery; linear scaling would flatten almost
    every genuine detection to near zero.
    """
    if relative_area <= 0.0:
        return 0.0

    ratio = _clamp(relative_area / AREA_SATURATION)
    return math.sqrt(ratio) if AREA_USES_SQRT else ratio


def severity_score(relative_area: float, class_id: int, confidence: float) -> float:
    """Weighted severity in 0-1.

    Raises KeyError for an unknown ``class_id`` - a model whose label indices disagree
    with our taxonomy must fail loudly rather than quietly scoring every detection as
    the mildest class.
    """
    damage = CLASS_BY_ID[class_id]

    score = (
        W_AREA * area_factor(relative_area)
        + W_TYPE * damage.type_factor
        + W_CONF * _clamp(confidence)
    )
    return _clamp(score)


def severity_level(score: float) -> SeverityLevel:
    """Bucket a score into its band. Bounds are inclusive at the top of each band."""
    value = _clamp(score)
    for level in SEVERITY_LEVELS:
        if value <= level.upper:
            return level
    return SEVERITY_LEVELS[-1]


def classify(relative_area: float, class_id: int, confidence: float) -> tuple[float, str]:
    """Convenience wrapper returning ``(score, level_name)``."""
    score = severity_score(relative_area, class_id, confidence)
    return score, severity_level(score).name
