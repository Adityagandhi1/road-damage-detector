"""Damage taxonomy, severity tuning constants, and display colours.

Every tunable number in the scoring pipeline lives here rather than being scattered
through the code, so the heuristic can be inspected and adjusted in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# Damage classes
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DamageClass:
    class_id: int
    code: str           # RDD dataset code
    name: str           # human-readable label
    bgr: tuple[int, int, int]   # OpenCV draw colour
    hex: str                    # web/UI colour
    type_factor: float  # intrinsic seriousness, 0-1; feeds the severity score


# Ordering matters: class_id must match the label indices in the trained model.
# RDD YOLO exports order the classes D00, D10, D20, D40.
DAMAGE_CLASSES: tuple[DamageClass, ...] = (
    DamageClass(0, "D00", "Longitudinal Crack", (0, 200, 255), "#FFC800", 0.35),
    DamageClass(1, "D10", "Transverse Crack", (0, 140, 255), "#FF8C00", 0.45),
    DamageClass(2, "D20", "Alligator Crack", (60, 60, 220), "#DC3C3C", 0.75),
    DamageClass(3, "D40", "Pothole", (200, 40, 160), "#A028C8", 1.00),
)

CLASS_BY_ID: dict[int, DamageClass] = {c.class_id: c for c in DAMAGE_CLASSES}
CLASS_NAMES: tuple[str, ...] = tuple(c.name for c in DAMAGE_CLASSES)
NUM_CLASSES = len(DAMAGE_CLASSES)

# --------------------------------------------------------------------------------------
# Severity scoring
# --------------------------------------------------------------------------------------
#
# This is a DESIGNED HEURISTIC, not an established civil-engineering standard. It exists
# to rank detections against each other in a consistent, explainable way. The weights
# below were chosen by judgement, not fitted to repair-cost data, and no ground-truth
# severity labels exist in RDD to validate them against.
#
#     score = W_AREA * area_factor + W_TYPE * type_factor + W_CONF * confidence

W_AREA = 0.40
W_TYPE = 0.30
W_CONF = 0.30

# Fraction of the frame at which a single damage is treated as maximally severe.
# Road-damage boxes are typically 0.5-8% of the frame, so saturating at 15% keeps
# almost all real detections inside the responsive part of the curve.
AREA_SATURATION = 0.15

# Area is a squared quantity: a crack twice as long and twice as wide covers 4x the
# pixels. Taking the square root makes the factor track the damage's *linear extent*,
# which spreads scores far more usefully across the small boxes that dominate the data.
# Without it, nearly every real detection would score near zero on the area term.
AREA_USES_SQRT = True


@dataclass(frozen=True)
class SeverityLevel:
    name: str
    upper: float        # inclusive upper bound of the score band
    hex: str
    bgr: tuple[int, int, int]
    action: str


# Ordered low -> high. The final band's upper bound must be 1.0.
SEVERITY_LEVELS: tuple[SeverityLevel, ...] = (
    SeverityLevel("Low", 0.30, "#2ECC71", (113, 204, 46), "Monitor"),
    SeverityLevel("Medium", 0.55, "#F1C40F", (15, 196, 241), "Schedule repair"),
    SeverityLevel("High", 0.75, "#E67E22", (34, 126, 230), "Priority repair"),
    SeverityLevel("Critical", 1.00, "#E74C3C", (60, 76, 231), "Immediate action"),
)

SEVERITY_BY_NAME: dict[str, SeverityLevel] = {s.name: s for s in SEVERITY_LEVELS}
SEVERITY_ORDER: tuple[str, ...] = tuple(s.name for s in SEVERITY_LEVELS)

# --------------------------------------------------------------------------------------
# Inference defaults
# --------------------------------------------------------------------------------------

DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMAGE_SIZE = 640
SUPPORTED_IMAGE_SIZES = (320, 416, 640)
MAX_DETECTIONS = 100

# Two boxes in consecutive video frames overlapping by more than this are treated as
# the same physical damage. Prevents one pothole seen across 30 frames from being
# counted 30 times, which would inflate every statistic downstream.
TRACKING_IOU_THRESHOLD = 0.40

# Frames a track may go unmatched before it is considered gone from view.
TRACK_MAX_AGE = 15

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
