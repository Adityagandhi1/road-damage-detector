"""Filesystem paths and model resolution.

Paths are derived from this file's location so the app works regardless of the
directory it is launched from.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Demo images live at the repo root rather than under data/, which holds runtime
# output. Samples are curated inputs committed with the project, not generated data.
SAMPLES_DIR = PROJECT_ROOT / "samples"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
DB_PATH = Path(os.getenv("RDD_DB_PATH", DATA_DIR / "detections.db"))

# Preference order for automatic model selection. ONNX first: it is materially faster
# than PyTorch on CPU, which is the only backend available on the target machine.
MODEL_CANDIDATES = (
    "road_damage_best.onnx",
    "road_damage_best.pt",
)

# Stock COCO weights. Detects people and cars, not road damage - useful only as a
# stand-in so the app is runnable before training finishes.
FALLBACK_MODEL = "yolo26n.pt"


def ensure_dirs() -> None:
    """Create the runtime directories the app writes to."""
    for d in (MODELS_DIR, DATA_DIR, SAMPLES_DIR, UPLOADS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def resolve_model_path() -> tuple[Path | str, bool]:
    """Pick the best available model.

    Returns (path_or_name, is_trained). When ``is_trained`` is False the caller is
    holding stock COCO weights and must surface that in the UI - otherwise the app
    silently reports cars as road damage, which is worse than reporting nothing.
    """
    for name in MODEL_CANDIDATES:
        candidate = MODELS_DIR / name
        if candidate.exists():
            return candidate, True

    # Prefer an already-downloaded copy over the bare name, which would make
    # ultralytics re-fetch the weights into the current working directory.
    local_fallback = MODELS_DIR / FALLBACK_MODEL
    if local_fallback.exists():
        return local_fallback, False

    return FALLBACK_MODEL, False
