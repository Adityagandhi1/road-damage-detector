"""Shared Streamlit helpers: cached resources, sidebar controls, styling.

Streamlit reruns the whole script on every interaction, so the model and the database
connection must be cached or the app reloads a neural network on every slider drag.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes page files directly, so the project root is not on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import streamlit as st

from src.core.detector import Detection, RoadDamageDetector
from src.db.store import DetectionStore
from src.utils import config
from src.utils.constants import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU_THRESHOLD,
    SEVERITY_BY_NAME,
    SEVERITY_ORDER,
    SUPPORTED_IMAGE_SIZES,
)

PAGE_ICON = "🛣️"


# --------------------------------------------------------------------------------------
# cached resources
# --------------------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading detection model…")
def get_detector(model_path: str | None = None, imgsz: int = DEFAULT_IMAGE_SIZE):
    """One model instance per (path, size), reused across reruns and pages."""
    config.ensure_dirs()
    return RoadDamageDetector(model_path=model_path, imgsz=imgsz)


@st.cache_resource
def get_store() -> DetectionStore:
    config.ensure_dirs()
    return DetectionStore(config.DB_PATH)


# --------------------------------------------------------------------------------------
# page setup
# --------------------------------------------------------------------------------------


def setup_page(title: str, icon: str = PAGE_ICON) -> None:
    st.set_page_config(
        page_title=f"{title} · Road Damage Detection",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: radial-gradient(1200px 600px at 12% -10%,
                    #16233a 0%, #0d1117 55%); }

          .rdd-hero {
            padding: 1.6rem 1.9rem; border-radius: 16px; margin-bottom: 1.2rem;
            background: linear-gradient(120deg,
                        rgba(56,110,255,.20), rgba(150,60,220,.16) 55%,
                        rgba(0,190,180,.14));
            border: 1px solid rgba(255,255,255,.10);
          }
          .rdd-hero h1 { margin: 0 0 .3rem 0; font-size: 1.75rem; letter-spacing:-.02em; }
          .rdd-hero p  { margin: 0; opacity: .75; font-size: .95rem; }

          .rdd-card {
            background: rgba(255,255,255,.04);
            border: 1px solid rgba(255,255,255,.09);
            border-radius: 13px; padding: 1rem 1.15rem; height: 100%;
          }

          .rdd-badge {
            display:inline-block; padding:.16rem .6rem; border-radius:999px;
            font-size:.76rem; font-weight:600; letter-spacing:.02em;
          }

          section[data-testid="stSidebar"] {
            background: rgba(255,255,255,.025);
            border-right: 1px solid rgba(255,255,255,.07);
          }

          div[data-testid="stMetricValue"] { font-size: 1.65rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="rdd-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# model status
# --------------------------------------------------------------------------------------


def model_status(detector: RoadDamageDetector) -> None:
    """Show which weights are loaded.

    The untrained case gets an error, not an info box, on purpose: stock COCO weights
    detect people and cars, and the modulo fold in `build_detection` will cheerfully
    relabel a bus as a pothole. A demo that looks like it works while reporting
    nonsense is worse than one that plainly says it is not ready.
    """
    if detector.is_trained:
        st.sidebar.success(
            f"**Trained model loaded**\n\n"
            f"`{Path(str(detector.model_path)).name}` · {detector.backend}"
        )
    else:
        st.sidebar.error(
            "**Untrained weights — results are meaningless**\n\n"
            f"Running stock `{Path(str(detector.model_path)).name}` "
            f"({detector.model_class_count} COCO classes). It detects cars and people, "
            "then mislabels them as road damage.\n\n"
            "Train via `training/train_yolo26_rdd.ipynb`, then drop "
            "`road_damage_best.pt` into `models/`."
        )


# --------------------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------------------


def sidebar_controls() -> dict[str, float | int]:
    st.sidebar.markdown("### Inference settings")

    conf = st.sidebar.slider(
        "Confidence threshold", 0.05, 0.95, DEFAULT_CONF_THRESHOLD, 0.05,
        help="Lower finds more damage but raises false positives.",
    )
    iou = st.sidebar.slider(
        "IoU threshold", 0.10, 0.90, DEFAULT_IOU_THRESHOLD, 0.05,
        help="How aggressively overlapping boxes are merged.",
    )
    imgsz = st.sidebar.select_slider(
        "Input size", options=list(SUPPORTED_IMAGE_SIZES), value=DEFAULT_IMAGE_SIZE,
        help="Smaller is faster and misses small cracks. 640 is the trained resolution.",
    )
    return {"conf": conf, "iou": iou, "imgsz": int(imgsz)}


# --------------------------------------------------------------------------------------
# display helpers
# --------------------------------------------------------------------------------------


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """OpenCV works in BGR; Streamlit expects RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def severity_badge(level: str) -> str:
    spec = SEVERITY_BY_NAME.get(level)
    colour = spec.hex if spec else "#888"
    return (
        f'<span class="rdd-badge" style="background:{colour}22;color:{colour};'
        f'border:1px solid {colour}55">{level}</span>'
    )


def severity_counts(detections: list[Detection]) -> dict[str, int]:
    """Counts for every band, including zeros, so the layout does not jump around."""
    counts = {name: 0 for name in SEVERITY_ORDER}
    for d in detections:
        counts[d.severity_level] += 1
    return counts


def render_metrics(detections: list[Detection]) -> None:
    counts = severity_counts(detections)
    worst = max((d.severity_score for d in detections), default=0.0)

    cols = st.columns(6)
    cols[0].metric("Damages", len(detections))
    for col, name in zip(cols[1:5], SEVERITY_ORDER):
        col.metric(name, counts[name])
    cols[5].metric("Worst score", f"{worst:.2f}")


def detections_dataframe(detections: list[Detection]):
    import pandas as pd

    if not detections:
        return pd.DataFrame(columns=["Type", "Code", "Confidence", "Severity",
                                     "Score", "Area %", "Box"])
    return pd.DataFrame([d.as_row() for d in detections])
