"""Single-image detection page."""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import streamlit as st

from shared import (
    bgr_to_rgb,
    detections_dataframe,
    get_detector,
    get_store,
    hero,
    model_status,
    render_metrics,
    setup_page,
)
from src.geo.geotagging import extract_gps, parse_coordinates
from src.utils import config
from src.utils.constants import SUPPORTED_IMAGE_EXTENSIONS
from src.utils.files import persist_upload

setup_page("Detection", "🔍")
hero("🔍 Image Detection", "Upload a road image to detect, classify and score damage.")

detector = get_detector()
store = get_store()
model_status(detector)

from shared import sidebar_controls  # noqa: E402

settings = sidebar_controls()

# --------------------------------------------------------------------------- input

config.ensure_dirs()
samples = sorted(
    p for p in config.SAMPLES_DIR.glob("*")
    if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
)

tab_upload, tab_sample = st.tabs(["Upload", f"Samples ({len(samples)})"])

image_path: Path | None = None
display_name: str | None = None

with tab_upload:
    uploaded = st.file_uploader(
        "Road image",
        type=[e.lstrip(".") for e in SUPPORTED_IMAGE_EXTENSIONS],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        # Written to disk rather than decoded in memory so EXIF stays readable and the
        # saved database row points at a file that still exists later. Content-addressed
        # so the rerun that follows every slider drag reuses the file instead of writing
        # another copy.
        path = persist_upload(uploaded.getvalue(), uploaded.name, config.UPLOADS_DIR)
        st.session_state["rdd_upload"] = str(path)
        st.session_state["rdd_upload_name"] = uploaded.name
        st.session_state["rdd_source"] = "upload"
    elif st.session_state.get("rdd_source") == "upload":
        # The file was cleared from the uploader; do not keep analysing a ghost.
        st.session_state.pop("rdd_upload", None)
        st.session_state.pop("rdd_source", None)

with tab_sample:
    if not samples:
        st.caption(
            "Drop a few road photos into `data/samples/` and they will appear here."
        )
    else:
        chosen = st.selectbox("Sample image", samples, format_func=lambda p: p.name)
        if st.button("Use this sample", use_container_width=True):
            st.session_state["rdd_sample"] = str(chosen)
            st.session_state["rdd_source"] = "sample"

# Whichever input the user acted on most recently wins. Without an explicit choice
# here, a sample selected earlier in the session would keep overriding new uploads,
# because the sample tab's code runs after the uploader's.
active = st.session_state.get("rdd_source")

if active == "upload" and st.session_state.get("rdd_upload"):
    image_path = Path(st.session_state["rdd_upload"])
    display_name = st.session_state.get("rdd_upload_name", image_path.name)
elif active == "sample" and st.session_state.get("rdd_sample"):
    image_path = Path(st.session_state["rdd_sample"])
    display_name = image_path.name

if image_path is None or not image_path.exists():
    st.info("Upload an image or pick a sample to begin.")
    st.stop()

image = cv2.imread(str(image_path))
if image is None:
    st.error(f"Could not read `{display_name}`. The file may be corrupt.")
    st.stop()

# --------------------------------------------------------------------------- detect

start = time.perf_counter()
detections = detector.detect(
    image, conf=settings["conf"], iou=settings["iou"], imgsz=settings["imgsz"]
)
elapsed_ms = (time.perf_counter() - start) * 1000

annotated = detector.annotate(image, detections)

h, w = image.shape[:2]
st.caption(
    f"**{display_name}** · {w}×{h} px · {elapsed_ms:.0f} ms "
    f"· {detector.backend} @ {settings['imgsz']}px"
)

render_metrics(detections)
st.divider()

left, right = st.columns(2)
with left:
    st.markdown("**Original**")
    st.image(bgr_to_rgb(image), use_container_width=True)
with right:
    st.markdown("**Detected**")
    st.image(bgr_to_rgb(annotated), use_container_width=True)

# --------------------------------------------------------------------------- results

if not detections:
    st.success("No damage detected above the current confidence threshold.")
else:
    st.markdown("#### Detections")
    st.dataframe(
        detections_dataframe(detections),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Inspect individual damages"):
        for i, det in enumerate(detections, start=1):
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            pad = 18
            crop = image[
                max(0, y1 - pad): min(h, y2 + pad),
                max(0, x1 - pad): min(w, x2 + pad),
            ]

            col_img, col_meta = st.columns([1, 3])
            if crop.size:
                col_img.image(bgr_to_rgb(crop), use_container_width=True)
            col_meta.markdown(
                f"**{i}. {det.class_name}** (`{det.code}`)  \n"
                f"Confidence **{det.confidence:.1%}** · "
                f"Severity <span style='color:{det.severity_hex};font-weight:700'>"
                f"{det.severity_level}</span> ({det.severity_score:.2f}) · "
                f"covers **{det.relative_area * 100:.2f}%** of frame",
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------- save

st.divider()
st.markdown("#### Location & save")

exif_gps = extract_gps(image_path)
if exif_gps:
    st.caption(f"GPS read from image EXIF: `{exif_gps[0]:.5f}, {exif_gps[1]:.5f}`")
else:
    st.caption("No GPS in EXIF — enter coordinates manually to place this on the map.")

col_a, col_b = st.columns([2, 3])
coord_default = f"{exif_gps[0]:.6f}, {exif_gps[1]:.6f}" if exif_gps else ""
coord_text = col_a.text_input("Coordinates (lat, lng)", value=coord_default,
                              placeholder="11.258753, 75.780411")
notes = col_b.text_input("Notes", placeholder="e.g. NH-66 near the flyover")

coords = parse_coordinates(coord_text) if coord_text else None
if coord_text and coords is None:
    st.warning("Coordinates not understood — expected `lat, lng` within valid ranges.")

_, dl_col, save_col = st.columns([2, 1, 1])

ok, buf = cv2.imencode(".png", annotated)
if ok:
    dl_col.download_button(
        "Download image",
        data=buf.tobytes(),
        file_name=f"annotated_{Path(display_name).stem}.png",
        mime="image/png",
        use_container_width=True,
    )

if save_col.button("Save to database", type="primary", use_container_width=True):
    out_path = config.RESULTS_DIR / f"annotated_{uuid.uuid4().hex[:12]}.jpg"
    cv2.imwrite(str(out_path), annotated)

    new_id = store.save_detection(
        image_path=str(image_path),
        detections=detections,
        annotated_path=str(out_path),
        original_filename=display_name,
        source_type="image",
        latitude=coords[0] if coords else None,
        longitude=coords[1] if coords else None,
        notes=notes or None,
    )
    st.success(f"Saved as detection #{new_id} with {len(detections)} damage(s).")

    if not coords:
        st.caption("Saved without coordinates, so it will not appear on the map.")
