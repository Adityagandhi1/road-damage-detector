"""Video analysis page: detect damage through a clip and export an annotated copy."""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from shared import (
    bgr_to_rgb,
    get_detector,
    get_store,
    hero,
    model_status,
    setup_page,
    sidebar_controls,
)
from src.core.video import analyse_video, find_ffmpeg, transcode_for_browser
from src.utils import config
from src.utils.constants import SEVERITY_ORDER, SUPPORTED_VIDEO_EXTENSIONS
from src.utils.files import persist_upload

setup_page("Video", "📹")
hero(
    "📹 Video Analysis",
    "Process dashcam footage frame by frame, deduplicate damage across frames, "
    "and export an annotated clip.",
)

detector = get_detector()
store = get_store()
model_status(detector)
settings = sidebar_controls()

config.ensure_dirs()

st.sidebar.markdown("### Video settings")
frame_stride = st.sidebar.slider(
    "Process every Nth frame", 1, 10, 3,
    help="Consecutive frames overlap heavily, so skipping loses little detail and "
         "cuts runtime proportionally. On CPU this is the main speed lever.",
)

uploaded = st.file_uploader(
    "Road video",
    type=[e.lstrip(".") for e in SUPPORTED_VIDEO_EXTENSIONS],
)

if uploaded is None:
    st.info("Upload a dashcam or handheld clip to begin.")
    st.caption(
        "A 20–30 second clip is plenty for a demo. Longer footage works but processing "
        "is CPU-bound — expect roughly 3–8 frames per second at 640px."
    )
    st.stop()

# Content-addressed: a rerun fires on every slider drag, and re-copying a 50 MB clip
# into data/uploads/ each time would both stall the UI and grow the folder without end.
video_path = persist_upload(uploaded.getvalue(), uploaded.name, config.UPLOADS_DIR)

# A different clip invalidates the previous run's results and its transcoded preview.
if st.session_state.get("rdd_video", {}).get("source") not in (None, str(video_path)):
    st.session_state.pop("rdd_video", None)
    st.session_state.pop("rdd_video_playable", None)

st.caption(f"**{uploaded.name}** · {video_path.stat().st_size / 1e6:.1f} MB")

if st.button("Analyse video", type="primary", use_container_width=True):
    # Cleared before the run, not after: the preview is derived from the export we are
    # about to overwrite, so a stale entry here would play the previous analysis.
    st.session_state.pop("rdd_video_playable", None)

    progress = st.progress(0.0, text="Starting…")
    preview = st.empty()
    timeline: list[dict] = []
    started = time.perf_counter()

    def on_frame(index, total, annotated, detections):
        timeline.append({"Frame": index, "Damages": len(detections)})

        fraction = min(1.0, (index + 1) / total) if total else 0.0
        progress.progress(
            fraction,
            text=f"Frame {index + 1} of {total} · {len(detections)} damage(s) in view",
        )
        # Refreshing every frame would spend more time pushing images to the browser
        # than running inference.
        if len(timeline) % 5 == 0:
            preview.image(
                bgr_to_rgb(annotated), caption=f"Frame {index}", use_container_width=True
            )

    output_path = config.RESULTS_DIR / f"annotated_{uuid.uuid4().hex[:12]}.mp4"

    try:
        summary = analyse_video(
            detector,
            video_path,
            output_path=output_path,
            frame_stride=frame_stride,
            conf=settings["conf"],
            iou_threshold=settings["iou"],
            imgsz=settings["imgsz"],
            on_frame=on_frame,
        )
    except ValueError as exc:
        progress.empty()
        st.error(f"{exc}. The file may be corrupt or in an unsupported format.")
        st.stop()

    progress.empty()
    preview.empty()

    st.session_state["rdd_video"] = {
        "summary": summary,
        "timeline": timeline,
        "elapsed": time.perf_counter() - started,
        "source": str(video_path),
        "name": uploaded.name,
    }

# --------------------------------------------------------------------------- results

state = st.session_state.get("rdd_video")
if not state:
    st.stop()

summary = state["summary"]
elapsed = state["elapsed"]

throughput = f"{summary.frames_processed / elapsed:.1f} fps" if elapsed > 0 else "—"
st.success(
    f"Processed {summary.frames_processed} of {summary.frames_total} frames "
    f"in {elapsed:.1f}s ({throughput})"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Unique damages", summary.unique_damages)
c2.metric("Raw boxes", summary.raw_detections)
c3.metric(
    "Duplicates removed",
    summary.raw_detections - summary.unique_damages,
    help="Boxes collapsed by cross-frame tracking. Without this, one pothole seen "
         "across many frames would be reported as many potholes.",
)
c4.metric("Clip length", f"{summary.duration_s:.1f}s")

if summary.unique_damages:
    st.caption(
        f"Each damage appeared in roughly **{summary.duplication_ratio:.1f}** processed "
        f"frames on average — that multiple is exactly what deduplication removes."
    )

st.divider()

left, right = st.columns([3, 2])

with left:
    st.markdown("#### Damage timeline")
    if state["timeline"]:
        st.area_chart(
            pd.DataFrame(state["timeline"]).set_index("Frame"),
            color="#386EFF",
            height=240,
        )
        st.caption(
            f"Peak: {summary.peak_count} simultaneous damage(s) at frame "
            f"{summary.peak_frame}"
        )

with right:
    st.markdown("#### Severity mix")
    if summary.by_severity:
        st.bar_chart(
            pd.DataFrame(
                {
                    "Severity": list(SEVERITY_ORDER),
                    "Count": [summary.by_severity.get(s, 0) for s in SEVERITY_ORDER],
                }
            ).set_index("Severity"),
            color="#E67E22",
            height=240,
        )
    else:
        st.caption("No damage detected in this clip.")

# --------------------------------------------------------------------------- damages

if summary.tracks:
    st.markdown("#### Unique damages found")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "#": t.track_id + 1,
                    "Type": t.best.class_name,
                    "Severity": t.best.severity_level,
                    "Score": round(t.best.severity_score, 3),
                    "Confidence": round(t.best.confidence, 3),
                    "First frame": t.first_frame,
                    "Frames seen": t.hits,
                }
                for t in sorted(
                    summary.tracks, key=lambda t: t.best.severity_score, reverse=True
                )
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

# --------------------------------------------------------------------------- export

st.divider()
st.markdown("#### Annotated video")

out_path = summary.output_path
if out_path and out_path.exists():
    # OpenCV writes mp4v, which browsers will not play. Transcode to H.264 for inline
    # preview where ffmpeg is available, and always offer the download regardless.
    playable = st.session_state.get("rdd_video_playable")
    if playable is None:
        if find_ffmpeg():
            with st.spinner("Preparing preview…"):
                converted = transcode_for_browser(
                    out_path, out_path.with_name(f"{out_path.stem}_h264.mp4")
                )
            playable = str(converted) if converted else ""
        else:
            playable = ""
        st.session_state["rdd_video_playable"] = playable

    if playable:
        st.video(playable)
    else:
        st.info(
            "Inline preview needs **ffmpeg** on PATH to convert the export to H.264 "
            "— OpenCV cannot write browser-playable video on this platform. "
            "The download below works either way."
        )

    st.download_button(
        "Download annotated video",
        data=out_path.read_bytes(),
        file_name=f"annotated_{Path(state['name']).stem}.mp4",
        mime="video/mp4",
        use_container_width=True,
    )

# --------------------------------------------------------------------------- save

st.markdown("#### Save results")
col_a, col_b = st.columns([2, 3])
coord_text = col_a.text_input("Coordinates (lat, lng)", placeholder="11.258753, 75.780411")
notes = col_b.text_input("Notes", placeholder="e.g. NH-66 southbound, 2 km stretch")

from src.geo.geotagging import parse_coordinates  # noqa: E402

coords = parse_coordinates(coord_text) if coord_text else None
if coord_text and coords is None:
    st.warning("Coordinates not understood — expected `lat, lng` within valid ranges.")

if st.button("Save to database", type="primary"):
    # Only the best observation per track is stored. Saving every raw box would
    # inflate every aggregate the map and overview pages read.
    new_id = store.save_detection(
        image_path=state["source"],
        detections=[t.best for t in summary.tracks],
        annotated_path=str(out_path) if out_path else None,
        original_filename=state["name"],
        source_type="video",
        latitude=coords[0] if coords else None,
        longitude=coords[1] if coords else None,
        notes=notes or None,
    )
    st.success(
        f"Saved as detection #{new_id} with {summary.unique_damages} unique damage(s)."
    )
