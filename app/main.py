"""Road Damage Detection — app entry point.

Run with:  streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from shared import (
    get_detector,
    get_store,
    hero,
    model_status,
    setup_page,
)
from src.utils.constants import SEVERITY_BY_NAME, SEVERITY_ORDER

setup_page("Overview")

hero(
    "🛣️ Road Damage Detection",
    "YOLO26 road damage detection with severity scoring, video analysis "
    "and geospatial mapping — running locally on CPU.",
)

detector = get_detector()
store = get_store()
model_status(detector)

stats = store.summary_stats()

# --------------------------------------------------------------------------- KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Images analysed", stats["total_detections"])
c2.metric("Damages found", stats["total_damages"])
c3.metric("Critical", stats["by_severity"].get("Critical", 0))

most_common = max(stats["by_class"], key=stats["by_class"].get) if stats["by_class"] else "—"
c4.metric("Most common", most_common)

st.divider()

if stats["total_damages"] == 0:
    st.info(
        "**No detections yet.** Head to **Detection** in the sidebar, upload a road "
        "image, and the result will be saved here."
    )
else:
    left, right = st.columns(2)

    with left:
        st.markdown("#### Severity breakdown")
        # Every band is listed even at zero, so the chart keeps a stable shape
        # and the colours always mean the same thing.
        sev = pd.DataFrame(
            {
                "Severity": list(SEVERITY_ORDER),
                "Count": [stats["by_severity"].get(s, 0) for s in SEVERITY_ORDER],
            }
        ).set_index("Severity")
        st.bar_chart(sev, color="#E67E22", height=260)

    with right:
        st.markdown("#### Damage types")
        by_class = pd.DataFrame(
            {"Type": list(stats["by_class"]), "Count": list(stats["by_class"].values())}
        ).set_index("Type")
        st.bar_chart(by_class, color="#386EFF", height=260)

    st.markdown("#### Recent activity")
    recent = store.list_detections(limit=8)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "When": r["detected_at"],
                    "File": r["original_filename"] or Path(r["image_path"]).name,
                    "Source": r["source_type"],
                    "Damages": r["damage_count"],
                    "Worst": r["max_severity_level"],
                    "Location": r["address"]
                    or (
                        f"{r['latitude']:.4f}, {r['longitude']:.4f}"
                        if r["latitude"] is not None
                        else "—"
                    ),
                }
                for r in recent
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.markdown(
    """
    #### How severity is scored

    `score = 0.40 × area + 0.30 × damage type + 0.30 × confidence`

    Area uses a square-root curve saturating at 15% of the frame, so the score tracks a
    damage's *linear extent* rather than its raw pixel count — without that, almost every
    real detection would flatten to near zero.

    > This is a **designed heuristic for ranking detections consistently**, not a
    > calibrated model of repair urgency. The weights were chosen by judgement; RDD
    > carries no ground-truth severity labels to fit them against.
    """
)

cols = st.columns(4)
for col, name in zip(cols, SEVERITY_ORDER):
    spec = SEVERITY_BY_NAME[name]
    col.markdown(
        f"""<div class="rdd-card">
              <div style="color:{spec.hex};font-weight:700">{name}</div>
              <div style="opacity:.6;font-size:.8rem;margin-top:.25rem">
                ≤ {spec.upper:.2f} · {spec.action}
              </div>
            </div>""",
        unsafe_allow_html=True,
    )
