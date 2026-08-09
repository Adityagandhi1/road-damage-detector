"""Geospatial view of saved detections."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Fullscreen, HeatMap, MarkerCluster
from streamlit_folium import st_folium

from shared import get_store, hero, setup_page
from src.utils.constants import SEVERITY_BY_NAME, SEVERITY_ORDER

setup_page("Damage Map", "🗺️")
hero(
    "🗺️ Damage Map",
    "Geotagged detections clustered by location and coloured by worst severity.",
)

store = get_store()
rows = store.geotagged_detections()

if not rows:
    st.info(
        "**No geotagged detections yet.** Save a detection with coordinates — either "
        "from a phone photo's EXIF or typed in manually — and it will appear here."
    )
    st.caption(
        "To populate the map with demo data for screenshots, run "
        "`python scripts/seed_demo_data.py`."
    )
    st.stop()

# --------------------------------------------------------------------------- filters

st.sidebar.markdown("### Map filters")

levels = st.sidebar.multiselect(
    "Severity", list(SEVERITY_ORDER), default=list(SEVERITY_ORDER)
)
sources = sorted({r["source_type"] for r in rows})
chosen_sources = st.sidebar.multiselect("Source", sources, default=sources)
min_damages = st.sidebar.slider("Minimum damages at location", 1, 10, 1)

view = st.sidebar.radio("View", ["Markers", "Heatmap"], horizontal=True)

filtered = [
    r
    for r in rows
    if r["max_severity_level"] in levels
    and r["source_type"] in chosen_sources
    and r["damage_count"] >= min_damages
]

if not filtered:
    st.warning("No detections match the current filters.")
    st.stop()

# --------------------------------------------------------------------------- metrics

c1, c2, c3, c4 = st.columns(4)
c1.metric("Locations", len(filtered))
c2.metric("Damages", sum(r["damage_count"] for r in filtered))
c3.metric(
    "Critical sites",
    sum(1 for r in filtered if r["max_severity_level"] == "Critical"),
)
c4.metric(
    "Worst score",
    f"{max(r['max_severity_score'] for r in filtered):.2f}",
)

# --------------------------------------------------------------------------- map

lats = [r["latitude"] for r in filtered]
lngs = [r["longitude"] for r in filtered]
centre = [sum(lats) / len(lats), sum(lngs) / len(lngs)]

fmap = folium.Map(location=centre, zoom_start=13, tiles="OpenStreetMap")
Fullscreen().add_to(fmap)

if view == "Heatmap":
    # Weighted by severity so a cluster of critical damage outweighs a cluster of
    # hairline cracks, rather than the map simply showing where photos were taken.
    HeatMap(
        [[r["latitude"], r["longitude"], r["max_severity_score"]] for r in filtered],
        radius=18,
        blur=22,
        min_opacity=0.35,
    ).add_to(fmap)
else:
    cluster = MarkerCluster().add_to(fmap)
    for r in filtered:
        colour = r["max_severity_hex"]
        when = str(r["detected_at"])[:16]
        name = r["original_filename"] or Path(r["image_path"]).name

        popup = folium.Popup(
            f"""
            <div style="font-family:system-ui,sans-serif;min-width:190px">
              <div style="font-weight:700;margin-bottom:.35rem">{name}</div>
              <div style="background:{colour};color:#fff;display:inline-block;
                          padding:2px 9px;border-radius:999px;font-size:.75rem">
                {r['max_severity_level']} · {r['max_severity_score']:.2f}
              </div>
              <div style="margin-top:.5rem;font-size:.82rem;line-height:1.5">
                <b>{r['damage_count']}</b> damage(s)<br>
                Source: {r['source_type']}<br>
                {when}<br>
                <span style="opacity:.65">
                  {r['latitude']:.5f}, {r['longitude']:.5f}
                </span>
                {f"<br>{r['notes']}" if r["notes"] else ""}
              </div>
            </div>
            """,
            max_width=280,
        )

        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=6 + min(r["damage_count"], 8),   # bigger dot = more damage
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.75,
            weight=2,
            popup=popup,
            tooltip=f"{r['max_severity_level']} · {r['damage_count']} damage(s)",
        ).add_to(cluster)

st_folium(fmap, use_container_width=True, height=520, returned_objects=[])

# --------------------------------------------------------------------------- legend

cols = st.columns(len(SEVERITY_ORDER))
for col, name in zip(cols, SEVERITY_ORDER):
    spec = SEVERITY_BY_NAME[name]
    count = sum(1 for r in filtered if r["max_severity_level"] == name)
    col.markdown(
        f"""<div class="rdd-card">
              <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                           background:{spec.hex};margin-right:.45rem"></span>
              <b>{name}</b>
              <div style="opacity:.6;font-size:.8rem;margin-top:.2rem">
                {count} location(s) · {spec.action}
              </div>
            </div>""",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- table

st.divider()
st.markdown("#### Locations by severity")
st.dataframe(
    pd.DataFrame(
        [
            {
                "File": r["original_filename"] or Path(r["image_path"]).name,
                "Severity": r["max_severity_level"],
                "Score": round(r["max_severity_score"], 3),
                "Damages": r["damage_count"],
                "Source": r["source_type"],
                "Latitude": round(r["latitude"], 6),
                "Longitude": round(r["longitude"], 6),
                "When": str(r["detected_at"])[:16],
                "Notes": r["notes"] or "",
            }
            for r in filtered
        ]
    ),
    use_container_width=True,
    hide_index=True,
)
