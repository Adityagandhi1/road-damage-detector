"""SQLite persistence for detections and their damages.

Plain ``sqlite3`` - two tables and a handful of queries do not justify an ORM.

Foreign keys are enforced per-connection (SQLite defaults them OFF), so deleting a
detection cascades to its damages. Without that, deleted rows would leave orphaned
damage records that keep inflating every aggregate on the map and analytics pages.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.core.detector import Detection
from src.core.severity import severity_level

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path        TEXT NOT NULL,
    annotated_path    TEXT,
    original_filename TEXT,
    detected_at       TEXT NOT NULL DEFAULT (datetime('now')),
    source_type       TEXT NOT NULL DEFAULT 'image'
                      CHECK (source_type IN ('image', 'video', 'batch')),
    latitude          REAL,
    longitude         REAL,
    address           TEXT,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS damages (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id   INTEGER NOT NULL
                   REFERENCES detections(id) ON DELETE CASCADE,
    class_id       INTEGER NOT NULL,
    code           TEXT    NOT NULL,
    class_name     TEXT    NOT NULL,
    confidence     REAL    NOT NULL,
    severity_score REAL    NOT NULL,
    severity_level TEXT    NOT NULL,
    bbox_x1        REAL,
    bbox_y1        REAL,
    bbox_x2        REAL,
    bbox_y2        REAL,
    bbox_area      REAL,
    relative_area  REAL
);

CREATE INDEX IF NOT EXISTS idx_damages_detection ON damages(detection_id);
CREATE INDEX IF NOT EXISTS idx_detections_time   ON detections(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_detections_geo    ON detections(latitude, longitude);
"""


class DetectionStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False: Streamlit reruns scripts on other threads.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ---------------------------------------------------------------- write

    def save_detection(
        self,
        image_path: str,
        detections: Iterable[Detection],
        annotated_path: str | None = None,
        original_filename: str | None = None,
        source_type: str = "image",
        latitude: float | None = None,
        longitude: float | None = None,
        address: str | None = None,
        notes: str | None = None,
    ) -> int:
        """Insert one detection run and its damages. Returns the new detection id."""
        cur = self._conn.execute(
            """INSERT INTO detections
                 (image_path, annotated_path, original_filename, source_type,
                  latitude, longitude, address, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(image_path),
                str(annotated_path) if annotated_path else None,
                original_filename,
                source_type,
                latitude,
                longitude,
                address,
                notes,
            ),
        )
        detection_id = int(cur.lastrowid)

        rows = [
            (
                detection_id, d.class_id, d.code, d.class_name, d.confidence,
                d.severity_score, d.severity_level,
                d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
                d.bbox_area, d.relative_area,
            )
            for d in detections
        ]
        if rows:
            self._conn.executemany(
                """INSERT INTO damages
                     (detection_id, class_id, code, class_name, confidence,
                      severity_score, severity_level,
                      bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_area, relative_area)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

        self._conn.commit()
        return detection_id

    def delete_detection(self, detection_id: int) -> bool:
        """Delete a detection and (via cascade) its damages. False if it did not exist."""
        cur = self._conn.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # ---------------------------------------------------------------- read

    def get_detection(self, detection_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM detections WHERE id = ?", (detection_id,)
        ).fetchone()
        if row is None:
            return None

        record = dict(row)
        record["damages"] = [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM damages WHERE detection_id = ? ORDER BY severity_score DESC",
                (detection_id,),
            )
        ]
        return record

    def list_detections(
        self,
        limit: int | None = None,
        source_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Newest first, each row carrying its damage count and worst severity."""
        sql = """
            SELECT d.*,
                   COUNT(dm.id)                      AS damage_count,
                   COALESCE(MAX(dm.severity_score), 0.0) AS max_severity_score
            FROM detections d
            LEFT JOIN damages dm ON dm.detection_id = d.id
        """
        params: list[Any] = []
        if source_type:
            sql += " WHERE d.source_type = ?"
            params.append(source_type)

        # id DESC breaks ties: several saves can land in the same second, and
        # timestamp alone would order them arbitrarily.
        sql += " GROUP BY d.id ORDER BY d.detected_at DESC, d.id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        return [self._with_level(dict(r)) for r in self._conn.execute(sql, params)]

    def geotagged_detections(self) -> list[dict[str, Any]]:
        """Rows carrying coordinates, for the map. Rows without GPS are excluded."""
        sql = """
            SELECT d.*,
                   COUNT(dm.id)                          AS damage_count,
                   COALESCE(MAX(dm.severity_score), 0.0) AS max_severity_score
            FROM detections d
            LEFT JOIN damages dm ON dm.detection_id = d.id
            WHERE d.latitude IS NOT NULL AND d.longitude IS NOT NULL
            GROUP BY d.id
            ORDER BY max_severity_score DESC
        """
        return [self._with_level(dict(r)) for r in self._conn.execute(sql)]

    def all_damages(self) -> list[dict[str, Any]]:
        """Flat damage rows joined to their parent's location and timestamp."""
        return [
            dict(r)
            for r in self._conn.execute(
                """SELECT dm.*, d.detected_at, d.latitude, d.longitude, d.address
                   FROM damages dm
                   JOIN detections d ON d.id = dm.detection_id
                   ORDER BY dm.severity_score DESC"""
            )
        ]

    def summary_stats(self) -> dict[str, Any]:
        total_detections = self._conn.execute(
            "SELECT COUNT(*) FROM detections"
        ).fetchone()[0]

        rows = list(self._conn.execute("SELECT class_name, severity_level FROM damages"))

        return {
            "total_detections": int(total_detections),
            "total_damages": len(rows),
            "by_class": dict(Counter(r["class_name"] for r in rows)),
            "by_severity": dict(Counter(r["severity_level"] for r in rows)),
        }

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _with_level(row: dict[str, Any]) -> dict[str, Any]:
        """Attach the band matching the row's worst score, for marker colouring."""
        level = severity_level(row.get("max_severity_score") or 0.0)
        row["max_severity_level"] = level.name
        row["max_severity_hex"] = level.hex
        return row

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DetectionStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
