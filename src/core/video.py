"""Video analysis: per-frame detection, cross-frame deduplication, annotated export.

The tracker is the point of this module. A pothole visible for one second of 30fps
footage yields ~30 boxes. Reporting those as 30 potholes would inflate every count,
every map marker and every severity aggregate downstream, so overlapping detections of
the same class in nearby frames are collapsed into a single tracked damage.

Matching is greedy IoU against recently-seen tracks - deliberately simple. A Kalman
filter or a full ByteTrack would handle occlusion better, but the input here is a
forward-facing dashcam where damage drifts predictably down the frame, and simple
matching is enough.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

from src.core.detector import Detection, RoadDamageDetector
from src.utils.constants import TRACK_MAX_AGE, TRACKING_IOU_THRESHOLD

Box = Sequence[float]


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    intersection = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    """One physical damage, observed across one or more frames."""

    track_id: int
    class_id: int
    best: Detection          # the most severe observation seen
    bbox: tuple[float, ...]  # most recent position, used for matching
    first_frame: int
    last_frame: int
    hits: int = 1

    def observe(self, detection: Detection, frame_idx: int) -> None:
        self.bbox = detection.bbox
        self.last_frame = frame_idx
        self.hits += 1
        if detection.severity_score > self.best.severity_score:
            self.best = detection


class DamageTracker:
    def __init__(
        self,
        iou_threshold: float = TRACKING_IOU_THRESHOLD,
        max_age: int = TRACK_MAX_AGE,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: list[Track] = []      # every track ever opened
        self._active: list[Track] = []
        self._next_id = 0

    @property
    def tracks(self) -> list[Track]:
        """All tracks ever opened, in creation order."""
        return self._tracks

    @property
    def unique_count(self) -> int:
        return len(self._tracks)

    def update(
        self, detections: Iterable[Detection], frame_idx: int
    ) -> list[tuple[Detection, int]]:
        """Match this frame's detections to existing tracks. Returns (detection, id)."""
        detections = list(detections)

        # Retire tracks not seen recently, so a damage reappearing much later is
        # correctly counted as a separate one.
        self._active = [
            t for t in self._active if frame_idx - t.last_frame <= self.max_age
        ]

        # Score every legal pairing, then take them greedily best-first. Classes are
        # never mixed: a pothole inside an alligator-cracked patch is two damages.
        candidates = [
            (iou(d.bbox, t.bbox), di, ti)
            for di, d in enumerate(detections)
            for ti, t in enumerate(self._active)
            if d.class_id == t.class_id
        ]
        candidates.sort(reverse=True)

        # Track objects are stored rather than indices into `self._active`: new tracks
        # are appended to that list inside the loop below, so an index captured here
        # would only stay valid by accident.
        matched_dets: dict[int, Track] = {}
        used_tracks: set[int] = set()

        for score, di, ti in candidates:
            if score < self.iou_threshold:
                break
            if di in matched_dets or ti in used_tracks:
                continue
            matched_dets[di] = self._active[ti]
            used_tracks.add(ti)

        results: list[tuple[Detection, int]] = []
        for di, detection in enumerate(detections):
            if di in matched_dets:
                track = matched_dets[di]
                track.observe(detection, frame_idx)
            else:
                track = Track(
                    track_id=self._next_id,
                    class_id=detection.class_id,
                    best=detection,
                    bbox=detection.bbox,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                )
                self._next_id += 1
                self._tracks.append(track)
                self._active.append(track)

            results.append((detection, track.track_id))

        return results


def find_ffmpeg() -> str | None:
    """Path to an ffmpeg binary, or None."""
    return shutil.which("ffmpeg")


def transcode_for_browser(
    source: str | Path,
    output_path: str | Path,
    timeout: int = 300,
) -> Path | None:
    """Re-encode a video to H.264/AAC so a browser can play it inline.

    OpenCV on Windows generally cannot write H.264 - the bundled FFmpeg looks for an
    OpenH264 DLL that is usually missing, and it reports success while producing a file
    no browser will play. So exports are written with mp4v (reliable, valid) and
    optionally transcoded here for preview.

    Returns None whenever ffmpeg is missing or the encode fails; the caller falls back
    to offering a download. This is a convenience, never a hard requirement.
    """
    exe = find_ffmpeg()
    if exe is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                exe, "-y",
                "-i", str(source),
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",   # required for broad browser support
                "-movflags", "+faststart",
                "-an",                   # dashcam audio is irrelevant here
                str(output_path),
            ],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None

    if result.returncode != 0 or not output_path.exists():
        return None
    return output_path if output_path.stat().st_size > 0 else None


@dataclass
class VideoSummary:
    frames_total: int
    frames_processed: int
    fps: float
    duration_s: float
    raw_detections: int          # every box in every processed frame
    unique_damages: int          # after deduplication
    tracks: list[Track] = field(default_factory=list)
    by_class: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    peak_frame: int = 0          # frame carrying the most simultaneous damages
    peak_count: int = 0
    output_path: Path | None = None

    @property
    def duplication_ratio(self) -> float:
        """Raw boxes per unique damage. Shows what deduplication removed."""
        return self.raw_detections / self.unique_damages if self.unique_damages else 0.0


def analyse_video(
    detector: RoadDamageDetector,
    source: str | Path,
    output_path: str | Path | None = None,
    frame_stride: int = 2,
    conf: float | None = None,
    iou_threshold: float | None = None,
    imgsz: int | None = None,
    on_frame: Callable[[int, int, np.ndarray, list[Detection]], None] | None = None,
) -> VideoSummary:
    """Detect damage through a video, optionally writing an annotated copy.

    ``frame_stride`` trades accuracy for speed: at 2, every second frame is processed.
    Consecutive dashcam frames overlap heavily, so skipping loses little and roughly
    halves the runtime - which matters on CPU.

    ``on_frame(index, total, annotated, detections)`` is called per processed frame so
    a UI can show progress without this module importing Streamlit.
    """
    source = Path(source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_estimate = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tracker = DamageTracker()
    writer: cv2.VideoWriter | None = None

    frames_read = 0
    frames_processed = 0
    raw_detections = 0
    peak_frame = 0
    peak_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            index = frames_read
            frames_read += 1

            if index % frame_stride != 0:
                continue

            detections = detector.detect(
                frame, conf=conf, iou=iou_threshold, imgsz=imgsz
            )
            tracker.update(detections, index)

            raw_detections += len(detections)
            frames_processed += 1
            if len(detections) > peak_count:
                peak_count, peak_frame = len(detections), index

            annotated = detector.annotate(frame, detections)

            if output_path is not None:
                if writer is None:
                    h, w = annotated.shape[:2]
                    # Written at fps/stride so the annotated copy plays back over the
                    # same wall-clock duration as the source.
                    out_fps = max(1.0, fps / frame_stride)
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        out_fps,
                        (w, h),
                    )
                writer.write(annotated)

            if on_frame is not None:
                on_frame(index, total_estimate or frames_read, annotated, detections)

    finally:
        capture.release()
        if writer is not None:
            writer.release()

    # Aggregate over unique damages, not raw boxes - that is the whole point.
    best_detections = [t.best for t in tracker.tracks]

    return VideoSummary(
        frames_total=frames_read,
        frames_processed=frames_processed,
        fps=fps,
        duration_s=frames_read / fps if fps else 0.0,
        raw_detections=raw_detections,
        unique_damages=tracker.unique_count,
        tracks=tracker.tracks,
        by_class=dict(Counter(d.class_name for d in best_detections)),
        by_severity=dict(Counter(d.severity_level for d in best_detections)),
        peak_frame=peak_frame,
        peak_count=peak_count,
        output_path=Path(output_path) if output_path else None,
    )
