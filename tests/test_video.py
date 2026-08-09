"""Tracking and video-processing tests.

Deduplication is the reason this module exists. A pothole visible for one second of
30fps footage produces ~30 boxes; without tracking, the app reports 30 potholes and
every aggregate downstream is inflated. These tests pin that behaviour.
"""

import cv2
import numpy as np
import pytest

from src.core.detector import build_detection
from src.core.video import DamageTracker, analyse_video, iou

SHAPE = (480, 640)


def det(class_id, bbox, conf=0.8):
    return build_detection(class_id=class_id, confidence=conf, bbox=bbox, image_shape=SHAPE)


# --------------------------------------------------------------------------------------
# iou
# --------------------------------------------------------------------------------------


def test_identical_boxes_overlap_completely():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_disjoint_boxes_do_not_overlap():
    assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_touching_boxes_do_not_overlap():
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_half_overlap_is_one_third():
    # intersection 50, union 150
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_zero_area_boxes_do_not_divide_by_zero():
    assert iou((5, 5, 5, 5), (5, 5, 5, 5)) == 0.0


# --------------------------------------------------------------------------------------
# DamageTracker
# --------------------------------------------------------------------------------------


def test_a_stationary_damage_across_frames_is_counted_once():
    tracker = DamageTracker()
    for frame in range(30):
        tracker.update([det(3, (100, 100, 200, 200))], frame)

    assert tracker.unique_count == 1


def test_a_drifting_damage_keeps_its_track():
    # Simulates a dashcam approaching a pothole: the box moves a little each frame.
    tracker = DamageTracker()
    for frame in range(10):
        tracker.update([det(3, (100 + frame * 4, 100, 200 + frame * 4, 200))], frame)

    assert tracker.unique_count == 1


def test_a_damage_appearing_elsewhere_is_a_new_track():
    tracker = DamageTracker()
    tracker.update([det(3, (0, 0, 50, 50))], 0)
    tracker.update([det(3, (400, 400, 450, 450))], 1)

    assert tracker.unique_count == 2


def test_overlapping_boxes_of_different_classes_stay_separate():
    # A pothole inside an alligator-cracked patch is two distinct damages.
    tracker = DamageTracker()
    tracker.update([det(3, (100, 100, 200, 200))], 0)
    tracker.update([det(2, (100, 100, 200, 200))], 1)

    assert tracker.unique_count == 2


def test_two_simultaneous_damages_are_two_tracks():
    tracker = DamageTracker()
    tracker.update([det(3, (0, 0, 50, 50)), det(3, (300, 300, 380, 380))], 0)

    assert tracker.unique_count == 2


def test_a_damage_returning_after_a_long_gap_is_a_new_track():
    tracker = DamageTracker(max_age=5)
    tracker.update([det(3, (100, 100, 200, 200))], 0)
    for frame in range(1, 12):
        tracker.update([], frame)
    tracker.update([det(3, (100, 100, 200, 200))], 12)

    assert tracker.unique_count == 2


def test_a_brief_gap_does_not_split_a_track():
    # A damage occluded by a passing car for a couple of frames is still one damage.
    tracker = DamageTracker(max_age=5)
    tracker.update([det(3, (100, 100, 200, 200))], 0)
    tracker.update([], 1)
    tracker.update([], 2)
    tracker.update([det(3, (100, 100, 200, 200))], 3)

    assert tracker.unique_count == 1


def test_a_track_keeps_its_most_severe_observation():
    # The clearest view of a damage is the one worth reporting, whichever frame it
    # arrived in. Same box, so the two observations definitely match as one track.
    tracker = DamageTracker()
    tracker.update([det(3, (100, 100, 300, 300), conf=0.40)], 0)
    tracker.update([det(3, (100, 100, 300, 300), conf=0.95)], 1)

    assert tracker.unique_count == 1
    assert tracker.tracks[0].best.confidence == pytest.approx(0.95)


def test_a_later_weaker_observation_does_not_replace_the_best():
    tracker = DamageTracker()
    tracker.update([det(3, (100, 100, 300, 300), conf=0.95)], 0)
    tracker.update([det(3, (100, 100, 300, 300), conf=0.30)], 1)

    assert tracker.tracks[0].best.confidence == pytest.approx(0.95)


def test_tracks_record_when_they_were_seen():
    tracker = DamageTracker()
    for frame in range(5):
        tracker.update([det(3, (100, 100, 200, 200))], frame)

    track = tracker.tracks[0]
    assert track.first_frame == 0
    assert track.last_frame == 4
    assert track.hits == 5


def test_tracker_returns_ids_alongside_detections():
    tracker = DamageTracker()
    pairs = tracker.update([det(3, (0, 0, 50, 50))], 0)
    assert len(pairs) == 1
    assert isinstance(pairs[0][1], int)


# --------------------------------------------------------------------------------------
# analyse_video
# --------------------------------------------------------------------------------------


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240)
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        writer.write(rng.integers(40, 160, size=(240, 320, 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture(scope="module")
def detector():
    from src.core.detector import RoadDamageDetector

    return RoadDamageDetector(imgsz=320)


def test_reports_the_number_of_frames_it_processed(detector, sample_video):
    summary = analyse_video(detector, sample_video, frame_stride=1)
    assert summary.frames_processed == 20
    assert summary.frames_total == 20


def test_stride_processes_every_nth_frame(detector, sample_video):
    summary = analyse_video(detector, sample_video, frame_stride=4)
    assert summary.frames_processed == 5
    assert summary.frames_total == 20


def test_unique_damages_never_exceeds_raw_detections(detector, sample_video):
    summary = analyse_video(detector, sample_video, frame_stride=2)
    assert summary.unique_damages <= summary.raw_detections


def test_writes_a_playable_annotated_video(detector, sample_video, tmp_path):
    out = tmp_path / "annotated.mp4"
    analyse_video(detector, sample_video, output_path=out, frame_stride=1)

    assert out.exists() and out.stat().st_size > 0

    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert ok and frame is not None
    assert frames > 0


def test_progress_callback_fires_for_each_processed_frame(detector, sample_video):
    seen = []
    analyse_video(
        detector, sample_video, frame_stride=5,
        on_frame=lambda idx, total, frame, dets: seen.append(idx),
    )
    assert len(seen) == 4
    assert seen == sorted(seen)


def test_duration_is_derived_from_frame_count_and_fps(detector, sample_video):
    summary = analyse_video(detector, sample_video, frame_stride=1)
    assert summary.fps == pytest.approx(10.0, abs=0.5)
    assert summary.duration_s == pytest.approx(2.0, abs=0.3)


def test_transcode_returns_none_when_ffmpeg_is_absent(sample_video, tmp_path, monkeypatch):
    # The feature is optional. Without ffmpeg the app must fall back to a download
    # link, not crash the page.
    import src.core.video as video_mod

    monkeypatch.setattr(video_mod.shutil, "which", lambda _: None)
    assert video_mod.transcode_for_browser(sample_video, tmp_path / "out.mp4") is None


def test_transcode_produces_a_playable_file_when_ffmpeg_exists(sample_video, tmp_path):
    from src.core.video import find_ffmpeg, transcode_for_browser

    if find_ffmpeg() is None:
        pytest.skip("ffmpeg not installed")

    out = transcode_for_browser(sample_video, tmp_path / "browser.mp4")
    assert out is not None and out.exists() and out.stat().st_size > 0

    cap = cv2.VideoCapture(str(out))
    ok, _ = cap.read()
    cap.release()
    assert ok


def test_transcoding_a_corrupt_file_returns_none(tmp_path):
    from src.core.video import find_ffmpeg, transcode_for_browser

    if find_ffmpeg() is None:
        pytest.skip("ffmpeg not installed")

    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"nonsense")
    assert transcode_for_browser(bad, tmp_path / "out.mp4") is None


def test_an_unreadable_video_raises_a_clear_error(detector, tmp_path):
    bad = tmp_path / "not-a-video.mp4"
    bad.write_bytes(b"nonsense")

    with pytest.raises(ValueError, match="[Cc]ould not open"):
        analyse_video(detector, bad)
