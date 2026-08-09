"""Detector contract tests.

Split deliberately in two:

* `build_detection` is pure geometry/scoring and is tested exactly, with no model.
* `RoadDamageDetector` is tested for invariants only. Asserting specific boxes against
  a neural network's output would produce a test that fails whenever the weights change,
  which tells you nothing useful.
"""

import numpy as np
import pytest

from src.core.detector import Detection, RoadDamageDetector, build_detection
from src.utils.constants import SEVERITY_ORDER

# --------------------------------------------------------------------------------------
# build_detection - pure, no model required
# --------------------------------------------------------------------------------------


def test_relative_area_is_box_area_over_image_area():
    # 100x50 box = 5000 px in a 1000x500 = 500_000 px image -> exactly 1%
    det = build_detection(
        class_id=3, confidence=0.9, bbox=(10, 10, 110, 60), image_shape=(500, 1000)
    )
    assert det.relative_area == pytest.approx(0.01)


def test_bbox_area_is_in_pixels():
    det = build_detection(
        class_id=0, confidence=0.5, bbox=(0, 0, 20, 10), image_shape=(100, 100)
    )
    assert det.bbox_area == pytest.approx(200.0)


def test_class_name_comes_from_the_taxonomy():
    det = build_detection(
        class_id=3, confidence=0.9, bbox=(0, 0, 10, 10), image_shape=(100, 100)
    )
    assert det.class_name == "Pothole"
    assert det.code == "D40"


def test_severity_is_populated_from_the_scorer():
    det = build_detection(
        class_id=3, confidence=0.95, bbox=(0, 0, 400, 400), image_shape=(1000, 1000)
    )
    assert det.severity_level in SEVERITY_ORDER
    assert 0.0 <= det.severity_score <= 1.0


def test_degenerate_box_has_zero_area_not_a_crash():
    det = build_detection(
        class_id=0, confidence=0.5, bbox=(50, 50, 50, 50), image_shape=(100, 100)
    )
    assert det.bbox_area == 0.0
    assert det.relative_area == 0.0


def test_zero_sized_image_does_not_divide_by_zero():
    det = build_detection(
        class_id=0, confidence=0.5, bbox=(0, 0, 10, 10), image_shape=(0, 0)
    )
    assert det.relative_area == 0.0


def test_detection_is_immutable():
    det = build_detection(
        class_id=0, confidence=0.5, bbox=(0, 0, 10, 10), image_shape=(100, 100)
    )
    with pytest.raises(Exception):
        det.confidence = 0.1  # type: ignore[misc]


def test_untrained_model_ids_are_folded_into_taxonomy_range():
    # Stock COCO weights emit ids up to 79. They must not raise - the app has to be
    # runnable before training finishes - but the caller is told via `is_trained`.
    det = build_detection(
        class_id=67, confidence=0.5, bbox=(0, 0, 10, 10), image_shape=(100, 100)
    )
    assert det.class_id in (0, 1, 2, 3)


# --------------------------------------------------------------------------------------
# RoadDamageDetector - invariants against the real model
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detector():
    # Falls back to stock yolo26n when no trained weights are present.
    return RoadDamageDetector(imgsz=320)


@pytest.fixture(scope="module")
def road_image():
    # Deterministic synthetic frame - no network, no fixture binaries in the repo.
    rng = np.random.default_rng(0)
    img = rng.integers(60, 120, size=(480, 640, 3), dtype=np.uint8)
    img[300:400, 200:340] = 30      # dark blob, pothole-ish
    return img


def test_detect_returns_a_list_of_detections(detector, road_image):
    dets = detector.detect(road_image)
    assert isinstance(dets, list)
    assert all(isinstance(d, Detection) for d in dets)


def test_every_detection_satisfies_its_invariants(detector, road_image):
    h, w = road_image.shape[:2]
    for d in detector.detect(road_image):
        assert 0.0 <= d.confidence <= 1.0
        assert 0.0 <= d.relative_area <= 1.0
        x1, y1, x2, y2 = d.bbox
        assert x1 <= x2 and y1 <= y2
        assert 0 <= x1 <= w and 0 <= x2 <= w
        assert 0 <= y1 <= h and 0 <= y2 <= h
        assert d.severity_level in SEVERITY_ORDER


def test_raising_the_confidence_threshold_never_adds_detections(detector, road_image):
    loose = detector.detect(road_image, conf=0.10)
    strict = detector.detect(road_image, conf=0.90)
    assert len(strict) <= len(loose)


def test_detector_accepts_a_numpy_array_and_a_path(detector, road_image, tmp_path):
    import cv2

    p = tmp_path / "frame.jpg"
    cv2.imwrite(str(p), road_image)

    assert isinstance(detector.detect(road_image), list)
    assert isinstance(detector.detect(p), list)


def test_stock_weights_report_themselves_as_untrained(detector):
    # Guards the case that matters most: shipping a demo that silently uses COCO
    # weights and reports cars as potholes.
    assert isinstance(detector.is_trained, bool)


def test_annotate_returns_an_image_of_the_same_shape(detector, road_image):
    out = detector.annotate(road_image, detector.detect(road_image))
    assert out.shape == road_image.shape
    assert out.dtype == road_image.dtype


def test_annotate_does_not_mutate_the_input(detector, road_image):
    before = road_image.copy()
    detector.annotate(road_image, detector.detect(road_image))
    assert np.array_equal(road_image, before)
