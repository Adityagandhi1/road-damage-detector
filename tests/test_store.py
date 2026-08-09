"""Storage round-trip tests.

The map, the history view and every aggregate number read from here, so the tests
focus on what actually breaks silently: values changing type or precision across the
round trip, orphaned child rows, and filters that quietly return everything.
"""

import pytest

from src.core.detector import build_detection
from src.db.store import DetectionStore


@pytest.fixture
def store(tmp_path):
    s = DetectionStore(tmp_path / "test.db")
    yield s
    s.close()


def make_detections(n=2):
    return [
        build_detection(
            class_id=i % 4,
            confidence=0.5 + i * 0.1,
            bbox=(10 * i, 10 * i, 60 + 10 * i, 60 + 10 * i),
            image_shape=(480, 640),
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------------------


def test_opening_an_existing_database_does_not_wipe_it(tmp_path):
    path = tmp_path / "reopen.db"
    first = DetectionStore(path)
    first.save_detection(image_path="a.jpg", detections=make_detections(1))
    first.close()

    second = DetectionStore(path)
    assert len(second.list_detections()) == 1
    second.close()


# --------------------------------------------------------------------------------------
# save / read
# --------------------------------------------------------------------------------------


def test_save_returns_a_usable_row_id(store):
    new_id = store.save_detection(image_path="road.jpg", detections=make_detections(2))
    assert isinstance(new_id, int) and new_id > 0


def test_saved_detection_round_trips(store):
    new_id = store.save_detection(
        image_path="road.jpg",
        detections=make_detections(2),
        original_filename="road.jpg",
        source_type="image",
        latitude=11.2588,
        longitude=75.7804,
        address="Kozhikode, Kerala",
        notes="near the bridge",
    )

    row = store.get_detection(new_id)
    assert row["original_filename"] == "road.jpg"
    assert row["source_type"] == "image"
    assert row["latitude"] == pytest.approx(11.2588)
    assert row["longitude"] == pytest.approx(75.7804)
    assert row["address"] == "Kozhikode, Kerala"
    assert row["notes"] == "near the bridge"
    assert len(row["damages"]) == 2


def test_damage_fields_survive_the_round_trip(store):
    original = make_detections(1)[0]
    new_id = store.save_detection(image_path="x.jpg", detections=[original])

    saved = store.get_detection(new_id)["damages"][0]
    assert saved["class_name"] == original.class_name
    assert saved["confidence"] == pytest.approx(original.confidence)
    assert saved["severity_score"] == pytest.approx(original.severity_score)
    assert saved["severity_level"] == original.severity_level
    assert saved["relative_area"] == pytest.approx(original.relative_area)
    assert saved["bbox_x1"] == pytest.approx(original.bbox[0])


def test_a_detection_with_no_damages_is_still_recorded(store):
    # A clean stretch of road is a real result and must not vanish from the history.
    new_id = store.save_detection(image_path="clean.jpg", detections=[])
    assert store.get_detection(new_id)["damages"] == []


def test_get_returns_none_for_a_missing_id(store):
    assert store.get_detection(9999) is None


# --------------------------------------------------------------------------------------
# listing
# --------------------------------------------------------------------------------------


def test_listing_is_newest_first(store):
    first = store.save_detection(image_path="1.jpg", detections=[])
    second = store.save_detection(image_path="2.jpg", detections=[])

    ids = [r["id"] for r in store.list_detections()]
    assert ids.index(second) < ids.index(first)


def test_listing_respects_the_limit(store):
    for i in range(5):
        store.save_detection(image_path=f"{i}.jpg", detections=[])
    assert len(store.list_detections(limit=3)) == 3


def test_source_type_filter_excludes_other_sources(store):
    store.save_detection(image_path="a.jpg", detections=[], source_type="image")
    store.save_detection(image_path="b.mp4", detections=[], source_type="video")

    rows = store.list_detections(source_type="video")
    assert len(rows) == 1
    assert rows[0]["source_type"] == "video"


# --------------------------------------------------------------------------------------
# geotagged - feeds the map
# --------------------------------------------------------------------------------------


def test_geotagged_skips_rows_without_coordinates(store):
    store.save_detection(image_path="no_gps.jpg", detections=make_detections(1))
    store.save_detection(
        image_path="gps.jpg",
        detections=make_detections(1),
        latitude=11.25,
        longitude=75.78,
    )

    rows = store.geotagged_detections()
    assert len(rows) == 1
    assert rows[0]["image_path"] == "gps.jpg"


def test_geotagged_reports_worst_severity_per_location(store):
    dets = [
        build_detection(0, 0.30, (0, 0, 20, 20), (480, 640)),      # mild
        build_detection(3, 0.95, (0, 0, 400, 400), (480, 640)),    # severe
    ]
    store.save_detection(
        image_path="p.jpg", detections=dets, latitude=11.0, longitude=75.0
    )

    row = store.geotagged_detections()[0]
    assert row["damage_count"] == 2
    assert row["max_severity_score"] == pytest.approx(
        max(d.severity_score for d in dets)
    )


# --------------------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------------------


def test_stats_on_an_empty_database_are_zero_not_an_error(store):
    stats = store.summary_stats()
    assert stats["total_detections"] == 0
    assert stats["total_damages"] == 0
    assert stats["by_class"] == {}


def test_stats_count_damages_by_class_and_severity(store):
    store.save_detection(image_path="a.jpg", detections=make_detections(4))

    stats = store.summary_stats()
    assert stats["total_detections"] == 1
    assert stats["total_damages"] == 4
    assert sum(stats["by_class"].values()) == 4
    assert sum(stats["by_severity"].values()) == 4


# --------------------------------------------------------------------------------------
# delete
# --------------------------------------------------------------------------------------


def test_deleting_a_detection_removes_its_damages(store):
    new_id = store.save_detection(image_path="a.jpg", detections=make_detections(3))
    store.delete_detection(new_id)

    assert store.get_detection(new_id) is None
    # The cascade is what matters - orphaned damage rows would inflate every
    # aggregate on the analytics and map pages forever.
    assert store.summary_stats()["total_damages"] == 0


def test_deleting_a_missing_row_reports_false(store):
    assert store.delete_detection(4242) is False
