"""Smoke tests for the Streamlit pages.

`streamlit run` starts successfully even when a page raises on every request - the
traceback is rendered into the browser instead of the terminal. These tests execute the
page scripts headlessly and fail on any uncaught exception, which is the only cheap way
to catch a broken page without clicking through the UI.
"""

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest

from src.utils import config

# Model load plus a warm-up inference on CPU; the default 3s deadline is far too tight.
TIMEOUT = 120


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point the app at a scratch database so tests never touch real detections.

    The cache must be cleared on both sides: `st.cache_resource` would otherwise hand
    back a store still bound to the developer's real detections.db, making these tests
    both destructive and dependent on local state.
    """
    from shared import get_store

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    get_store.clear()
    yield
    get_store.clear()


def run_page(path: str) -> AppTest:
    at = AppTest.from_file(path, default_timeout=TIMEOUT)
    at.run()
    return at


def test_overview_page_runs_without_exceptions():
    at = run_page("app/main.py")
    assert not at.exception, [e.value for e in at.exception]


def test_overview_page_shows_its_heading():
    at = run_page("app/main.py")
    assert any("Road Damage Detection" in m.value for m in at.markdown)


def test_detection_page_runs_without_exceptions():
    at = run_page("app/pages/1_Detection.py")
    assert not at.exception, [e.value for e in at.exception]


def test_detection_page_prompts_for_input_when_idle():
    # With nothing uploaded the page must stop cleanly at the prompt rather than
    # falling through and trying to read `None` as an image.
    at = run_page("app/pages/1_Detection.py")
    assert any("Upload an image" in i.value for i in at.info)


def test_video_page_runs_without_exceptions():
    at = run_page("app/pages/2_Video.py")
    assert not at.exception, [e.value for e in at.exception]


def test_video_page_prompts_for_input_when_idle():
    at = run_page("app/pages/2_Video.py")
    assert any("Upload a dashcam" in i.value for i in at.info)


def test_map_page_runs_without_exceptions_when_empty():
    # Empty database is the first-run state; the map must explain itself rather
    # than divide by zero computing a centre point from no coordinates.
    at = run_page("app/pages/3_Map.py")
    assert not at.exception, [e.value for e in at.exception]
    assert any("No geotagged detections" in i.value for i in at.info)


def test_map_page_renders_with_seeded_data(tmp_path, monkeypatch):
    """Exercises the populated path: centring, clustering, filters and the legend."""
    import scripts.seed_demo_data as seeder
    from src.db.store import DetectionStore

    db = tmp_path / "seeded.db"
    monkeypatch.setattr(config, "DB_PATH", db)

    store = DetectionStore(db)
    created = seeder.seed(store)
    store.close()
    assert created > 0

    from shared import get_store

    get_store.clear()

    at = run_page("app/pages/3_Map.py")
    assert not at.exception, [e.value for e in at.exception]
    get_store.clear()


def test_untrained_model_warning_is_shown_when_weights_are_stock():
    """The single most important thing this app can get wrong is presenting COCO
    output as road damage. If no trained weights are present, the page must say so."""
    from shared import get_detector

    get_detector.clear()
    detector = get_detector()

    at = run_page("app/main.py")
    if not detector.is_trained:
        assert at.error, "stock weights loaded but no error banner was rendered"
        assert any("Untrained" in e.value for e in at.error)
