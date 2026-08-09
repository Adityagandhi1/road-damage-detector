"""Upload persistence tests.

Streamlit reruns the entire page script on every widget interaction, and an uploaded
file object survives those reruns. Naming each saved copy with a fresh UUID therefore
writes the whole file to disk again on every slider drag - unbounded growth in
data/uploads/, and painfully slow for a video. Content addressing fixes both.
"""

import pytest

from src.utils.files import persist_upload


def test_returns_a_path_that_exists(tmp_path):
    path = persist_upload(b"image-bytes", "road.jpg", tmp_path)
    assert path.exists()
    assert path.read_bytes() == b"image-bytes"


def test_identical_content_reuses_the_same_path(tmp_path):
    first = persist_upload(b"same", "a.jpg", tmp_path)
    second = persist_upload(b"same", "a.jpg", tmp_path)

    assert first == second
    assert len(list(tmp_path.iterdir())) == 1


def test_repeated_saves_do_not_rewrite_the_file(tmp_path):
    """The regression itself: a rerun must not touch the disk again."""
    path = persist_upload(b"payload", "clip.mp4", tmp_path)
    before = path.stat().st_mtime_ns

    # Overwrite behind the helper's back; a rewrite would restore the original bytes.
    path.write_bytes(b"sentinel")

    again = persist_upload(b"payload", "clip.mp4", tmp_path)
    assert again == path
    assert path.read_bytes() == b"sentinel", "file was rewritten on the second call"
    assert path.stat().st_mtime_ns >= before


def test_different_content_gets_a_different_path(tmp_path):
    a = persist_upload(b"one", "x.jpg", tmp_path)
    b = persist_upload(b"two", "x.jpg", tmp_path)

    assert a != b
    assert len(list(tmp_path.iterdir())) == 2


def test_same_content_under_two_names_is_stored_once(tmp_path):
    a = persist_upload(b"dup", "first.jpg", tmp_path)
    b = persist_upload(b"dup", "second.jpg", tmp_path)

    assert a == b
    assert len(list(tmp_path.iterdir())) == 1


def test_extension_is_preserved(tmp_path):
    assert persist_upload(b"v", "clip.MP4", tmp_path).suffix == ".mp4"
    assert persist_upload(b"i", "photo.jpeg", tmp_path).suffix == ".jpeg"


def test_missing_extension_falls_back_to_a_default(tmp_path):
    assert persist_upload(b"x", "noext", tmp_path).suffix == ".bin"


def test_directory_is_created_if_absent(tmp_path):
    target = tmp_path / "nested" / "uploads"
    path = persist_upload(b"x", "a.jpg", target)
    assert path.exists() and path.parent == target


def test_filename_cannot_escape_the_target_directory(tmp_path):
    # The stored name comes from a content hash, so a hostile upload name has no
    # influence on where the bytes land.
    path = persist_upload(b"x", "../../evil.jpg", tmp_path)
    assert path.parent == tmp_path
    assert ".." not in path.name


def test_a_weird_extension_does_not_become_a_path(tmp_path):
    path = persist_upload(b"x", "a.jpg/../../b", tmp_path)
    assert path.parent == tmp_path
