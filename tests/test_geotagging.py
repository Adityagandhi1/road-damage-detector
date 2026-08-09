"""GPS parsing tests.

A hemisphere sign error puts a marker on the wrong continent while everything still
looks like it worked, so the conversion is pinned exactly.
"""

import cv2
import numpy as np
import pytest

from src.geo.geotagging import (
    dms_to_decimal,
    extract_gps,
    is_valid_coordinate,
    parse_coordinates,
)


# --------------------------------------------------------------------------------------
# dms_to_decimal
# --------------------------------------------------------------------------------------


def test_converts_degrees_minutes_seconds_to_decimal():
    # 11 deg 15' 31.68" N = 11.2588
    assert dms_to_decimal((11, 15, 31.68), "N") == pytest.approx(11.2588, abs=1e-4)


def test_south_is_negative():
    assert dms_to_decimal((33, 51, 54.0), "S") == pytest.approx(-33.865, abs=1e-4)


def test_west_is_negative():
    assert dms_to_decimal((118, 14, 37.0), "W") == pytest.approx(-118.2436, abs=1e-4)


def test_north_and_east_stay_positive():
    assert dms_to_decimal((10, 0, 0), "N") > 0
    assert dms_to_decimal((10, 0, 0), "E") > 0


def test_lowercase_hemisphere_reference_is_handled():
    assert dms_to_decimal((10, 0, 0), "s") == pytest.approx(-10.0)


# --------------------------------------------------------------------------------------
# is_valid_coordinate
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lat,lng,ok",
    [
        (11.2588, 75.7804, True),
        (0.0, 0.0, True),
        (90.0, 180.0, True),
        (-90.0, -180.0, True),
        (91.0, 0.0, False),      # latitude out of range
        (0.0, 181.0, False),     # longitude out of range
        (None, 75.0, False),
    ],
)
def test_coordinate_range_validation(lat, lng, ok):
    assert is_valid_coordinate(lat, lng) is ok


# --------------------------------------------------------------------------------------
# parse_coordinates
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("11.2588, 75.7804", (11.2588, 75.7804)),
        ("11.2588,75.7804", (11.2588, 75.7804)),
        ("  11.2588 , 75.7804  ", (11.2588, 75.7804)),
        ("-33.865, 151.209", (-33.865, 151.209)),
    ],
)
def test_parses_a_comma_separated_pair(text, expected):
    assert parse_coordinates(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "not coords", "11.2588", "abc, def", "999, 999"])
def test_unparseable_or_out_of_range_input_returns_none(text):
    assert parse_coordinates(text) is None


# --------------------------------------------------------------------------------------
# extract_gps
# --------------------------------------------------------------------------------------


def test_image_without_exif_returns_none(tmp_path):
    path = tmp_path / "plain.jpg"
    cv2.imwrite(str(path), np.zeros((40, 40, 3), dtype=np.uint8))
    assert extract_gps(path) is None


def test_missing_file_returns_none_rather_than_raising(tmp_path):
    # A corrupt or absent upload must not take the whole page down.
    assert extract_gps(tmp_path / "nope.jpg") is None


def test_survives_the_pil_patch_ultralytics_installs(tmp_path):
    """Regression: importing ultralytics replaces PIL.Image.open with a wrapper that,
    on any open failure, tries to pip-install `pi-heif` for HEIC support. When that
    install fails - as it does wherever site-packages is not writable - the wrapper
    raises ModuleNotFoundError instead of FileNotFoundError, after stalling on two
    install attempts. The app always has ultralytics loaded, so extract_gps must be
    immune to this.
    """
    import ultralytics  # noqa: F401  - imported for its global PIL side effect

    assert extract_gps(tmp_path / "still-missing.jpg") is None

    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"this is not a jpeg")
    assert extract_gps(corrupt) is None
