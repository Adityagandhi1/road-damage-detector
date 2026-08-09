"""GPS extraction from image EXIF, plus manual coordinate entry.

Phone photos usually carry GPS tags; dashcam stills and downloaded images usually do
not. Every function here returns ``None`` rather than raising, because a missing or
malformed tag is the normal case and must not take a page down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ExifTags

# Reverse lookups from the human-readable tag names PIL exposes.
_GPS_IFD_TAG = next(
    (k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"), 34853
)
_GPS_TAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}


def dms_to_decimal(dms: Sequence[Any], ref: str) -> float:
    """Convert EXIF degrees/minutes/seconds plus a hemisphere ref to signed decimal.

    ``ref`` of S or W yields a negative value. Getting this wrong places the marker on
    the wrong side of the equator or prime meridian while otherwise looking correct.
    """
    degrees, minutes, seconds = (float(v) for v in dms)
    decimal = degrees + minutes / 60.0 + seconds / 3600.0

    if str(ref).strip().upper() in ("S", "W"):
        decimal = -decimal
    return decimal


def is_valid_coordinate(lat: float | None, lng: float | None) -> bool:
    """True only for a real point on Earth."""
    if lat is None or lng is None:
        return False
    try:
        lat, lng = float(lat), float(lng)
    except (TypeError, ValueError):
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0


def parse_coordinates(text: str) -> tuple[float, float] | None:
    """Parse a ``"lat, lng"`` string from manual entry. None if unusable."""
    if not text:
        return None

    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        return None

    try:
        lat, lng = float(parts[0]), float(parts[1])
    except ValueError:
        return None

    return (lat, lng) if is_valid_coordinate(lat, lng) else None


def extract_gps(image_path: str | Path) -> tuple[float, float] | None:
    """Read GPS coordinates from an image's EXIF. None when absent or unreadable."""
    path = Path(image_path)

    # Checked before opening: ultralytics replaces PIL.Image.open with a wrapper that
    # reacts to *any* open failure by trying to pip-install `pi-heif`. Where
    # site-packages is not writable that attempt fails slowly and then raises
    # ModuleNotFoundError. Not reaching the patched failure path is the cheap fix.
    if not path.is_file():
        return None

    try:
        with Image.open(str(path)) as img:
            exif = img.getexif()
            if not exif:
                return None

            gps = exif.get_ifd(_GPS_IFD_TAG)
            if not gps:
                return None

            lat_dms = gps.get(_GPS_TAGS["GPSLatitude"])
            lat_ref = gps.get(_GPS_TAGS["GPSLatitudeRef"])
            lng_dms = gps.get(_GPS_TAGS["GPSLongitude"])
            lng_ref = gps.get(_GPS_TAGS["GPSLongitudeRef"])

            if not (lat_dms and lat_ref and lng_dms and lng_ref):
                return None

            lat = dms_to_decimal(lat_dms, lat_ref)
            lng = dms_to_decimal(lng_dms, lng_ref)

    except Exception:
        # Deliberately broad. This function's contract is "coordinates or None", and
        # the failure modes are open-ended: truncated EXIF, a rational with a zero
        # denominator, an unidentifiable format, or ultralytics' patched opener
        # raising ModuleNotFoundError from its failed pi-heif install. A missing
        # geotag must never be able to take down the page that called us.
        return None

    return (lat, lng) if is_valid_coordinate(lat, lng) else None
