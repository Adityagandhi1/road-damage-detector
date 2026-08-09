"""Populate the database with geotagged demo detections.

An empty map is a useless portfolio screenshot, and collecting real geotagged road
photos takes a day you may not have. This fabricates plausible detections along a road
so the map, filters and legend can be demonstrated.

Every row it writes is tagged "DEMO DATA" in its notes and uses a `demo://` image path,
so seeded rows are always distinguishable from real ones.

    python scripts/seed_demo_data.py            # add demo rows
    python scripts/seed_demo_data.py --clear    # remove them again
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.detector import build_detection
from src.db.store import DetectionStore
from src.utils import config

DEMO_MARKER = "DEMO DATA"
DEMO_PREFIX = "demo://"

# A stretch of road near Kozhikode, Kerala. Any real road works - the point is that
# the points sit on a plausible line rather than scattered at random.
ROUTE = [
    (11.24870, 75.78030), (11.25120, 75.78210), (11.25390, 75.78380),
    (11.25660, 75.78520), (11.25940, 75.78690), (11.26210, 75.78840),
    (11.26480, 75.79010), (11.26770, 75.79180), (11.27050, 75.79320),
    (11.27340, 75.79490), (11.26050, 75.77420), (11.24310, 75.79860),
]

PLACES = [
    "NH-66 northbound", "NH-66 southbound", "Near the flyover", "Bypass junction",
    "Beach road", "Market approach", "School zone", "Railway underpass",
    "Bus stand approach", "Hospital road", "Link road", "Canal bridge",
]


def make_damages(rng: random.Random) -> list:
    """A handful of damages with a realistic spread of size and confidence."""
    damages = []
    for _ in range(rng.randint(1, 5)):
        class_id = rng.choices([0, 1, 2, 3], weights=[35, 25, 22, 18])[0]

        # Most real boxes are small; a few are large. Skewed low deliberately.
        side = rng.choice([40, 60, 85, 120, 170, 240, 320])
        x, y = rng.randint(0, 640 - 20), rng.randint(0, 480 - 20)

        damages.append(
            build_detection(
                class_id=class_id,
                confidence=round(rng.uniform(0.32, 0.94), 3),
                bbox=(x, y, min(x + side, 640), min(y + side, 480)),
                image_shape=(480, 640),
            )
        )
    return damages


def seed(store: DetectionStore, seed_value: int = 7) -> int:
    rng = random.Random(seed_value)   # reproducible, so screenshots are repeatable
    created = 0

    for i, (lat, lng) in enumerate(ROUTE):
        # Jitter so markers do not sit on a suspiciously perfect line.
        lat += rng.uniform(-0.0007, 0.0007)
        lng += rng.uniform(-0.0007, 0.0007)

        store.save_detection(
            image_path=f"{DEMO_PREFIX}road_{i:02d}.jpg",
            detections=make_damages(rng),
            original_filename=f"road_{i:02d}.jpg",
            source_type=rng.choice(["image", "image", "video"]),
            latitude=round(lat, 6),
            longitude=round(lng, 6),
            notes=f"{DEMO_MARKER} — {PLACES[i % len(PLACES)]}",
        )
        created += 1

    return created


def clear(store: DetectionStore) -> int:
    removed = 0
    for row in store.list_detections():
        if str(row["image_path"]).startswith(DEMO_PREFIX):
            store.delete_detection(row["id"])
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear", action="store_true", help="remove seeded demo rows")
    args = parser.parse_args()

    config.ensure_dirs()
    store = DetectionStore(config.DB_PATH)

    try:
        if args.clear:
            print(f"Removed {clear(store)} demo detection(s) from {config.DB_PATH}")
        else:
            count = seed(store)
            stats = store.summary_stats()
            print(f"Seeded {count} geotagged demo detections into {config.DB_PATH}")
            print(f"  {stats['total_damages']} damages total")
            print(f"  by severity: {stats['by_severity']}")
            print("\nOpen the Damage Map page to see them.")
            print("Remove later with: python scripts/seed_demo_data.py --clear")
    finally:
        store.close()


if __name__ == "__main__":
    main()
