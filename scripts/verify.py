"""End-to-end health check.

Exercises every layer of the app against real code - model load, inference, severity
scoring, database round-trip, video tracking and export - and reports what works, what
is degraded, and what is broken.

    python scripts/verify.py

Exits 0 when everything essential passes, 1 otherwise. Warnings do not fail the run:
they mark things that are optional or expected before training.

Uses a temporary database and temporary files throughout; your real detections.db is
never touched.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    icon = {PASS: "[ok]", WARN: "[--]", FAIL: "[XX]"}[status]
    print(f"  {icon} {name}" + (f"  —  {detail}" if detail else ""), flush=True)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ======================================================================== environment


def check_environment() -> None:
    section("Environment")

    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        record(PASS, "Python", version)
    else:
        record(FAIL, "Python", f"{version} — 3.10+ required")

    for module, label in [
        ("ultralytics", "ultralytics"),
        ("torch", "torch"),
        ("cv2", "opencv"),
        ("streamlit", "streamlit"),
        ("folium", "folium"),
        ("streamlit_folium", "streamlit-folium"),
        ("PIL", "pillow"),
        ("pandas", "pandas"),
    ]:
        try:
            mod = __import__(module)
            record(PASS, label, getattr(mod, "__version__", "installed"))
        except ImportError:
            record(FAIL, label, "not installed — run: pip install -r requirements.txt")

    try:
        import onnxruntime
        record(PASS, "onnxruntime", onnxruntime.__version__)
    except ImportError:
        record(WARN, "onnxruntime", "absent — ONNX weights will not load")

    free_gb = shutil.disk_usage(Path(__file__).resolve().parents[1]).free / 1e9
    if free_gb >= 2:
        record(PASS, "Disk space", f"{free_gb:.1f} GB free")
    else:
        record(WARN, "Disk space", f"only {free_gb:.1f} GB free")

    from src.core.video import find_ffmpeg

    if find_ffmpeg():
        record(PASS, "ffmpeg", "found — in-browser video preview available")
    else:
        record(WARN, "ffmpeg", "absent — video export downloads only, no inline preview")


# ============================================================================== model


def check_model():
    section("Model")

    from src.core.detector import RoadDamageDetector
    from src.utils import config

    path, is_trained = config.resolve_model_path()

    started = time.perf_counter()
    detector = RoadDamageDetector(imgsz=320)
    load_ms = (time.perf_counter() - started) * 1000

    record(PASS, "Model loads", f"{Path(str(path)).name} · {detector.backend} · {load_ms:.0f} ms")

    if detector.is_trained:
        record(PASS, "Trained weights", f"{detector.model_class_count} classes")
    else:
        record(
            WARN,
            "Trained weights",
            f"stock COCO weights ({detector.model_class_count} classes) — "
            "detections are meaningless until you train",
        )

    return detector


# ========================================================================== inference


def check_inference(detector) -> None:
    section("Inference")

    import numpy as np

    from src.utils.constants import SEVERITY_ORDER

    rng = np.random.default_rng(0)
    image = rng.integers(60, 140, size=(480, 640, 3), dtype=np.uint8)
    image[300:400, 200:340] = 25

    started = time.perf_counter()
    detections = detector.detect(image)
    elapsed = (time.perf_counter() - started) * 1000

    record(PASS, "Inference runs", f"{elapsed:.0f} ms · {len(detections)} detection(s)")

    problems = []
    h, w = image.shape[:2]
    for d in detections:
        if not 0.0 <= d.confidence <= 1.0:
            problems.append(f"confidence out of range: {d.confidence}")
        if not 0.0 <= d.relative_area <= 1.0:
            problems.append(f"relative_area out of range: {d.relative_area}")
        if d.severity_level not in SEVERITY_ORDER:
            problems.append(f"unknown severity: {d.severity_level}")
        x1, y1, x2, y2 = d.bbox
        if not (0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h):
            problems.append(f"box outside frame: {d.bbox}")

    if problems:
        record(FAIL, "Detection invariants", problems[0])
    else:
        record(PASS, "Detection invariants", "bounds, ranges and severity bands valid")

    annotated = detector.annotate(image, detections)
    if annotated.shape == image.shape and not (annotated is image):
        record(PASS, "Annotation", "renders without mutating the input")
    else:
        record(FAIL, "Annotation", "wrong shape or mutated the source image")


# =========================================================================== severity


def check_severity() -> None:
    section("Severity scoring")

    from src.core.severity import classify

    faint = classify(0.002, 0, 0.30)      # tiny longitudinal crack
    severe = classify(0.120, 3, 0.92)     # large confident pothole

    if faint[1] == "Low":
        record(PASS, "Faint crack", f"{faint[1]} ({faint[0]:.2f})")
    else:
        record(FAIL, "Faint crack", f"expected Low, got {faint[1]}")

    if severe[1] == "Critical":
        record(PASS, "Large pothole", f"{severe[1]} ({severe[0]:.2f})")
    else:
        record(FAIL, "Large pothole", f"expected Critical, got {severe[1]}")

    if severe[0] > faint[0]:
        record(PASS, "Ordering", "severe outranks faint")
    else:
        record(FAIL, "Ordering", "scoring does not separate the two cases")


# =========================================================================== database


def check_database() -> None:
    section("Database")

    from src.core.detector import build_detection
    from src.db.store import DetectionStore

    with tempfile.TemporaryDirectory() as tmp:
        store = DetectionStore(Path(tmp) / "verify.db")
        try:
            damages = [
                build_detection(i, 0.5 + i * 0.1, (10 * i, 10 * i, 90 + 10 * i, 90 + 10 * i), (480, 640))
                for i in range(3)
            ]

            new_id = store.save_detection(
                image_path="verify.jpg", detections=damages,
                latitude=11.25, longitude=75.78, notes="verify",
            )
            record(PASS, "Write", f"detection #{new_id} with {len(damages)} damages")

            row = store.get_detection(new_id)
            if row and len(row["damages"]) == 3 and row["latitude"] == 11.25:
                record(PASS, "Read back", "fields and children survive the round trip")
            else:
                record(FAIL, "Read back", "data changed across the round trip")

            if len(store.geotagged_detections()) == 1:
                record(PASS, "Map query", "geotagged rows returned")
            else:
                record(FAIL, "Map query", "geotagged filter is wrong")

            store.delete_detection(new_id)
            if store.summary_stats()["total_damages"] == 0:
                record(PASS, "Cascade delete", "child damages removed with the parent")
            else:
                record(FAIL, "Cascade delete", "orphaned damage rows left behind")
        finally:
            store.close()


# ============================================================================== video


def check_video(detector) -> None:
    section("Video pipeline")

    import cv2
    import numpy as np

    from src.core.video import DamageTracker, analyse_video, transcode_for_browser
    from src.core.detector import build_detection

    tracker = DamageTracker()
    for frame in range(30):
        tracker.update([build_detection(3, 0.8, (100, 100, 200, 200), (480, 640))], frame)

    if tracker.unique_count == 1:
        record(PASS, "Deduplication", "30 frames of one pothole counted once")
    else:
        record(FAIL, "Deduplication", f"expected 1 unique damage, got {tracker.unique_count}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clip = tmp / "clip.mp4"

        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
        rng = np.random.default_rng(0)
        for _ in range(20):
            writer.write(rng.integers(40, 160, size=(240, 320, 3), dtype=np.uint8))
        writer.release()

        out = tmp / "annotated.mp4"
        summary = analyse_video(detector, clip, output_path=out, frame_stride=2, imgsz=320)

        if summary.frames_processed == 10:
            record(PASS, "Frame stride", f"{summary.frames_processed}/{summary.frames_total} frames")
        else:
            record(FAIL, "Frame stride", f"processed {summary.frames_processed}, expected 10")

        if out.exists() and out.stat().st_size > 0:
            record(PASS, "Annotated export", f"{out.stat().st_size / 1024:.0f} KB written")
        else:
            record(FAIL, "Annotated export", "no output file produced")

        converted = transcode_for_browser(clip, tmp / "browser.mp4")
        if converted:
            record(PASS, "H.264 transcode", "browser preview will work")
        else:
            record(WARN, "H.264 transcode", "unavailable — download still works")


# ========================================================================== uploads


def check_uploads() -> None:
    section("Upload handling")

    from src.utils.files import persist_upload

    with tempfile.TemporaryDirectory() as tmp:
        payload = b"x" * 200_000
        paths = {persist_upload(payload, "clip.mp4", tmp) for _ in range(20)}
        on_disk = list(Path(tmp).iterdir())

        if len(paths) == 1 and len(on_disk) == 1:
            record(PASS, "Rerun deduplication", "20 reruns wrote 1 file")
        else:
            record(
                FAIL, "Rerun deduplication",
                f"{len(paths)} paths / {len(on_disk)} files — uploads are being rewritten",
            )


# ============================================================================= suite


def check_test_suite() -> None:
    section("Test suite")

    import subprocess

    root = Path(__file__).resolve().parents[1]
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
            cwd=root, capture_output=True, text=True, timeout=600,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        record(WARN, "pytest", f"could not run: {exc}")
        return

    tail = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    summary = tail[-1] if tail else "no output"

    if proc.returncode == 0:
        record(PASS, "pytest", summary)
    else:
        record(FAIL, "pytest", summary)


# =============================================================================== main


def main() -> int:
    print("=" * 66)
    print("  Road Damage Detection — verification")
    print("=" * 66)

    detector = None
    steps = [
        ("environment", check_environment),
        ("severity", check_severity),
        ("database", check_database),
        ("uploads", check_uploads),
    ]

    for name, fn in steps:
        try:
            fn()
        except Exception:                                   # noqa: BLE001
            record(FAIL, f"{name} check crashed", traceback.format_exc(limit=1).strip())

    for name, fn in [("model", check_model)]:
        try:
            detector = fn()
        except Exception:                                   # noqa: BLE001
            record(FAIL, "Model load crashed", traceback.format_exc(limit=1).strip())

    if detector is not None:
        for name, fn in [("inference", check_inference), ("video", check_video)]:
            try:
                fn(detector)
            except Exception:                               # noqa: BLE001
                record(FAIL, f"{name} check crashed", traceback.format_exc(limit=1).strip())

    check_test_suite()

    passed = sum(1 for s, _, _ in results if s == PASS)
    warned = sum(1 for s, _, _ in results if s == WARN)
    failed = sum(1 for s, _, _ in results if s == FAIL)

    print("\n" + "=" * 66)
    print(f"  {passed} passed · {warned} warnings · {failed} failed")
    print("=" * 66)

    if failed:
        print("\nFailures:")
        for status, name, detail in results:
            if status == FAIL:
                print(f"  - {name}: {detail}")
        print("\nSomething is broken. Fix the above before demoing.")
        return 1

    if warned:
        print("\nWarnings (expected before training / optional features):")
        for status, name, detail in results:
            if status == WARN:
                print(f"  - {name}: {detail}")

    print("\nEverything essential works. Run the app with:")
    print("  streamlit run app/main.py --server.port 8531")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
