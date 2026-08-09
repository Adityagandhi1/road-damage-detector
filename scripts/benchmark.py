"""Benchmark inference latency across backends and input sizes.

Prints a markdown table ready to paste into the README.

    python scripts/benchmark.py
    python scripts/benchmark.py --runs 30 --sizes 320 640
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.detector import RoadDamageDetector
from src.utils import config
from src.utils.constants import SUPPORTED_IMAGE_EXTENSIONS


def load_images(limit: int = 5) -> list[np.ndarray]:
    """Real sample images if present, otherwise deterministic synthetic frames."""
    import cv2

    paths = [
        p for p in sorted(config.SAMPLES_DIR.glob("*"))
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ][:limit]

    images = [img for p in paths if (img := cv2.imread(str(p))) is not None]
    if images:
        return images

    print("No sample images found — benchmarking on synthetic frames.\n")
    rng = np.random.default_rng(0)
    return [
        rng.integers(40, 180, size=(720, 1280, 3), dtype=np.uint8) for _ in range(3)
    ]


def bench(model_path: Path | str, imgsz: int, images: list, runs: int) -> dict | None:
    try:
        detector = RoadDamageDetector(model_path=model_path, imgsz=imgsz, warmup=True)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  skipped {Path(str(model_path)).name} @ {imgsz}px — {exc}")
        return None

    timings: list[float] = []
    for i in range(runs):
        image = images[i % len(images)]
        start = time.perf_counter()
        detector.detect(image, imgsz=imgsz)
        timings.append((time.perf_counter() - start) * 1000)

    return {
        "backend": detector.backend,
        "imgsz": imgsz,
        "mean": statistics.mean(timings),
        # Median is the honest headline: means on a laptop CPU get dragged around by
        # background scheduling, and p95 shows the tail that a user actually notices.
        "median": statistics.median(timings),
        "p95": sorted(timings)[int(len(timings) * 0.95) - 1],
        "fps": 1000 / statistics.median(timings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--sizes", type=int, nargs="+", default=[320, 416, 640])
    args = parser.parse_args()

    config.ensure_dirs()
    images = load_images()

    candidates: list[Path | str] = []
    for name in ("road_damage_best.pt", "road_damage_best.onnx"):
        if (config.MODELS_DIR / name).exists():
            candidates.append(config.MODELS_DIR / name)

    if not candidates:
        fallback, _ = config.resolve_model_path()
        print(
            "No trained weights in models/ — benchmarking stock yolo26n instead.\n"
            "Latency is representative; detections are not.\n"
        )
        candidates = [fallback]

    print(f"Benchmarking {len(candidates)} model(s), {args.runs} runs each\n")

    results = []
    for model_path in candidates:
        for imgsz in args.sizes:
            print(f"  {Path(str(model_path)).name} @ {imgsz}px …", flush=True)
            if row := bench(model_path, imgsz, images, args.runs):
                row["model"] = Path(str(model_path)).name
                results.append(row)

    if not results:
        print("\nNothing to report.")
        return

    print("\n\n### Inference benchmark\n")
    print(f"CPU-only. Median of {args.runs} runs per configuration, after warm-up.\n")
    print("| Model | Backend | Size | Median | Mean | p95 | FPS |")
    print("|:--|:--|--:|--:|--:|--:|--:|")
    for r in results:
        print(
            f"| `{r['model']}` | {r['backend']} | {r['imgsz']}px "
            f"| {r['median']:.0f} ms | {r['mean']:.0f} ms "
            f"| {r['p95']:.0f} ms | {r['fps']:.1f} |"
        )

    best = min(results, key=lambda r: r["median"])
    print(
        f"\nFastest: `{best['model']}` ({best['backend']}) at {best['imgsz']}px "
        f"— {best['median']:.0f} ms median."
    )


if __name__ == "__main__":
    main()
