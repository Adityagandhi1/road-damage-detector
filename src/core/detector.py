"""YOLO26 wrapper producing structured, severity-scored detections.

Two pieces, deliberately separated:

* ``build_detection`` - pure geometry and scoring. No model, no I/O, exactly testable.
* ``RoadDamageDetector`` - owns the model and the inference call.

Backend is chosen from the weights' file extension: ``.onnx`` runs on onnxruntime,
anything else on PyTorch. On the target CPU (Ryzen 5 5625U, no NVIDIA GPU) ONNX is
materially faster, so ``config.resolve_model_path`` prefers it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from src.core.severity import classify, severity_level
from src.utils import config
from src.utils.constants import (
    CLASS_BY_ID,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU_THRESHOLD,
    MAX_DETECTIONS,
    NUM_CLASSES,
)


@dataclass(frozen=True)
class Detection:
    """One damage instance. Immutable so it can be cached and shared freely."""

    class_id: int
    code: str
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    bbox_area: float                          # pixels squared
    relative_area: float                      # fraction of frame, 0-1
    severity_score: float
    severity_level: str
    severity_hex: str
    severity_bgr: tuple[int, int, int]

    def as_row(self) -> dict[str, Any]:
        """Flat dict for DataFrame display and CSV export."""
        return {
            "Type": self.class_name,
            "Code": self.code,
            "Confidence": round(self.confidence, 3),
            "Severity": self.severity_level,
            "Score": round(self.severity_score, 3),
            "Area %": round(self.relative_area * 100, 2),
            "Box": tuple(round(v) for v in self.bbox),
        }


def build_detection(
    class_id: int,
    confidence: float,
    bbox: Sequence[float],
    image_shape: Sequence[int],
) -> Detection:
    """Assemble a scored Detection from raw model output.

    ``image_shape`` is ``(h, w)`` or ``(h, w, c)``. The box is clamped to the frame so
    downstream crops and drawing never index out of bounds.

    Class ids are folded into the taxonomy range with a modulo. That is only ever
    exercised by stock COCO weights, which emit ids up to 79; it keeps the app runnable
    before training finishes. Callers must check ``RoadDamageDetector.is_trained`` and
    tell the user, because in that mode the labels are meaningless.
    """
    h, w = int(image_shape[0]), int(image_shape[1])

    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    x1 = min(max(x1, 0.0), float(w))
    x2 = min(max(x2, 0.0), float(w))
    y1 = min(max(y1, 0.0), float(h))
    y2 = min(max(y2, 0.0), float(h))

    box_area = (x2 - x1) * (y2 - y1)
    frame_area = float(h * w)
    relative = (box_area / frame_area) if frame_area > 0 else 0.0

    cid = int(class_id) % NUM_CLASSES
    damage = CLASS_BY_ID[cid]

    conf = float(confidence)
    score, level_name = classify(relative, cid, conf)
    level = severity_level(score)

    return Detection(
        class_id=cid,
        code=damage.code,
        class_name=damage.name,
        confidence=conf,
        bbox=(x1, y1, x2, y2),
        bbox_area=box_area,
        relative_area=relative,
        severity_score=score,
        severity_level=level_name,
        severity_hex=level.hex,
        severity_bgr=level.bgr,
    )


class RoadDamageDetector:
    """Loads a YOLO model once and turns frames into scored detections."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        imgsz: int = DEFAULT_IMAGE_SIZE,
        conf: float = DEFAULT_CONF_THRESHOLD,
        iou: float = DEFAULT_IOU_THRESHOLD,
        max_det: int = MAX_DETECTIONS,
        warmup: bool = True,
    ) -> None:
        from ultralytics import YOLO   # imported lazily; it is a heavy import

        if model_path is None:
            model_path, resolved_trained = config.resolve_model_path()
        else:
            resolved_trained = True

        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det

        self._model = YOLO(str(model_path), task="detect")
        self.backend = "onnx" if str(model_path).endswith(".onnx") else "pytorch"

        # A road-damage model has exactly 4 classes. Anything else is stock COCO
        # weights standing in before training finishes.
        n_classes = len(getattr(self._model, "names", {}) or {})
        self.is_trained = bool(resolved_trained and n_classes == NUM_CLASSES)
        self.model_class_count = n_classes

        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """First inference allocates buffers and is several times slower than steady
        state. Doing it here keeps the user's first real request honest, and keeps the
        benchmark numbers from being skewed by startup cost."""
        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self._model.predict(blank, imgsz=self.imgsz, verbose=False)

    def detect(
        self,
        source: str | Path | np.ndarray | Any,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
    ) -> list[Detection]:
        """Run inference on one image. Returns detections sorted most severe first."""
        result = self._model.predict(
            source if not isinstance(source, Path) else str(source),
            conf=self.conf if conf is None else conf,
            iou=self.iou if iou is None else iou,
            imgsz=self.imgsz if imgsz is None else imgsz,
            max_det=self.max_det,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        shape = result.orig_shape   # (h, w)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        detections = [
            build_detection(int(c), float(p), box, shape)
            for box, p, c in zip(xyxy, confs, clss)
        ]
        detections.sort(key=lambda d: d.severity_score, reverse=True)
        return detections

    def annotate(
        self,
        image: np.ndarray,
        detections: list[Detection],
        show_labels: bool = True,
    ) -> np.ndarray:
        """Draw severity-coloured boxes. Returns a new array; never mutates ``image``."""
        canvas = image.copy()
        if not detections:
            return canvas

        h, w = canvas.shape[:2]
        # Scale line and text with resolution so a 4K photo and a 480p frame both
        # come out legible.
        thickness = max(1, round(min(h, w) / 400))
        font_scale = max(0.4, min(h, w) / 1400)

        for det in detections:
            x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
            colour = det.severity_bgr
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

            if not show_labels:
                continue

            label = f"{det.class_name} {det.confidence:.0%} | {det.severity_level}"
            (tw, th), base = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Keep the label inside the frame when the box hugs the top edge.
            ty = y1 - base if y1 - th - base >= 0 else y2 + th + base
            ty = min(max(ty, th), h - base)

            cv2.rectangle(
                canvas,
                (x1, ty - th - base),
                (min(x1 + tw, w), ty + base),
                colour,
                -1,
            )
            cv2.putText(
                canvas,
                label,
                (x1, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return canvas
