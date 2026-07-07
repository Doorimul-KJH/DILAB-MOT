"""MOTChallenge public detection loader."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from motlab.core.types import BoundingBoxTLWH, Detection


MOT_DETECTION_COLUMN_COUNT = 10


def load_mot_public_detections(
    detection_path: str | Path,
    min_confidence: float = 0.0,
) -> dict[int, list[Detection]]:
    """Load MOTChallenge public detections grouped by 1-based frame index."""
    path = Path(detection_path)
    if not path.exists():
        raise FileNotFoundError(f"MOT public detection file does not exist: {path}")

    detections_by_frame: dict[int, list[Detection]] = defaultdict(list)

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        for line_number, row in enumerate(reader, start=1):
            if not row or all(not column.strip() for column in row):
                continue

            detection = _parse_detection_row(row, line_number=line_number, source=path)
            if detection.confidence >= min_confidence:
                detections_by_frame[detection.frame].append(detection)

    return dict(sorted(detections_by_frame.items()))


def _parse_detection_row(row: list[str], line_number: int, source: Path) -> Detection:
    if len(row) != MOT_DETECTION_COLUMN_COUNT:
        raise ValueError(
            f"Invalid MOT detection row at {source} line {line_number}: "
            f"expected {MOT_DETECTION_COLUMN_COUNT} columns, got {len(row)}"
        )

    try:
        frame = int(row[0])
        left = float(row[2])
        top = float(row[3])
        width = float(row[4])
        height = float(row[5])
        confidence = float(row[6])
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value in MOT detection row at {source} line {line_number}"
        ) from exc

    if frame < 1:
        raise ValueError(f"MOT detection frame must be 1-based at {source} line {line_number}")

    return Detection(
        frame=frame,
        bbox=BoundingBoxTLWH(left=left, top=top, width=width, height=height),
        confidence=confidence,
    )
