"""SORT pipeline for MOTChallenge public detections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from motlab.detectors.mot_public_detection import load_mot_public_detections
from motlab.evaluation.mot_format import write_mot_tracking_results
from motlab.trackers.sort.tracker import SortTracker


@dataclass(frozen=True)
class SortPipelineResult:
    """Summary of one SORT MOT public detection pipeline run."""

    detection_path: Path
    output_path: Path
    frame_count: int
    input_detection_count: int
    output_track_count: int
    max_frame: int
    tracker_config: dict[str, float | int]


def run_sort_on_mot_detections(
    detection_path: str | Path,
    output_path: str | Path,
    min_confidence: float = 0.0,
    max_age: int = 1,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    max_frame: int | None = None,
) -> SortPipelineResult:
    """Run frame-level SORT on MOTChallenge public detection rows."""
    detection_file = Path(detection_path)
    output_file = Path(output_path)

    all_detections_by_frame = load_mot_public_detections(
        detection_file,
        min_confidence=float("-inf"),
    )
    if not all_detections_by_frame:
        raise ValueError(f"MOT public detection file is empty: {detection_file}")

    last_detection_frame = max(all_detections_by_frame)
    requested_max_frame = last_detection_frame if max_frame is None else max_frame
    if requested_max_frame < last_detection_frame:
        raise ValueError(
            "max_frame must be greater than or equal to the last detection frame "
            f"({last_detection_frame})."
        )

    detections_by_frame = load_mot_public_detections(
        detection_file,
        min_confidence=min_confidence,
    )
    input_detection_count = sum(len(detections) for detections in detections_by_frame.values())

    tracker = SortTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    track_results = []
    for frame in range(1, requested_max_frame + 1):
        track_results.extend(tracker.update(detections_by_frame.get(frame, [])))

    write_mot_tracking_results(track_results, output_file)

    return SortPipelineResult(
        detection_path=detection_file,
        output_path=output_file,
        frame_count=requested_max_frame,
        input_detection_count=input_detection_count,
        output_track_count=len(track_results),
        max_frame=requested_max_frame,
        tracker_config={
            "max_age": max_age,
            "min_hits": min_hits,
            "iou_threshold": iou_threshold,
            "min_confidence": min_confidence,
        },
    )
