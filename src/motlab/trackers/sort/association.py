"""Hungarian association utilities for SORT-style IoU matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AssociationResult:
    """Association output using matrix row and column indices."""

    matches: list[tuple[int, int]]
    unmatched_tracks: list[int]
    unmatched_detections: list[int]


def associate_by_iou(iou_matrix: np.ndarray, iou_threshold: float) -> AssociationResult:
    """Associate tracks and detections with Hungarian assignment over IoU."""
    if iou_matrix.ndim != 2:
        raise ValueError("iou_matrix must be a 2D array.")

    track_count, detection_count = iou_matrix.shape
    if track_count == 0:
        return AssociationResult(
            matches=[],
            unmatched_tracks=[],
            unmatched_detections=list(range(detection_count)),
        )
    if detection_count == 0:
        return AssociationResult(
            matches=[],
            unmatched_tracks=list(range(track_count)),
            unmatched_detections=[],
        )

    row_indices, column_indices = linear_sum_assignment(1.0 - iou_matrix)

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()

    for track_index, detection_index in zip(row_indices.tolist(), column_indices.tolist()):
        if iou_matrix[track_index, detection_index] < iou_threshold:
            continue
        matches.append((track_index, detection_index))
        matched_tracks.add(track_index)
        matched_detections.add(detection_index)

    unmatched_tracks = [index for index in range(track_count) if index not in matched_tracks]
    unmatched_detections = [index for index in range(detection_count) if index not in matched_detections]

    return AssociationResult(
        matches=matches,
        unmatched_tracks=unmatched_tracks,
        unmatched_detections=unmatched_detections,
    )
