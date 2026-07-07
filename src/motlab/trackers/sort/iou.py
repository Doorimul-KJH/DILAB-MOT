"""IoU utilities for SORT-style association."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from motlab.core.types import BoundingBoxTLWH


def compute_iou(box_a: BoundingBoxTLWH, box_b: BoundingBoxTLWH) -> float:
    """Compute IoU between two tlwh bounding boxes."""
    a_right = box_a.left + box_a.width
    a_bottom = box_a.top + box_a.height
    b_right = box_b.left + box_b.width
    b_bottom = box_b.top + box_b.height

    intersection_left = max(box_a.left, box_b.left)
    intersection_top = max(box_a.top, box_b.top)
    intersection_right = min(a_right, b_right)
    intersection_bottom = min(a_bottom, b_bottom)

    intersection_width = max(0.0, intersection_right - intersection_left)
    intersection_height = max(0.0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height

    if intersection_area == 0.0:
        return 0.0

    area_a = box_a.width * box_a.height
    area_b = box_b.width * box_b.height
    union_area = area_a + area_b - intersection_area
    return intersection_area / union_area


def compute_iou_matrix(
    track_boxes: Sequence[BoundingBoxTLWH],
    detection_boxes: Sequence[BoundingBoxTLWH],
) -> np.ndarray:
    """Compute a track-by-detection IoU matrix."""
    matrix = np.zeros((len(track_boxes), len(detection_boxes)), dtype=float)

    for track_index, track_box in enumerate(track_boxes):
        for detection_index, detection_box in enumerate(detection_boxes):
            matrix[track_index, detection_index] = compute_iou(track_box, detection_box)

    return matrix
