import numpy as np
import pytest

from motlab.core.types import BoundingBoxTLWH
from motlab.trackers.sort.iou import compute_iou, compute_iou_matrix


def box(left: float, top: float, width: float, height: float) -> BoundingBoxTLWH:
    return BoundingBoxTLWH(left=left, top=top, width=width, height=height)


def test_compute_iou_returns_one_for_identical_boxes():
    assert compute_iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == 1.0


def test_compute_iou_returns_zero_for_non_overlapping_boxes():
    assert compute_iou(box(0, 0, 10, 10), box(20, 20, 10, 10)) == 0.0


def test_compute_iou_returns_expected_value_for_partial_overlap():
    result = compute_iou(box(0, 0, 10, 10), box(5, 5, 10, 10))

    assert result == pytest.approx(25 / 175)


def test_compute_iou_matrix_returns_track_by_detection_scores():
    matrix = compute_iou_matrix(
        track_boxes=[box(0, 0, 10, 10), box(20, 20, 10, 10)],
        detection_boxes=[box(0, 0, 10, 10), box(25, 25, 10, 10)],
    )

    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, 25 / 175],
        ]
    )
    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(matrix, expected)


def test_compute_iou_matrix_handles_empty_inputs():
    detections = [box(0, 0, 10, 10), box(10, 10, 10, 10)]
    tracks = [box(0, 0, 10, 10)]

    assert compute_iou_matrix([], detections).shape == (0, 2)
    assert compute_iou_matrix(tracks, []).shape == (1, 0)
    assert compute_iou_matrix([], []).shape == (0, 0)
