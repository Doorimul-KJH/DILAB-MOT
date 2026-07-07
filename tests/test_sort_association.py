import numpy as np
import pytest

from motlab.trackers.sort.association import AssociationResult, associate_by_iou


def test_associate_by_iou_returns_clear_one_to_one_matches():
    result = associate_by_iou(
        np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
            ]
        ),
        iou_threshold=0.3,
    )

    assert result == AssociationResult(
        matches=[(0, 0), (1, 1)],
        unmatched_tracks=[],
        unmatched_detections=[],
    )


def test_associate_by_iou_discards_matches_below_threshold():
    result = associate_by_iou(np.array([[0.2]]), iou_threshold=0.3)

    assert result.matches == []
    assert result.unmatched_tracks == [0]
    assert result.unmatched_detections == [0]


def test_associate_by_iou_handles_no_tracks():
    result = associate_by_iou(np.empty((0, 3)), iou_threshold=0.3)

    assert result.matches == []
    assert result.unmatched_tracks == []
    assert result.unmatched_detections == [0, 1, 2]


def test_associate_by_iou_handles_no_detections():
    result = associate_by_iou(np.empty((2, 0)), iou_threshold=0.3)

    assert result.matches == []
    assert result.unmatched_tracks == [0, 1]
    assert result.unmatched_detections == []


def test_associate_by_iou_returns_expected_pairs_for_multiple_candidates():
    result = associate_by_iou(
        np.array(
            [
                [0.1, 0.7, 0.2],
                [0.6, 0.4, 0.2],
                [0.1, 0.2, 0.9],
            ]
        ),
        iou_threshold=0.5,
    )

    assert result.matches == [(0, 1), (1, 0), (2, 2)]
    assert result.unmatched_tracks == []
    assert result.unmatched_detections == []


def test_associate_by_iou_leaves_unmatched_indices_after_assignment():
    result = associate_by_iou(
        np.array(
            [
                [0.9, 0.1, 0.0],
                [0.2, 0.1, 0.0],
            ]
        ),
        iou_threshold=0.3,
    )

    assert result.matches == [(0, 0)]
    assert result.unmatched_tracks == [1]
    assert result.unmatched_detections == [1, 2]


@pytest.mark.parametrize("iou_threshold", [-0.1, 1.1])
def test_associate_by_iou_rejects_threshold_outside_unit_interval(iou_threshold):
    with pytest.raises(ValueError, match="iou_threshold"):
        associate_by_iou(np.array([[0.5]]), iou_threshold=iou_threshold)


def test_associate_by_iou_rejects_nan_values():
    with pytest.raises(ValueError, match="finite"):
        associate_by_iou(np.array([[np.nan]]), iou_threshold=0.3)


def test_associate_by_iou_rejects_inf_values():
    with pytest.raises(ValueError, match="finite"):
        associate_by_iou(np.array([[np.inf]]), iou_threshold=0.3)


def test_associate_by_iou_rejects_negative_iou_values():
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        associate_by_iou(np.array([[-0.1]]), iou_threshold=0.3)


def test_associate_by_iou_rejects_iou_values_greater_than_one():
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        associate_by_iou(np.array([[1.1]]), iou_threshold=0.3)


def test_associate_by_iou_rejects_1d_arrays():
    with pytest.raises(ValueError, match="2D"):
        associate_by_iou(np.array([0.5, 0.6]), iou_threshold=0.3)
