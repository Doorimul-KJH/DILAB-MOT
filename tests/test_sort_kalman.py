import pytest

from motlab.core.types import BoundingBoxTLWH
from motlab.trackers.sort.kalman import KalmanBoxTracker


def assert_bbox_close(actual: BoundingBoxTLWH, expected: BoundingBoxTLWH, abs_tol: float = 1e-6) -> None:
    assert actual.left == pytest.approx(expected.left, abs=abs_tol)
    assert actual.top == pytest.approx(expected.top, abs=abs_tol)
    assert actual.width == pytest.approx(expected.width, abs=abs_tol)
    assert actual.height == pytest.approx(expected.height, abs=abs_tol)


def test_kalman_box_tracker_initializes_from_bbox():
    bbox = BoundingBoxTLWH(left=10, top=20, width=30, height=40)
    tracker = KalmanBoxTracker(bbox)

    assert_bbox_close(tracker.get_state(), bbox)


def test_kalman_box_tracker_predict_returns_bbox():
    tracker = KalmanBoxTracker(BoundingBoxTLWH(left=10, top=20, width=30, height=40))

    prediction = tracker.predict()

    assert isinstance(prediction, BoundingBoxTLWH)
    assert prediction.width > 0
    assert prediction.height > 0


def test_kalman_box_tracker_update_returns_bbox():
    tracker = KalmanBoxTracker(BoundingBoxTLWH(left=10, top=20, width=30, height=40))

    updated = tracker.update(BoundingBoxTLWH(left=10, top=20, width=30, height=40))

    assert isinstance(updated, BoundingBoxTLWH)
    assert_bbox_close(updated, BoundingBoxTLWH(left=10, top=20, width=30, height=40), abs_tol=1e-3)


def test_kalman_box_tracker_repeated_predict_update_keeps_positive_size():
    bbox = BoundingBoxTLWH(left=10, top=20, width=30, height=40)
    tracker = KalmanBoxTracker(bbox)

    for _ in range(5):
        predicted = tracker.predict()
        updated = tracker.update(bbox)

        assert predicted.width > 0
        assert predicted.height > 0
        assert updated.width > 0
        assert updated.height > 0

    assert_bbox_close(tracker.get_state(), bbox, abs_tol=1e-2)
