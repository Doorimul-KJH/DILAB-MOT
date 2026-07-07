import pytest

from motlab.core.types import BoundingBoxTLWH, Detection, TrackResult
from motlab.trackers.sort.tracker import SortTracker


def det(
    frame: int,
    left: float,
    top: float,
    width: float = 30,
    height: float = 40,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        frame=frame,
        bbox=BoundingBoxTLWH(left=left, top=top, width=width, height=height),
        confidence=confidence,
    )


def test_sort_tracker_initializes_with_defaults():
    tracker = SortTracker()

    assert tracker.max_age == 1
    assert tracker.min_hits == 3
    assert tracker.iou_threshold == 0.3
    assert tracker.tracks == []
    assert tracker.next_track_id == 1
    assert tracker.frame_count == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_age": 0}, "max_age"),
        ({"min_hits": 0}, "min_hits"),
        ({"iou_threshold": -0.1}, "iou_threshold"),
        ({"iou_threshold": 1.1}, "iou_threshold"),
    ],
)
def test_sort_tracker_rejects_invalid_init_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SortTracker(**kwargs)


def test_first_frame_detection_creates_track_and_returns_result():
    tracker = SortTracker()

    results = tracker.update([det(frame=1, left=10, top=20, confidence=0.8)])

    assert tracker.frame_count == 1
    assert tracker.next_track_id == 2
    assert len(tracker.tracks) == 1
    assert len(results) == 1
    assert results[0] == TrackResult(
        frame=1,
        track_id=1,
        bbox=tracker.tracks[0].get_state(),
        confidence=0.8,
    )


def test_same_position_detection_keeps_same_track_id():
    tracker = SortTracker()

    first = tracker.update([det(frame=1, left=10, top=20)])
    second = tracker.update([det(frame=2, left=10, top=20)])

    assert first[0].track_id == 1
    assert second[0].track_id == 1
    assert len(tracker.tracks) == 1
    assert tracker.next_track_id == 2


def test_far_detection_creates_new_track_id():
    tracker = SortTracker(iou_threshold=0.3)

    tracker.update([det(frame=1, left=10, top=20)])
    results = tracker.update([det(frame=2, left=500, top=500)])

    assert [result.track_id for result in results] == [2]
    assert len(tracker.tracks) == 2
    assert tracker.next_track_id == 3


def test_empty_detection_frame_returns_no_results_and_ages_tracks():
    tracker = SortTracker(max_age=2)
    tracker.update([det(frame=1, left=10, top=20)])

    results = tracker.update([])

    assert results == []
    assert tracker.frame_count == 2
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].time_since_update == 1


def test_unmatched_track_is_removed_after_max_age():
    tracker = SortTracker(max_age=1)
    tracker.update([det(frame=1, left=10, top=20)])

    tracker.update([])
    tracker.update([])

    assert tracker.frame_count == 3
    assert tracker.tracks == []


def test_update_rejects_detections_from_different_frames():
    tracker = SortTracker()

    with pytest.raises(ValueError, match="same frame"):
        tracker.update([det(frame=1, left=10, top=20), det(frame=2, left=100, top=120)])


def test_multiple_detections_create_multiple_tracks_and_results():
    tracker = SortTracker()

    results = tracker.update([det(frame=1, left=10, top=20), det(frame=1, left=100, top=120)])

    assert [result.track_id for result in results] == [1, 2]
    assert len(tracker.tracks) == 2
    assert tracker.next_track_id == 3


def test_returned_track_result_contains_frame_track_bbox_and_confidence():
    tracker = SortTracker()

    result = tracker.update([det(frame=4, left=10, top=20, confidence=0.77)])[0]

    assert result.frame == 4
    assert result.track_id == 1
    assert isinstance(result.bbox, BoundingBoxTLWH)
    assert result.confidence == 0.77
