import pytest

from motlab.core.types import BoundingBoxTLWH
from motlab.trackers.sort.track import SortTrack


def bbox() -> BoundingBoxTLWH:
    return BoundingBoxTLWH(left=10, top=20, width=30, height=40)


def assert_positive_bbox(box: BoundingBoxTLWH) -> None:
    assert box.width > 0
    assert box.height > 0


def test_sort_track_initializes_from_bbox_and_track_id():
    track = SortTrack(track_id=1, initial_bbox=bbox())

    assert track.track_id == 1
    assert track.age == 0
    assert track.hits == 1
    assert track.hit_streak == 1
    assert track.time_since_update == 0
    assert track.get_state().width == pytest.approx(30)
    assert track.get_state().height == pytest.approx(40)


def test_sort_track_rejects_non_positive_track_id():
    with pytest.raises(ValueError, match="track_id"):
        SortTrack(track_id=0, initial_bbox=bbox())


def test_sort_track_predict_updates_age_and_time_since_update():
    track = SortTrack(track_id=1, initial_bbox=bbox())

    prediction = track.predict()

    assert isinstance(prediction, BoundingBoxTLWH)
    assert track.age == 1
    assert track.time_since_update == 1
    assert track.hits == 1
    assert track.hit_streak == 1


def test_sort_track_update_resets_time_since_update_and_counts_hits():
    track = SortTrack(track_id=1, initial_bbox=bbox())
    track.predict()

    updated = track.update(bbox())

    assert isinstance(updated, BoundingBoxTLWH)
    assert track.time_since_update == 0
    assert track.hits == 2
    assert track.hit_streak == 2


def test_sort_track_predict_resets_hit_streak_after_missed_frame():
    track = SortTrack(track_id=1, initial_bbox=bbox())

    track.predict()
    track.predict()

    assert track.age == 2
    assert track.time_since_update == 2
    assert track.hit_streak == 0


def test_sort_track_predict_update_flow_keeps_positive_bbox_size():
    track = SortTrack(track_id=1, initial_bbox=bbox())

    for _ in range(5):
        prediction = track.predict()
        updated = track.update(bbox())

        assert_positive_bbox(prediction)
        assert_positive_bbox(updated)
        assert track.time_since_update == 0

    assert track.age == 5
    assert track.hits == 6
    assert track.hit_streak == 6
