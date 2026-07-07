import pytest

from motlab.core.types import BoundingBoxTLWH, Detection, TrackResult


def test_mot_data_models_store_1_based_frame_and_tlwh_bbox():
    bbox = BoundingBoxTLWH(left=10.0, top=20.0, width=30.0, height=40.0)
    detection = Detection(frame=1, bbox=bbox, confidence=0.9)
    track = TrackResult(frame=1, track_id=7, bbox=bbox, confidence=0.8)

    assert detection.frame == 1
    assert detection.bbox == bbox
    assert detection.confidence == 0.9
    assert detection.class_id is None
    assert detection.label is None
    assert track.track_id == 7
    assert track.bbox.width == 30.0


@pytest.mark.parametrize("width", [0, -1])
def test_bounding_box_rejects_non_positive_width(width):
    with pytest.raises(ValueError, match="width"):
        BoundingBoxTLWH(left=10.0, top=20.0, width=width, height=40.0)


@pytest.mark.parametrize("height", [0, -1])
def test_bounding_box_rejects_non_positive_height(height):
    with pytest.raises(ValueError, match="height"):
        BoundingBoxTLWH(left=10.0, top=20.0, width=30.0, height=height)


def test_detection_rejects_non_1_based_frame():
    bbox = BoundingBoxTLWH(left=10.0, top=20.0, width=30.0, height=40.0)

    with pytest.raises(ValueError, match="frame"):
        Detection(frame=0, bbox=bbox, confidence=0.9)


def test_track_result_rejects_non_1_based_frame():
    bbox = BoundingBoxTLWH(left=10.0, top=20.0, width=30.0, height=40.0)

    with pytest.raises(ValueError, match="frame"):
        TrackResult(frame=0, track_id=1, bbox=bbox, confidence=0.9)


def test_track_result_rejects_non_positive_track_id():
    bbox = BoundingBoxTLWH(left=10.0, top=20.0, width=30.0, height=40.0)

    with pytest.raises(ValueError, match="track_id"):
        TrackResult(frame=1, track_id=0, bbox=bbox, confidence=0.9)
