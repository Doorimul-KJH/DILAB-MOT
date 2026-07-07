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
