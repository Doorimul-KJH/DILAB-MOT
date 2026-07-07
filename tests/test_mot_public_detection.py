from pathlib import Path

import pytest

from motlab.detectors.mot_public_detection import load_mot_public_detections


def fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "mot" / "det.txt"


def test_load_mot_public_detections_groups_rows_by_1_based_frame():
    detections_by_frame = load_mot_public_detections(fixture_path())

    assert sorted(detections_by_frame) == [1, 2]
    assert len(detections_by_frame[1]) == 2
    assert len(detections_by_frame[2]) == 1
    assert detections_by_frame[1][0].bbox.left == 10.0
    assert detections_by_frame[1][0].bbox.top == 20.0
    assert detections_by_frame[1][0].bbox.width == 30.0
    assert detections_by_frame[1][0].bbox.height == 40.0
    assert detections_by_frame[1][0].confidence == 0.9


def test_load_mot_public_detections_applies_confidence_threshold():
    detections_by_frame = load_mot_public_detections(fixture_path(), min_confidence=0.8)

    assert len(detections_by_frame[1]) == 1
    assert len(detections_by_frame[2]) == 1
    assert detections_by_frame[1][0].confidence == 0.9
    assert detections_by_frame[2][0].confidence == 0.8


def test_load_mot_public_detections_raises_value_error_for_bad_rows(tmp_path):
    bad_file = tmp_path / "bad_det.txt"
    bad_file.write_text("1,-1,10,20,30\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_mot_public_detections(bad_file)
