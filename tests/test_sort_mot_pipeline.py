import subprocess
import sys

import pytest

from motlab.pipelines.sort_mot_pipeline import run_sort_on_mot_detections


FIXTURE_DET = "tests/fixtures/mot/det.txt"
FIXTURE_DET_WITH_GAP = "tests/fixtures/mot/det_with_gap.txt"


def test_sort_mot_pipeline_writes_mot_tracking_result_file(tmp_path):
    output_path = tmp_path / "sort_results.txt"

    result = run_sort_on_mot_detections(FIXTURE_DET, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert output_path.exists()
    assert len(lines) == 3
    assert all(len(line.split(",")) == 10 for line in lines)
    assert lines[0].startswith("1,1,")
    assert result.frame_count == 2
    assert result.input_detection_count == 3
    assert result.output_track_count == 3
    assert result.max_frame == 2


def test_sort_mot_pipeline_result_contains_config_and_paths(tmp_path):
    output_path = tmp_path / "nested" / "sort_results.txt"

    result = run_sort_on_mot_detections(
        FIXTURE_DET,
        output_path,
        max_age=2,
        min_hits=1,
        iou_threshold=0.25,
    )

    assert result.detection_path.name == "det.txt"
    assert result.output_path == output_path
    assert result.tracker_config == {
        "max_age": 2,
        "min_hits": 1,
        "iou_threshold": 0.25,
        "min_confidence": 0.0,
    }


def test_sort_mot_pipeline_filters_input_detections_by_confidence(tmp_path):
    output_path = tmp_path / "filtered.txt"

    result = run_sort_on_mot_detections(FIXTURE_DET, output_path, min_confidence=0.8)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert result.input_detection_count == 2
    assert result.output_track_count == 2
    assert len(lines) == 2


def test_sort_mot_pipeline_processes_gap_frames(tmp_path):
    output_path = tmp_path / "gap.txt"

    result = run_sort_on_mot_detections(FIXTURE_DET_WITH_GAP, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert result.frame_count == 3
    assert result.input_detection_count == 2
    assert result.output_track_count == 2
    assert [line.split(",")[0] for line in lines] == ["1", "3"]


def test_sort_mot_pipeline_processes_empty_frames_until_requested_max_frame(tmp_path):
    output_path = tmp_path / "extended.txt"

    result = run_sort_on_mot_detections(FIXTURE_DET_WITH_GAP, output_path, max_frame=4)

    assert result.frame_count == 4
    assert result.max_frame == 4
    assert output_path.exists()


def test_sort_mot_pipeline_rejects_max_frame_before_last_detection(tmp_path):
    with pytest.raises(ValueError, match="max_frame"):
        run_sort_on_mot_detections(FIXTURE_DET, tmp_path / "too_short.txt", max_frame=1)


def test_sort_mot_pipeline_rejects_empty_detection_file(tmp_path):
    empty_detection_path = tmp_path / "empty_det.txt"
    empty_detection_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        run_sort_on_mot_detections(empty_detection_path, tmp_path / "empty_results.txt")


def test_cli_run_sort_mot_writes_output_file(tmp_path):
    output_path = tmp_path / "cli_sort_results.txt"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "motlab.app.cli_main",
            "run-sort-mot",
            "--detections",
            FIXTURE_DET,
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    assert "Output file:" in completed.stdout
    assert "Processed frames: 2" in completed.stdout
    assert "Input detections: 3" in completed.stdout
    assert "Output track rows: 3" in completed.stdout
