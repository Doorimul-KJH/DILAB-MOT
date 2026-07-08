import json
import subprocess
import sys
from pathlib import Path

import pytest

from motlab.datasets.motchallenge import load_motchallenge_sequence_info
from motlab.pipelines.sort_mot_pipeline import run_sort_on_mot_sequence


def _write_fake_sequence(tmp_path: Path, include_gt: bool = True) -> Path:
    sequence_dir = tmp_path / "MOT17-02-FRCNN"
    (sequence_dir / "det").mkdir(parents=True)
    (sequence_dir / "img1").mkdir()
    if include_gt:
        (sequence_dir / "gt").mkdir()
        (sequence_dir / "gt" / "gt.txt").write_text("", encoding="utf-8")

    (sequence_dir / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                "name=MOT17-02-FRCNN",
                "imDir=img1",
                "frameRate=30",
                "seqLength=2",
                "imWidth=1920",
                "imHeight=1080",
                "imExt=.jpg",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (sequence_dir / "det" / "det.txt").write_text(
        "\n".join(
            [
                "1,-1,10,20,30,40,0.90,-1,-1,-1",
                "2,-1,12,22,30,40,0.80,-1,-1,-1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return sequence_dir


def test_load_motchallenge_sequence_info_parses_seqinfo_and_paths(tmp_path):
    sequence_dir = _write_fake_sequence(tmp_path)

    info = load_motchallenge_sequence_info(sequence_dir, require_gt=True)

    assert info.sequence_dir == sequence_dir
    assert info.name == "MOT17-02-FRCNN"
    assert info.seqinfo_path == sequence_dir / "seqinfo.ini"
    assert info.detection_path == sequence_dir / "det" / "det.txt"
    assert info.gt_path == sequence_dir / "gt" / "gt.txt"
    assert info.image_dir == sequence_dir / "img1"
    assert info.seq_length == 2
    assert info.frame_rate == 30
    assert info.image_width == 1920
    assert info.image_height == 1080
    assert info.image_extension == ".jpg"


def test_load_motchallenge_sequence_info_requires_gt_when_requested(tmp_path):
    sequence_dir = _write_fake_sequence(tmp_path, include_gt=False)

    with pytest.raises(FileNotFoundError, match="gt.txt"):
        load_motchallenge_sequence_info(sequence_dir, require_gt=True)


def test_load_motchallenge_sequence_info_requires_detection_by_default(tmp_path):
    sequence_dir = _write_fake_sequence(tmp_path)
    (sequence_dir / "det" / "det.txt").unlink()

    with pytest.raises(FileNotFoundError, match="det.txt"):
        load_motchallenge_sequence_info(sequence_dir)


def test_load_motchallenge_sequence_info_rejects_missing_seqinfo(tmp_path):
    sequence_dir = tmp_path / "MOT17-02-FRCNN"
    sequence_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="seqinfo.ini"):
        load_motchallenge_sequence_info(sequence_dir)


def test_load_motchallenge_sequence_info_rejects_missing_sequence_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="sequence_dir"):
        load_motchallenge_sequence_info(tmp_path / "missing")


def test_run_sort_on_mot_sequence_creates_run_folder_and_manifest(tmp_path):
    sequence_dir = _write_fake_sequence(tmp_path)

    result = run_sort_on_mot_sequence(sequence_dir, output_root=tmp_path / "runs")

    assert result.sequence_info.name == "MOT17-02-FRCNN"
    assert result.experiment_result.tracks_path.exists()
    assert result.experiment_result.pipeline_result.frame_count == 2
    assert result.experiment_result.pipeline_result.input_detection_count == 2

    manifest = json.loads(
        result.experiment_result.manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["sequence_name"] == "MOT17-02-FRCNN"
    assert manifest["sequence_dir"] == str(sequence_dir)
    assert manifest["seq_length"] == 2
    assert manifest["frame_rate"] == 30
    assert manifest["image_width"] == 1920
    assert manifest["image_height"] == 1080


def test_cli_inspect_mot_sequence_succeeds(tmp_path):
    sequence_dir = _write_fake_sequence(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "motlab.app.cli_main",
            "inspect-mot-sequence",
            "--sequence-dir",
            str(sequence_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "sequence_name: MOT17-02-FRCNN" in completed.stdout
    assert "detection_path:" in completed.stdout
    assert "gt_path:" in completed.stdout
    assert "seq_length: 2" in completed.stdout
    assert "frame_rate: 30" in completed.stdout
    assert "image_size: 1920x1080" in completed.stdout


def test_cli_run_sort_sequence_succeeds(tmp_path):
    sequence_dir = _write_fake_sequence(tmp_path)
    output_root = tmp_path / "runs"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "motlab.app.cli_main",
            "run-sort-sequence",
            "--sequence-dir",
            str(sequence_dir),
            "--output-root",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "sequence_name: MOT17-02-FRCNN" in completed.stdout
    assert "output_dir:" in completed.stdout
    assert "tracks_path:" in completed.stdout
    assert "processed frames: 2" in completed.stdout
    assert "input detections: 2" in completed.stdout
    assert "output track rows: 2" in completed.stdout
    assert len(list(output_root.glob("*_sort_mot/tracks.txt"))) == 1
