import json
import subprocess
import sys
from pathlib import Path

from motlab.pipelines.sort_mot_pipeline import run_sort_sequence_evaluation


def _write_fake_sequence(tmp_path: Path) -> Path:
    sequence_dir = tmp_path / "MOT17-02-FRCNN"
    (sequence_dir / "det").mkdir(parents=True)
    (sequence_dir / "img1").mkdir()
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


def _write_fake_trackeval(tmp_path: Path) -> Path:
    trackeval_root = tmp_path / "TrackEval"
    script_path = trackeval_root / "scripts" / "run_mot_challenge.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text(
        "import sys\n"
        "print('fake TrackEval ok')\n"
        "print('fake TrackEval warning', file=sys.stderr)\n",
        encoding="utf-8",
    )
    return trackeval_root


def test_run_sort_sequence_evaluation_dry_run_creates_outputs_without_stdout_stderr(tmp_path):
    sequence_dir = _write_fake_sequence(tmp_path)
    trackeval_root = _write_fake_trackeval(tmp_path)

    result = run_sort_sequence_evaluation(
        sequence_dir=sequence_dir,
        output_root=tmp_path / "runs",
        trackeval_output_root=tmp_path / "trackeval",
        trackeval_log_root=tmp_path / "trackeval_logs",
        trackeval_root=trackeval_root,
        execute_trackeval=False,
    )

    assert result.sequence_info.name == "MOT17-02-FRCNN"
    assert result.executed_trackeval is False
    assert result.sequence_run_result.experiment_result.tracks_path.exists()
    assert result.trackeval_layout_result.tracker_result_path.exists()
    assert result.trackeval_layout_result.seqmap_path.exists()
    assert result.trackeval_run_result.command_path.exists()
    assert result.trackeval_run_result.stdout_path is None
    assert result.trackeval_run_result.stderr_path is None
    assert not (result.evaluation_log_dir / "stdout.txt").exists()
    assert not (result.evaluation_log_dir / "stderr.txt").exists()

    manifest = json.loads(
        result.sequence_run_result.experiment_result.manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["trackeval_layout_tracker_result_path"] == str(
        result.trackeval_layout_result.tracker_result_path
    )
    assert manifest["trackeval_seqmap_path"] == str(result.trackeval_layout_result.seqmap_path)
    assert manifest["trackeval_command_path"] == str(result.trackeval_run_result.command_path)
    assert manifest["trackeval_stdout_path"] is None
    assert manifest["trackeval_stderr_path"] is None
    assert manifest["trackeval_executed"] is False
    assert manifest["trackeval_returncode"] is None
    assert manifest["gt_folder"] == str(sequence_dir.parent)
    assert manifest["trackers_folder"] == str(result.trackers_folder)
    assert manifest["seqmap_file"] == str(result.seqmap_file)
    assert manifest["tracker_name"] == "sort"
    assert manifest["sequence_name_for_eval"] == "MOT17-02-FRCNN"


def test_run_sort_sequence_evaluation_uses_sequence_name_override(tmp_path):
    sequence_dir = _write_fake_sequence(tmp_path)

    result = run_sort_sequence_evaluation(
        sequence_dir=sequence_dir,
        output_root=tmp_path / "runs",
        trackeval_output_root=tmp_path / "trackeval",
        trackeval_log_root=tmp_path / "trackeval_logs",
        trackeval_root=tmp_path / "missing" / "TrackEval",
        sequence_name="MOT17-02",
    )

    assert result.trackeval_layout_result.sequence_name == "MOT17-02"
    assert result.trackeval_layout_result.tracker_result_path.name == "MOT17-02.txt"
    assert result.trackeval_layout_result.seqmap_path.read_text(encoding="utf-8") == (
        "name\nMOT17-02\n"
    )


def test_run_sort_sequence_evaluation_uses_explicit_gt_folder_in_command(tmp_path):
    sequence_dir = _write_fake_sequence(tmp_path)
    gt_folder = tmp_path / "custom_gt"

    result = run_sort_sequence_evaluation(
        sequence_dir=sequence_dir,
        output_root=tmp_path / "runs",
        trackeval_output_root=tmp_path / "trackeval",
        trackeval_log_root=tmp_path / "trackeval_logs",
        trackeval_root=tmp_path / "missing" / "TrackEval",
        gt_folder=gt_folder,
    )

    command_text = result.trackeval_run_result.command_path.read_text(encoding="utf-8")
    assert str(gt_folder) in command_text
    assert result.gt_folder == gt_folder


def test_run_sort_sequence_evaluation_execute_uses_fake_trackeval(tmp_path):
    sequence_dir = _write_fake_sequence(tmp_path)
    trackeval_root = _write_fake_trackeval(tmp_path)

    result = run_sort_sequence_evaluation(
        sequence_dir=sequence_dir,
        output_root=tmp_path / "runs",
        trackeval_output_root=tmp_path / "trackeval",
        trackeval_log_root=tmp_path / "trackeval_logs",
        trackeval_root=trackeval_root,
        execute_trackeval=True,
    )

    assert result.executed_trackeval is True
    assert result.trackeval_run_result.executed is True
    assert result.trackeval_run_result.returncode == 0
    assert result.trackeval_run_result.stdout_path.read_text(encoding="utf-8") == (
        "fake TrackEval ok\n"
    )
    assert result.trackeval_run_result.stderr_path.read_text(encoding="utf-8") == (
        "fake TrackEval warning\n"
    )


def test_cli_run_sort_sequence_eval_dry_run_succeeds(tmp_path):
    sequence_dir = _write_fake_sequence(tmp_path)
    trackeval_root = _write_fake_trackeval(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "motlab.app.cli_main",
            "run-sort-sequence-eval",
            "--sequence-dir",
            str(sequence_dir),
            "--output-root",
            str(tmp_path / "runs"),
            "--trackeval-output-root",
            str(tmp_path / "trackeval"),
            "--trackeval-log-root",
            str(tmp_path / "trackeval_logs"),
            "--trackeval-root",
            str(trackeval_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "sequence_name: MOT17-02-FRCNN" in completed.stdout
    assert "trackeval_executed: False" in completed.stdout
    assert "trackeval_returncode: None" in completed.stdout
    assert len(list((tmp_path / "trackeval_logs").glob("*/command.txt"))) == 1


def test_cli_run_sort_sequence_eval_execute_uses_fake_trackeval(tmp_path):
    sequence_dir = _write_fake_sequence(tmp_path)
    trackeval_root = _write_fake_trackeval(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "motlab.app.cli_main",
            "run-sort-sequence-eval",
            "--sequence-dir",
            str(sequence_dir),
            "--output-root",
            str(tmp_path / "runs"),
            "--trackeval-output-root",
            str(tmp_path / "trackeval"),
            "--trackeval-log-root",
            str(tmp_path / "trackeval_logs"),
            "--trackeval-root",
            str(trackeval_root),
            "--execute-trackeval",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "trackeval_executed: True" in completed.stdout
    assert "trackeval_returncode: 0" in completed.stdout
    assert len(list((tmp_path / "trackeval_logs").glob("*/stdout.txt"))) == 1
    assert len(list((tmp_path / "trackeval_logs").glob("*/stderr.txt"))) == 1
