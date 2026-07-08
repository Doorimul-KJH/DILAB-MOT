import subprocess
import sys

from motlab.evaluation.trackeval_runner import (
    build_trackeval_mot_command,
    check_trackeval_available,
)


def test_check_trackeval_available_handles_missing_root(tmp_path):
    result = check_trackeval_available(tmp_path / "missing" / "TrackEval")

    assert result.exists is False
    assert result.script_path is None
    assert result.can_import_or_help is False
    assert "not found" in result.message


def test_check_trackeval_available_finds_script_path(tmp_path):
    trackeval_root = tmp_path / "TrackEval"
    script_path = trackeval_root / "scripts" / "run_mot_challenge.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("print('fake TrackEval script')\n", encoding="utf-8")

    result = check_trackeval_available(trackeval_root)

    assert result.exists is True
    assert result.script_path == script_path


def test_check_trackeval_available_runs_help_when_script_supports_help(tmp_path):
    trackeval_root = tmp_path / "TrackEval"
    script_path = trackeval_root / "scripts" / "run_mot_challenge.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--GT_FOLDER')\n"
        "parser.parse_args()\n",
        encoding="utf-8",
    )

    result = check_trackeval_available(trackeval_root)

    assert result.can_import_or_help is True
    assert "--GT_FOLDER" in result.message


def test_build_trackeval_mot_command_contains_required_options(tmp_path):
    command = build_trackeval_mot_command(
        trackeval_root=tmp_path / "TrackEval",
        gt_folder=tmp_path / "datasets" / "MOT17" / "train",
        trackers_folder=tmp_path / "trackeval" / "sort" / "run" / "trackers",
        seqmap_file=tmp_path / "trackeval" / "sort" / "run" / "seqmaps" / "MOT17-test.txt",
        tracker_name="sort",
    )

    assert command[0] == sys.executable
    assert "run_mot_challenge.py" in command[1]
    assert "--GT_FOLDER" in command
    assert "--TRACKERS_FOLDER" in command
    assert "--SEQMAP_FILE" in command
    assert "--TRACKERS_TO_EVAL" in command
    assert "sort" in command


def test_cli_check_trackeval_does_not_fail_for_missing_root(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "motlab.app.cli_main",
            "check-trackeval",
            "--trackeval-root",
            str(tmp_path / "missing" / "TrackEval"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "TrackEval availability check completed. TrackEval root was not found." in completed.stdout
    assert "exists: False" in completed.stdout
    assert "script_path: None" in completed.stdout
    assert "can_import_or_help: False" in completed.stdout
    assert "message:" in completed.stdout
    assert "not found" in completed.stdout


def test_cli_check_trackeval_reports_available_when_help_succeeds(tmp_path):
    script_path = tmp_path / "TrackEval" / "scripts" / "run_mot_challenge.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--GT_FOLDER')\n"
        "parser.parse_args()\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "motlab.app.cli_main",
            "check-trackeval",
            "--trackeval-root",
            str(tmp_path / "TrackEval"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "TrackEval availability check completed. TrackEval appears available." in completed.stdout
    assert "exists: True" in completed.stdout
    assert f"script_path: {script_path}" in completed.stdout
    assert "can_import_or_help: True" in completed.stdout
    assert "message:" in completed.stdout


def test_cli_check_trackeval_reports_failed_help_check(tmp_path):
    script_path = tmp_path / "TrackEval" / "scripts" / "run_mot_challenge.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text(
        "import sys\n"
        "print('help failed')\n"
        "sys.exit(2)\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "motlab.app.cli_main",
            "check-trackeval",
            "--trackeval-root",
            str(tmp_path / "TrackEval"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert (
        "TrackEval availability check completed. TrackEval was found, "
        "but help/import check failed."
    ) in completed.stdout
    assert "exists: True" in completed.stdout
    assert f"script_path: {script_path}" in completed.stdout
    assert "can_import_or_help: False" in completed.stdout
    assert "message:" in completed.stdout


def test_cli_build_trackeval_command_prints_command(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "motlab.app.cli_main",
            "build-trackeval-command",
            "--trackeval-root",
            str(tmp_path / "TrackEval"),
            "--gt-folder",
            str(tmp_path / "datasets" / "MOT17" / "train"),
            "--trackers-folder",
            str(tmp_path / "trackeval" / "sort" / "run" / "trackers"),
            "--seqmap-file",
            str(tmp_path / "trackeval" / "sort" / "run" / "seqmaps" / "MOT17-test.txt"),
            "--tracker-name",
            "sort",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    stdout_lines = completed.stdout.splitlines()

    assert "TrackEval command, readable:" in completed.stdout
    assert "TrackEval command, one-line:" in completed.stdout
    assert "  --GT_FOLDER" in stdout_lines
    assert "  --TRACKERS_FOLDER" in stdout_lines
    assert "  --SEQMAP_FILE" in stdout_lines
    assert "  --TRACKERS_TO_EVAL" in stdout_lines

    one_line_index = stdout_lines.index("TrackEval command, one-line:")
    one_line_command = stdout_lines[one_line_index + 1]
    assert "--GT_FOLDER" in one_line_command
    assert "--TRACKERS_FOLDER" in one_line_command
    assert "--SEQMAP_FILE" in one_line_command
    assert "--TRACKERS_TO_EVAL sort" in one_line_command
