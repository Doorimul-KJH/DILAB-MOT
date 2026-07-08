import subprocess
import sys

import pytest

from motlab.evaluation.trackeval_runner import (
    build_trackeval_mot_command,
    check_trackeval_available,
    run_trackeval_mot_command,
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


def test_run_trackeval_mot_command_dry_run_writes_command_without_subprocess(tmp_path, monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("subprocess.run should not be called for dry-run")

    monkeypatch.setattr("motlab.evaluation.trackeval_runner.subprocess.run", fake_run)
    command = [sys.executable, "fake_trackeval.py", "--TRACKERS_TO_EVAL", "sort"]

    result = run_trackeval_mot_command(command, output_dir=tmp_path, execute=False)

    assert calls == []
    assert result.executed is False
    assert result.returncode is None
    assert result.command_path == tmp_path / "command.txt"
    assert result.stdout_path is None
    assert result.stderr_path is None
    assert result.command_path.read_text(encoding="utf-8") == " ".join(command) + "\n"
    assert "Dry run" in result.message


def test_run_trackeval_mot_command_rejects_empty_command(tmp_path):
    with pytest.raises(ValueError, match="command"):
        run_trackeval_mot_command([], output_dir=tmp_path)


def test_run_trackeval_mot_command_execute_writes_stdout_and_stderr(tmp_path):
    script_path = tmp_path / "fake_trackeval.py"
    script_path.write_text(
        "import sys\n"
        "print('metric summary')\n"
        "print('warning detail', file=sys.stderr)\n",
        encoding="utf-8",
    )
    command = [sys.executable, str(script_path)]

    result = run_trackeval_mot_command(command, output_dir=tmp_path / "logs", execute=True)

    assert result.executed is True
    assert result.returncode == 0
    assert result.stdout == "metric summary\n"
    assert result.stderr == "warning detail\n"
    assert result.stdout_path.read_text(encoding="utf-8") == "metric summary\n"
    assert result.stderr_path.read_text(encoding="utf-8") == "warning detail\n"


def test_run_trackeval_mot_command_records_nonzero_returncode(tmp_path):
    script_path = tmp_path / "fake_fail.py"
    script_path.write_text(
        "import sys\n"
        "print('failed stdout')\n"
        "print('failed stderr', file=sys.stderr)\n"
        "sys.exit(7)\n",
        encoding="utf-8",
    )

    result = run_trackeval_mot_command(
        [sys.executable, str(script_path)],
        output_dir=tmp_path / "logs",
        execute=True,
    )

    assert result.executed is True
    assert result.returncode == 7
    assert result.stdout == "failed stdout\n"
    assert result.stderr == "failed stderr\n"
    assert "exit code 7" in result.message


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


def test_cli_run_trackeval_default_is_dry_run_and_writes_command(tmp_path):
    output_dir = tmp_path / "logs"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "motlab.app.cli_main",
            "run-trackeval",
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
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "executed: False" in completed.stdout
    assert "returncode: None" in completed.stdout
    assert (output_dir / "command.txt").exists()
    assert not (output_dir / "stdout.txt").exists()
    assert not (output_dir / "stderr.txt").exists()


def test_cli_run_trackeval_execute_uses_fake_script(tmp_path):
    trackeval_root = tmp_path / "TrackEval"
    script_path = trackeval_root / "scripts" / "run_mot_challenge.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text(
        "import sys\n"
        "print('fake run ok')\n"
        "print('fake run warning', file=sys.stderr)\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "logs"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "motlab.app.cli_main",
            "run-trackeval",
            "--trackeval-root",
            str(trackeval_root),
            "--gt-folder",
            str(tmp_path / "datasets" / "MOT17" / "train"),
            "--trackers-folder",
            str(tmp_path / "trackeval" / "sort" / "run" / "trackers"),
            "--seqmap-file",
            str(tmp_path / "trackeval" / "sort" / "run" / "seqmaps" / "MOT17-test.txt"),
            "--tracker-name",
            "sort",
            "--output-dir",
            str(output_dir),
            "--execute",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "executed: True" in completed.stdout
    assert "returncode: 0" in completed.stdout
    assert (output_dir / "command.txt").exists()
    assert (output_dir / "stdout.txt").read_text(encoding="utf-8") == "fake run ok\n"
    assert (output_dir / "stderr.txt").read_text(encoding="utf-8") == "fake run warning\n"
