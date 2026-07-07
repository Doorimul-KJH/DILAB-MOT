import json
import subprocess
import sys
from pathlib import Path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    project_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-m", "motlab.app.cli_main", *args],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )


def test_cli_list_papers_runs_without_installing_package():
    result = run_cli("list-papers")

    assert result.returncode == 0
    assert "sort" in result.stdout


def test_cli_inspect_sort_shows_detector_and_yolo_note():
    result = run_cli("inspect-paper", "sort")

    assert result.returncode == 0
    assert "faster_rcnn_or_mot_public_detection" in result.stdout
    assert "YOLO" in result.stdout


def test_cli_dry_run_creates_output_files():
    result = run_cli("run", "--paper", "sort", "--dry-run")

    assert result.returncode == 0
    assert "outputs" in result.stdout
    output_line = next(line for line in result.stdout.splitlines() if "Output folder:" in line)
    output_dir = Path(output_line.split("Output folder:", 1)[1].strip())

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["paper_id"] == "sort"
    assert manifest["dry_run"] is True
