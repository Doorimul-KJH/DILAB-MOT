import subprocess
import sys
from pathlib import Path


def test_cli_main_confirms_project_skeleton_ready():
    project_root = Path(__file__).resolve().parents[1]
    cli_path = project_root / "src" / "motlab" / "app" / "cli_main.py"

    result = subprocess.run(
        [sys.executable, str(cli_path)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "DILAB-MOT project skeleton is ready." in result.stdout
