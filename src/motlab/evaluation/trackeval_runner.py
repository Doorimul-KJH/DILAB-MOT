"""TrackEval availability checks and command construction."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackEvalCheckResult:
    """TrackEval local availability check result."""

    trackeval_root: Path
    exists: bool
    script_path: Path | None
    can_import_or_help: bool
    message: str


def check_trackeval_available(
    trackeval_root: str | Path = "third_party/TrackEval",
) -> TrackEvalCheckResult:
    """Check whether a local TrackEval checkout appears runnable."""
    root = Path(trackeval_root)
    if not root.exists() or not root.is_dir():
        return TrackEvalCheckResult(
            trackeval_root=root,
            exists=False,
            script_path=None,
            can_import_or_help=False,
            message=(
                f"TrackEval root not found: {root}. "
                "Prepare third_party/TrackEval manually; this command does not clone it."
            ),
        )

    script_path = _find_run_mot_script(root)
    if script_path is None:
        return TrackEvalCheckResult(
            trackeval_root=root,
            exists=True,
            script_path=None,
            can_import_or_help=False,
            message=f"TrackEval run_mot_challenge.py script was not found under: {root}",
        )

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return TrackEvalCheckResult(
            trackeval_root=root,
            exists=True,
            script_path=script_path,
            can_import_or_help=False,
            message=f"TrackEval help command could not be executed: {exc}",
        )

    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode == 0:
        return TrackEvalCheckResult(
            trackeval_root=root,
            exists=True,
            script_path=script_path,
            can_import_or_help=True,
            message=output,
        )

    return TrackEvalCheckResult(
        trackeval_root=root,
        exists=True,
        script_path=script_path,
        can_import_or_help=False,
        message=(
            f"TrackEval help command failed with exit code {completed.returncode}: {output}"
        ),
    )


def build_trackeval_mot_command(
    trackeval_root: str | Path,
    gt_folder: str | Path,
    trackers_folder: str | Path,
    seqmap_file: str | Path,
    tracker_name: str = "sort",
) -> list[str]:
    """Build a TrackEval MOTChallenge command without executing it."""
    script_path = _find_run_mot_script(Path(trackeval_root))
    if script_path is None:
        script_path = Path(trackeval_root) / "scripts" / "run_mot_challenge.py"

    return [
        sys.executable,
        str(script_path),
        "--GT_FOLDER",
        str(Path(gt_folder)),
        "--TRACKERS_FOLDER",
        str(Path(trackers_folder)),
        "--SEQMAP_FILE",
        str(Path(seqmap_file)),
        "--TRACKERS_TO_EVAL",
        tracker_name,
    ]


def _find_run_mot_script(trackeval_root: Path) -> Path | None:
    candidates = [
        trackeval_root / "scripts" / "run_mot_challenge.py",
        trackeval_root / "TrackEval" / "scripts" / "run_mot_challenge.py",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None
