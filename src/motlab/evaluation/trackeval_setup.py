"""Explicit TrackEval preparation helper."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from motlab.core.config import find_project_root


DEFAULT_TRACKEVAL_REPO_URL = "https://github.com/JonathonLuiten/TrackEval.git"
DEFAULT_TRACKEVAL_REF = "master"


@dataclass(frozen=True)
class TrackEvalPrepareResult:
    """Result returned by an explicit TrackEval preparation request."""

    trackeval_root: Path
    repo_url: str
    ref: str
    cloned: bool
    already_exists: bool
    commit_hash: str | None
    setup_info_path: Path
    message: str


def prepare_trackeval(
    trackeval_root: str | Path = "third_party/TrackEval",
    repo_url: str = DEFAULT_TRACKEVAL_REPO_URL,
    ref: str = DEFAULT_TRACKEVAL_REF,
    clone: bool = False,
) -> TrackEvalPrepareResult:
    """Prepare TrackEval only when the caller explicitly requests cloning."""
    root = Path(trackeval_root)
    setup_info_path = _setup_info_path(root)
    already_exists = root.exists()
    cloned = False

    if clone and not already_exists:
        root.parent.mkdir(parents=True, exist_ok=True)
        command = ["git", "clone", "--branch", ref, repo_url, str(root)]
        _run_git_command(command)
        cloned = True
    elif not clone and not already_exists:
        message = (
            "TrackEval was not cloned. To clone explicitly, run: "
            f"git clone --branch {ref} {repo_url} {root}"
        )
        commit_hash = None
        _write_setup_info(
            setup_info_path=setup_info_path,
            trackeval_root=root,
            repo_url=repo_url,
            ref=ref,
            commit_hash=commit_hash,
        )
        return TrackEvalPrepareResult(
            trackeval_root=root,
            repo_url=repo_url,
            ref=ref,
            cloned=False,
            already_exists=False,
            commit_hash=commit_hash,
            setup_info_path=setup_info_path,
            message=message,
        )

    commit_hash = _read_commit_hash(root) if root.exists() else None
    _write_setup_info(
        setup_info_path=setup_info_path,
        trackeval_root=root,
        repo_url=repo_url,
        ref=ref,
        commit_hash=commit_hash,
    )

    if already_exists:
        message = f"TrackEval root already exists; clone was skipped: {root}"
    elif cloned:
        message = f"TrackEval was cloned explicitly into: {root}"
    else:
        message = (
            "TrackEval was not cloned. To clone explicitly, run: "
            f"git clone --branch {ref} {repo_url} {root}"
        )

    return TrackEvalPrepareResult(
        trackeval_root=root,
        repo_url=repo_url,
        ref=ref,
        cloned=cloned,
        already_exists=already_exists,
        commit_hash=commit_hash,
        setup_info_path=setup_info_path,
        message=message,
    )


def _setup_info_path(trackeval_root: Path) -> Path:
    return trackeval_root.with_name(f"{trackeval_root.name}_SETUP.json")


def _run_git_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"TrackEval preparation git command failed: {' '.join(command)}\n{detail}")
    return completed


def _read_commit_hash(trackeval_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(trackeval_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _write_setup_info(
    setup_info_path: Path,
    trackeval_root: Path,
    repo_url: str,
    ref: str,
    commit_hash: str | None,
) -> None:
    setup_info_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "repo_url": repo_url,
        "ref": ref,
        "commit_hash": commit_hash,
        "trackeval_root": str(trackeval_root),
        "prepared_at": datetime.now().astimezone().isoformat(),
    }
    setup_info_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
