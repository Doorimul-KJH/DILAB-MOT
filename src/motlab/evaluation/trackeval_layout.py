"""Utilities for exporting MOT results into a TrackEval-friendly layout."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackEvalLayoutResult:
    """Paths produced by a TrackEval layout export."""

    run_dir: Path
    sequence_name: str
    output_root: Path
    tracker_name: str
    tracker_result_path: Path
    seqmap_path: Path
    tracks_source_path: Path


def export_sort_run_to_trackeval_layout(
    run_dir: str | Path,
    sequence_name: str,
    output_root: str | Path = "outputs/trackeval",
    tracker_name: str = "sort",
    seqmap_name: str = "MOT17-test",
    overwrite: bool = False,
) -> TrackEvalLayoutResult:
    """Copy a SORT run folder's tracks.txt into a TrackEval-friendly result layout."""
    run_path = Path(run_dir)
    if not run_path.exists() or not run_path.is_dir():
        raise FileNotFoundError(f"TrackEval export run_dir does not exist: {run_path}")

    normalized_sequence_name = sequence_name.strip()
    if not normalized_sequence_name:
        raise ValueError("TrackEval export sequence_name must not be empty.")

    tracks_source_path = run_path / "tracks.txt"
    if not tracks_source_path.exists():
        raise FileNotFoundError(f"TrackEval export requires tracks.txt: {tracks_source_path}")

    output_root_path = Path(output_root)
    layout_root = output_root_path / tracker_name / run_path.name
    tracker_data_dir = layout_root / "trackers" / tracker_name / "data"
    seqmap_dir = layout_root / "seqmaps"
    tracker_result_path = tracker_data_dir / f"{normalized_sequence_name}.txt"
    seqmap_path = seqmap_dir / f"{seqmap_name}.txt"

    _ensure_writable(tracker_result_path, overwrite=overwrite)
    _ensure_writable(seqmap_path, overwrite=overwrite)

    tracker_data_dir.mkdir(parents=True, exist_ok=True)
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tracks_source_path, tracker_result_path)
    seqmap_path.write_text(f"name\n{normalized_sequence_name}\n", encoding="utf-8")

    return TrackEvalLayoutResult(
        run_dir=run_path,
        sequence_name=normalized_sequence_name,
        output_root=output_root_path,
        tracker_name=tracker_name,
        tracker_result_path=tracker_result_path,
        seqmap_path=seqmap_path,
        tracks_source_path=tracks_source_path,
    )


def _ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing TrackEval export file: {path}")
