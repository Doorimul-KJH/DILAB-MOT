"""SORT pipeline for MOTChallenge public detections."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from motlab.core.config import find_project_root
from motlab.datasets.motchallenge import (
    MOTChallengeSequenceInfo,
    load_motchallenge_sequence_info,
)
from motlab.detectors.mot_public_detection import load_mot_public_detections
from motlab.evaluation.mot_format import write_mot_tracking_results
from motlab.trackers.sort.tracker import SortTracker


@dataclass(frozen=True)
class SortPipelineResult:
    """Summary of one SORT MOT public detection pipeline run."""

    detection_path: Path
    output_path: Path
    frame_count: int
    input_detection_count: int
    output_track_count: int
    max_frame: int
    tracker_config: dict[str, float | int]


@dataclass(frozen=True)
class SortExperimentResult:
    """Paths and summary returned by a SORT MOT run folder execution."""

    run_id: str
    output_dir: Path
    tracks_path: Path
    manifest_path: Path
    environment_path: Path
    paper_config_path: Path
    pipeline_result: SortPipelineResult


@dataclass(frozen=True)
class SortSequenceRunResult:
    """Result returned by running SORT on one MOTChallenge sequence directory."""

    sequence_info: MOTChallengeSequenceInfo
    experiment_result: SortExperimentResult


def run_sort_on_mot_detections(
    detection_path: str | Path,
    output_path: str | Path,
    min_confidence: float = 0.0,
    max_age: int = 1,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    max_frame: int | None = None,
) -> SortPipelineResult:
    """Run frame-level SORT on MOTChallenge public detection rows."""
    detection_file = Path(detection_path)
    output_file = Path(output_path)

    all_detections_by_frame = load_mot_public_detections(
        detection_file,
        min_confidence=float("-inf"),
    )
    if not all_detections_by_frame:
        raise ValueError(f"MOT public detection file is empty: {detection_file}")

    last_detection_frame = max(all_detections_by_frame)
    requested_max_frame = last_detection_frame if max_frame is None else max_frame
    if requested_max_frame < last_detection_frame:
        raise ValueError(
            "max_frame must be greater than or equal to the last detection frame "
            f"({last_detection_frame})."
        )

    detections_by_frame = load_mot_public_detections(
        detection_file,
        min_confidence=min_confidence,
    )
    input_detection_count = sum(len(detections) for detections in detections_by_frame.values())

    tracker = SortTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    track_results = []
    for frame in range(1, requested_max_frame + 1):
        track_results.extend(tracker.update(detections_by_frame.get(frame, [])))

    write_mot_tracking_results(track_results, output_file)

    return SortPipelineResult(
        detection_path=detection_file,
        output_path=output_file,
        frame_count=requested_max_frame,
        input_detection_count=input_detection_count,
        output_track_count=len(track_results),
        max_frame=requested_max_frame,
        tracker_config={
            "max_age": max_age,
            "min_hits": min_hits,
            "iou_threshold": iou_threshold,
            "min_confidence": min_confidence,
        },
    )


def run_sort_mot_experiment(
    detection_path: str | Path,
    output_root: str | Path = "outputs/runs",
    min_confidence: float = 0.0,
    max_age: int = 1,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    max_frame: int | None = None,
) -> SortExperimentResult:
    """Run SORT on MOT public detections and store outputs in a run folder."""
    created_at = datetime.now().astimezone()
    output_dir = _create_run_output_dir(Path(output_root), created_at=created_at)
    run_id = output_dir.name
    tracks_path = output_dir / "tracks.txt"
    paper_config_path = output_dir / "paper_config.yaml"
    environment_path = output_dir / "environment.json"
    manifest_path = output_dir / "run_manifest.json"

    project_root = find_project_root()
    shutil.copyfile(project_root / "configs" / "papers" / "sort.yaml", paper_config_path)
    _write_json(environment_path, _collect_environment())

    pipeline_result = run_sort_on_mot_detections(
        detection_path=detection_path,
        output_path=tracks_path,
        min_confidence=min_confidence,
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        max_frame=max_frame,
    )
    manifest = _build_run_manifest(
        run_id=run_id,
        created_at=created_at,
        detection_path=pipeline_result.detection_path,
        output_dir=output_dir,
        tracks_path=tracks_path,
        min_confidence=min_confidence,
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        max_frame=max_frame,
        pipeline_result=pipeline_result,
    )
    _write_json(manifest_path, manifest)

    return SortExperimentResult(
        run_id=run_id,
        output_dir=output_dir,
        tracks_path=tracks_path,
        manifest_path=manifest_path,
        environment_path=environment_path,
        paper_config_path=paper_config_path,
        pipeline_result=pipeline_result,
    )


def run_sort_on_mot_sequence(
    sequence_dir: str | Path,
    output_root: str | Path = "outputs/runs",
    min_confidence: float = 0.0,
    max_age: int = 1,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
) -> SortSequenceRunResult:
    """Run SORT on the public detections inside one MOTChallenge sequence directory."""
    sequence_info = load_motchallenge_sequence_info(
        sequence_dir,
        require_detection=True,
        require_gt=False,
    )
    experiment_result = run_sort_mot_experiment(
        detection_path=sequence_info.detection_path,
        output_root=output_root,
        min_confidence=min_confidence,
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        max_frame=sequence_info.seq_length,
    )
    _append_sequence_info_to_manifest(
        experiment_result.manifest_path,
        sequence_info=sequence_info,
    )

    return SortSequenceRunResult(
        sequence_info=sequence_info,
        experiment_result=experiment_result,
    )


def _create_run_output_dir(output_root: Path, created_at: datetime) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{created_at.strftime('%Y%m%d_%H%M%S_%f')}_sort_mot"
    output_dir = output_root / run_id
    suffix = 1
    while output_dir.exists():
        output_dir = output_root / f"{run_id}_{suffix}"
        suffix += 1
    output_dir.mkdir(parents=True)
    return output_dir


def _collect_environment() -> dict[str, str]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "current_working_directory": str(Path.cwd()),
    }


def _build_run_manifest(
    run_id: str,
    created_at: datetime,
    detection_path: Path,
    output_dir: Path,
    tracks_path: Path,
    min_confidence: float,
    max_age: int,
    min_hits: int,
    iou_threshold: float,
    max_frame: int | None,
    pipeline_result: SortPipelineResult,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "paper_id": "sort",
        "mode": "paper_reproduction",
        "input_type": "mot_public_detections",
        "detection_path": str(detection_path),
        "output_dir": str(output_dir),
        "tracks_path": str(tracks_path),
        "dry_run": False,
        "created_at": created_at.isoformat(),
        "tracker_config": {
            "min_confidence": min_confidence,
            "max_age": max_age,
            "min_hits": min_hits,
            "iou_threshold": iou_threshold,
            "max_frame": max_frame,
        },
        "result_summary": {
            "frame_count": pipeline_result.frame_count,
            "input_detection_count": pipeline_result.input_detection_count,
            "output_track_count": pipeline_result.output_track_count,
        },
        "message": "SORT MOT public detection run completed.",
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_sequence_info_to_manifest(
    manifest_path: Path,
    sequence_info: MOTChallengeSequenceInfo,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "sequence_name": sequence_info.name,
            "sequence_dir": str(sequence_info.sequence_dir),
            "seq_length": sequence_info.seq_length,
            "frame_rate": sequence_info.frame_rate,
            "image_width": sequence_info.image_width,
            "image_height": sequence_info.image_height,
        }
    )
    _write_json(manifest_path, manifest)
