"""Command-line interface for DILAB-MOT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from motlab.core.experiment_runner import ExperimentRunner
from motlab.core.registry import PaperPresetRegistry
from motlab.pipelines.sort_mot_pipeline import (
    run_sort_mot_experiment,
    run_sort_on_mot_detections,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motlab",
        description="DILAB-MOT paper-based MOT reproduction CLI.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list-papers", help="List available paper presets.")

    inspect_parser = subparsers.add_parser("inspect-paper", help="Inspect one paper preset.")
    inspect_parser.add_argument("paper_id", help="Paper preset ID, for example: sort")

    run_parser = subparsers.add_parser("run", help="Create a dry-run experiment folder.")
    run_parser.add_argument("--paper", required=True, help="Paper preset ID, for example: sort")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create metadata only. Tracking is not implemented yet.",
    )
    run_parser.add_argument(
        "--output-root",
        help="Directory where dry-run folders are created. Defaults to outputs/runs.",
    )

    sort_mot_parser = subparsers.add_parser(
        "run-sort-mot",
        help="Run SORT on MOTChallenge public detection txt.",
    )
    sort_mot_parser.add_argument("--detections", required=True, help="MOTChallenge det.txt path.")
    sort_mot_parser.add_argument("--output", help="Output tracking result txt path.")
    sort_mot_parser.add_argument(
        "--output-root",
        default="outputs/runs",
        help="Directory where SORT MOT run folders are created.",
    )
    sort_mot_parser.add_argument(
        "--as-run-folder",
        action="store_true",
        help="Write tracks and metadata into a timestamped run folder.",
    )
    sort_mot_parser.add_argument("--min-confidence", type=float, default=0.0)
    sort_mot_parser.add_argument("--max-age", type=int, default=1)
    sort_mot_parser.add_argument("--min-hits", type=int, default=3)
    sort_mot_parser.add_argument("--iou-threshold", type=float, default=0.3)
    sort_mot_parser.add_argument("--max-frame", type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the DILAB-MOT CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        print("DILAB-MOT project skeleton is ready.")
        return 0

    if args.command == "list-papers":
        return _handle_list_papers()
    if args.command == "inspect-paper":
        return _handle_inspect_paper(args.paper_id)
    if args.command == "run":
        return _handle_run(args.paper, dry_run=args.dry_run, output_root=args.output_root)
    if args.command == "run-sort-mot":
        if args.as_run_folder and args.output:
            parser.error("--output cannot be used with --as-run-folder")
        if not args.as_run_folder and not args.output:
            parser.error("--output is required unless --as-run-folder is used")
        return _handle_run_sort_mot(
            detection_path=args.detections,
            output_path=args.output,
            output_root=args.output_root,
            as_run_folder=args.as_run_folder,
            min_confidence=args.min_confidence,
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold,
            max_frame=args.max_frame,
        )

    parser.error(f"Unknown command: {args.command}")
    return 0


def _handle_list_papers() -> int:
    registry = PaperPresetRegistry()
    print("Available paper presets:")
    for paper_id in registry.list_papers():
        config = registry.load_paper(paper_id)
        print(f"- {paper_id}: {config['paper_name']} ({config['title']})")
    return 0


def _handle_inspect_paper(paper_id: str) -> int:
    registry = PaperPresetRegistry()
    config = registry.load_paper(paper_id)
    pipeline = config["pipeline"]

    print(f"paper_id: {config['paper_id']}")
    print(f"paper_name: {config['paper_name']}")
    print(f"title: {config['title']}")
    print(f"mode: {config['mode']}")
    print("pipeline:")
    print(f"  detector: {pipeline['detector']}")
    print(f"  tracker: {pipeline['tracker']}")
    print(f"  motion_model: {pipeline['motion_model']}")
    print(f"  association: {pipeline['association']}")
    print(f"  matching_cost: {pipeline['matching_cost']}")
    print("notes:")
    for note in config.get("notes", []):
        print(f"  - {note}")
    return 0


def _handle_run(paper_id: str, dry_run: bool, output_root: str | None = None) -> int:
    if not dry_run:
        print("Error: only --dry-run is supported. Tracking is not implemented yet.")
        return 2

    runner = ExperimentRunner(output_root=output_root)
    result = runner.run(paper_id=paper_id, dry_run=True)

    print("Dry-run completed. Tracking is not implemented yet.")
    print(f"Output folder: {result.output_dir}")
    return 0


def _handle_run_sort_mot(
    detection_path: str,
    output_path: str | None,
    output_root: str,
    as_run_folder: bool,
    min_confidence: float,
    max_age: int,
    min_hits: int,
    iou_threshold: float,
    max_frame: int | None,
) -> int:
    if as_run_folder:
        experiment_result = run_sort_mot_experiment(
            detection_path=detection_path,
            output_root=output_root,
            min_confidence=min_confidence,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            max_frame=max_frame,
        )
        result = experiment_result.pipeline_result

        print("SORT MOT public detection run completed.")
        print(f"Output folder: {experiment_result.output_dir}")
        print(f"Tracks file: {experiment_result.tracks_path}")
        print(f"Processed frames: {result.frame_count}")
        print(f"Input detections: {result.input_detection_count}")
        print(f"Output track rows: {result.output_track_count}")
        return 0

    if output_path is None:
        raise ValueError("output_path is required when as_run_folder is false.")

    result = run_sort_on_mot_detections(
        detection_path=detection_path,
        output_path=output_path,
        min_confidence=min_confidence,
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        max_frame=max_frame,
    )

    print("SORT MOT public detection run completed.")
    print(f"Output file: {result.output_path}")
    print(f"Processed frames: {result.frame_count}")
    print(f"Input detections: {result.input_detection_count}")
    print(f"Output track rows: {result.output_track_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
