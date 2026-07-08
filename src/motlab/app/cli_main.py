"""Command-line interface for DILAB-MOT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from motlab.core.experiment_runner import ExperimentRunner
from motlab.core.registry import PaperPresetRegistry
from motlab.evaluation.trackeval_layout import export_sort_run_to_trackeval_layout
from motlab.evaluation.trackeval_runner import (
    build_trackeval_mot_command,
    check_trackeval_available,
)
from motlab.evaluation.trackeval_setup import prepare_trackeval
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

    trackeval_parser = subparsers.add_parser(
        "export-trackeval-layout",
        help="Export a SORT run folder into a TrackEval-friendly result layout.",
    )
    trackeval_parser.add_argument("--run-dir", required=True, help="SORT MOT run folder path.")
    trackeval_parser.add_argument("--sequence-name", required=True, help="MOT sequence name.")
    trackeval_parser.add_argument("--output-root", default="outputs/trackeval")
    trackeval_parser.add_argument("--tracker-name", default="sort")
    trackeval_parser.add_argument("--seqmap-name", default="MOT17-test")
    trackeval_parser.add_argument("--overwrite", action="store_true")

    check_trackeval_parser = subparsers.add_parser(
        "check-trackeval",
        help="Check whether a local TrackEval checkout is available.",
    )
    check_trackeval_parser.add_argument("--trackeval-root", default="third_party/TrackEval")

    build_trackeval_parser = subparsers.add_parser(
        "build-trackeval-command",
        help="Build a TrackEval MOTChallenge command without executing it.",
    )
    build_trackeval_parser.add_argument("--trackeval-root", required=True)
    build_trackeval_parser.add_argument("--gt-folder", required=True)
    build_trackeval_parser.add_argument("--trackers-folder", required=True)
    build_trackeval_parser.add_argument("--seqmap-file", required=True)
    build_trackeval_parser.add_argument("--tracker-name", default="sort")

    prepare_trackeval_parser = subparsers.add_parser(
        "prepare-trackeval",
        help="Explicitly prepare a local TrackEval checkout.",
    )
    prepare_trackeval_parser.add_argument("--trackeval-root", default="third_party/TrackEval")
    prepare_trackeval_parser.add_argument(
        "--repo-url",
        default="https://github.com/JonathonLuiten/TrackEval.git",
    )
    prepare_trackeval_parser.add_argument("--ref", default="master")
    prepare_trackeval_parser.add_argument("--clone", action="store_true")

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
    if args.command == "export-trackeval-layout":
        return _handle_export_trackeval_layout(
            run_dir=args.run_dir,
            sequence_name=args.sequence_name,
            output_root=args.output_root,
            tracker_name=args.tracker_name,
            seqmap_name=args.seqmap_name,
            overwrite=args.overwrite,
        )
    if args.command == "check-trackeval":
        return _handle_check_trackeval(trackeval_root=args.trackeval_root)
    if args.command == "build-trackeval-command":
        return _handle_build_trackeval_command(
            trackeval_root=args.trackeval_root,
            gt_folder=args.gt_folder,
            trackers_folder=args.trackers_folder,
            seqmap_file=args.seqmap_file,
            tracker_name=args.tracker_name,
        )
    if args.command == "prepare-trackeval":
        return _handle_prepare_trackeval(
            trackeval_root=args.trackeval_root,
            repo_url=args.repo_url,
            ref=args.ref,
            clone=args.clone,
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


def _handle_export_trackeval_layout(
    run_dir: str,
    sequence_name: str,
    output_root: str,
    tracker_name: str,
    seqmap_name: str,
    overwrite: bool,
) -> int:
    result = export_sort_run_to_trackeval_layout(
        run_dir=run_dir,
        sequence_name=sequence_name,
        output_root=output_root,
        tracker_name=tracker_name,
        seqmap_name=seqmap_name,
        overwrite=overwrite,
    )

    print("TrackEval layout export completed. TrackEval was not executed.")
    print(f"Tracker result path: {result.tracker_result_path}")
    print(f"Seqmap path: {result.seqmap_path}")
    print(f"Tracks source path: {result.tracks_source_path}")
    return 0


def _handle_check_trackeval(trackeval_root: str) -> int:
    result = check_trackeval_available(trackeval_root=trackeval_root)

    print(_format_trackeval_status_message(exists=result.exists, can_import_or_help=result.can_import_or_help))
    print(f"trackeval_root: {result.trackeval_root}")
    print(f"exists: {result.exists}")
    print(f"script_path: {result.script_path}")
    print(f"can_import_or_help: {result.can_import_or_help}")
    print(f"message: {result.message}")
    return 0


def _handle_build_trackeval_command(
    trackeval_root: str,
    gt_folder: str,
    trackers_folder: str,
    seqmap_file: str,
    tracker_name: str,
) -> int:
    command = build_trackeval_mot_command(
        trackeval_root=trackeval_root,
        gt_folder=gt_folder,
        trackers_folder=trackers_folder,
        seqmap_file=seqmap_file,
        tracker_name=tracker_name,
    )

    print("TrackEval command, readable:")
    for line in _format_command_readable(command):
        print(line)
    print()
    print("TrackEval command, one-line:")
    print(" ".join(command))
    return 0


def _handle_prepare_trackeval(
    trackeval_root: str,
    repo_url: str,
    ref: str,
    clone: bool,
) -> int:
    result = prepare_trackeval(
        trackeval_root=trackeval_root,
        repo_url=repo_url,
        ref=ref,
        clone=clone,
    )

    print("TrackEval preparation completed.")
    print(f"cloned: {result.cloned}")
    print(f"already_exists: {result.already_exists}")
    print(f"trackeval_root: {result.trackeval_root}")
    print(f"repo_url: {result.repo_url}")
    print(f"ref: {result.ref}")
    print(f"commit_hash: {result.commit_hash}")
    print(f"setup_info_path: {result.setup_info_path}")
    print(f"message: {result.message}")
    print("Next check command:")
    print(f"python -m motlab.app.cli_main check-trackeval --trackeval-root {result.trackeval_root}")
    return 0


def _format_trackeval_status_message(exists: bool, can_import_or_help: bool) -> str:
    if not exists:
        return "TrackEval availability check completed. TrackEval root was not found."
    if can_import_or_help:
        return "TrackEval availability check completed. TrackEval appears available."
    return (
        "TrackEval availability check completed. TrackEval was found, "
        "but help/import check failed."
    )


def _format_command_readable(command: list[str]) -> list[str]:
    if len(command) < 2:
        return [f"  {part}" for part in command]

    lines = [f"  {command[0]}", f"  {command[1]}"]
    index = 2
    while index < len(command):
        option = command[index]
        value = command[index + 1] if index + 1 < len(command) else ""
        lines.append(f"  {option}")
        if value:
            lines.append(f"    {value}")
        index += 2
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
