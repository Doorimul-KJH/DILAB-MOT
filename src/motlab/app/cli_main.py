"""Command-line interface for DILAB-MOT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from motlab.core.experiment_runner import ExperimentRunner
from motlab.core.registry import PaperPresetRegistry


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
        return _handle_run(args.paper, dry_run=args.dry_run)

    parser.error(f"Unknown command: {args.command}")
    return 0


def _handle_list_papers() -> int:
    registry = PaperPresetRegistry()
    print("사용 가능한 paper preset:")
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


def _handle_run(paper_id: str, dry_run: bool) -> int:
    if not dry_run:
        print("오류: 현재는 --dry-run만 지원합니다. 실제 tracking은 아직 구현하지 않았습니다.")
        return 2

    runner = ExperimentRunner()
    result = runner.run(paper_id=paper_id, dry_run=True)

    print("Dry-run 실행 완료. 실제 tracking은 아직 수행하지 않았습니다.")
    print(f"Output folder: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
