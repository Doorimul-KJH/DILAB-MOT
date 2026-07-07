"""Dry-run experiment runner for paper presets."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from motlab.core.config import find_project_root
from motlab.core.registry import PaperPresetRegistry


DRY_RUN_MESSAGE = "Dry run only. Tracking is not implemented yet."


@dataclass(frozen=True)
class ExperimentRunResult:
    """Result metadata returned by an experiment run."""

    output_dir: Path
    paper_config: dict[str, Any]
    environment: dict[str, str]
    manifest: dict[str, Any]


class ExperimentRunner:
    """Create run folders and metadata without executing tracking yet."""

    def __init__(self, project_root: str | Path | None = None, output_root: str | Path | None = None):
        self.project_root = Path(project_root).resolve() if project_root else find_project_root()
        self.output_root = Path(output_root).resolve() if output_root else self.project_root / "outputs" / "runs"
        self.registry = PaperPresetRegistry(project_root=self.project_root)

    def run(self, paper_id: str, dry_run: bool = True) -> ExperimentRunResult:
        """Run a paper preset in dry-run mode only."""
        if not dry_run:
            raise ValueError("Only dry-run mode is supported. Tracking is not implemented yet.")

        paper_config = self.registry.load_paper(paper_id)
        created_at = datetime.now().astimezone()
        output_dir = self._create_output_dir(paper_id=paper_id, created_at=created_at)
        environment = self._collect_environment()
        manifest = self._build_manifest(
            paper_config=paper_config,
            created_at=created_at,
            output_dir=output_dir,
        )

        self._write_yaml(output_dir / "paper_config.yaml", paper_config)
        self._write_json(output_dir / "environment.json", environment)
        self._write_json(output_dir / "run_manifest.json", manifest)

        return ExperimentRunResult(
            output_dir=output_dir,
            paper_config=paper_config,
            environment=environment,
            manifest=manifest,
        )

    def _create_output_dir(self, paper_id: str, created_at: datetime) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        base_name = f"{created_at.strftime('%Y%m%d_%H%M%S')}_{paper_id}_dry_run"
        output_dir = self.output_root / base_name
        suffix = 1
        while output_dir.exists():
            output_dir = self.output_root / f"{base_name}_{suffix}"
            suffix += 1
        output_dir.mkdir(parents=True)
        return output_dir

    def _collect_environment(self) -> dict[str, str]:
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "current_working_directory": str(Path.cwd()),
        }

    def _build_manifest(
        self,
        paper_config: dict[str, Any],
        created_at: datetime,
        output_dir: Path,
    ) -> dict[str, Any]:
        return {
            "paper_id": paper_config["paper_id"],
            "paper_name": paper_config["paper_name"],
            "mode": paper_config["mode"],
            "dry_run": True,
            "created_at": created_at.isoformat(),
            "project_root": str(self.project_root),
            "output_dir": str(output_dir),
            "message": DRY_RUN_MESSAGE,
        }

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _write_yaml(self, path: Path, data: dict[str, Any]) -> None:
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
