"""Configuration loading and validation for paper presets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_PAPER_FIELDS = (
    "paper_id",
    "paper_name",
    "title",
    "mode",
    "pipeline",
    "evaluation",
)
SORT_DETECTOR = "faster_rcnn_or_mot_public_detection"


def load_paper_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a paper preset YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Paper config does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    validate_paper_config(config, source=path)
    return config


def validate_paper_config(config: Any, source: str | Path = "<memory>") -> None:
    """Validate fields required by the core paper-preset architecture."""
    if not isinstance(config, dict):
        raise ValueError(f"Paper config must be a mapping: {source}")

    missing_fields = [field for field in REQUIRED_PAPER_FIELDS if field not in config]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Paper config is missing required fields in {source}: {missing}")

    if not isinstance(config["pipeline"], dict):
        raise ValueError(f"Paper config pipeline must be a mapping: {source}")
    if not isinstance(config["evaluation"], dict):
        raise ValueError(f"Paper config evaluation must be a mapping: {source}")

    if config["paper_id"] == "sort":
        _validate_sort_preset(config, source)


def _validate_sort_preset(config: dict[str, Any], source: str | Path) -> None:
    detector = config["pipeline"].get("detector")
    if detector != SORT_DETECTOR:
        raise ValueError(
            "SORT paper preset must use detector "
            f"'{SORT_DETECTOR}', but got '{detector}' in {source}"
        )

    notes = config.get("notes", [])
    if not isinstance(notes, list) or not any("must not use YOLO" in str(note) for note in notes):
        raise ValueError(f"SORT paper preset must keep the no-YOLO note in {source}")


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the project root by walking upward until ``pyproject.toml`` exists."""
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").exists():
            return candidate

    raise FileNotFoundError(f"Could not find DILAB-MOT project root from: {current}")
