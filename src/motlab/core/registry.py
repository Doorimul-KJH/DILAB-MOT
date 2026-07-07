"""Registry for paper preset YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from motlab.core.config import find_project_root, load_paper_config


class PaperPresetRegistry:
    """Discover and load paper presets from ``configs/papers``."""

    def __init__(self, project_root: str | Path | None = None, config_dir: str | Path | None = None):
        self.project_root = Path(project_root).resolve() if project_root else find_project_root()
        self.config_dir = Path(config_dir).resolve() if config_dir else self.project_root / "configs" / "papers"

    def list_papers(self) -> list[str]:
        """Return available paper IDs sorted alphabetically."""
        return sorted(self._load_all_presets().keys())

    def load_paper(self, paper_id: str) -> dict[str, Any]:
        """Load one paper preset by ID."""
        presets = self._load_all_presets()
        if paper_id not in presets:
            available = ", ".join(sorted(presets)) or "none"
            raise ValueError(f"Unknown paper_id '{paper_id}'. Available papers: {available}")
        return presets[paper_id]

    def _load_all_presets(self) -> dict[str, dict[str, Any]]:
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Paper preset directory does not exist: {self.config_dir}")

        presets: dict[str, dict[str, Any]] = {}
        for path in sorted([*self.config_dir.glob("*.yaml"), *self.config_dir.glob("*.yml")]):
            config = load_paper_config(path)
            paper_id = str(config["paper_id"])
            if paper_id in presets:
                raise ValueError(f"Duplicate paper_id '{paper_id}' found in {self.config_dir}")
            presets[paper_id] = config
        return presets
