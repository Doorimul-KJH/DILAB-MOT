from pathlib import Path

import pytest

from motlab.core.registry import PaperPresetRegistry


def test_registry_finds_sort_preset():
    project_root = Path(__file__).resolve().parents[1]
    registry = PaperPresetRegistry(project_root)

    assert registry.list_papers() == ["sort"]
    assert registry.load_paper("sort")["paper_name"] == "SORT"


def test_registry_reports_unknown_paper_id():
    project_root = Path(__file__).resolve().parents[1]
    registry = PaperPresetRegistry(project_root)

    with pytest.raises(ValueError, match="Unknown paper_id 'missing'"):
        registry.load_paper("missing")
