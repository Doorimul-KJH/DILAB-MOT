from pathlib import Path

from motlab.core.config import load_paper_config


def test_load_sort_paper_config_validates_paper_principles():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "papers" / "sort.yaml"

    config = load_paper_config(config_path)

    assert config["paper_id"] == "sort"
    assert config["pipeline"]["detector"] == "faster_rcnn_or_mot_public_detection"
    assert any("must not use YOLO" in note for note in config["notes"])
