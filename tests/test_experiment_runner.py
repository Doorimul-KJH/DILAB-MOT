import json
from pathlib import Path

import yaml

from motlab.core.experiment_runner import ExperimentRunner


def test_dry_run_creates_manifest_and_config_snapshot(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    runner = ExperimentRunner(project_root=project_root, output_root=tmp_path)

    result = runner.run(paper_id="sort", dry_run=True)

    assert result.output_dir.exists()
    assert result.output_dir.name.endswith("_sort_dry_run")

    paper_config_path = result.output_dir / "paper_config.yaml"
    environment_path = result.output_dir / "environment.json"
    manifest_path = result.output_dir / "run_manifest.json"

    assert paper_config_path.exists()
    assert environment_path.exists()
    assert manifest_path.exists()

    paper_config = yaml.safe_load(paper_config_path.read_text(encoding="utf-8"))
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert paper_config["paper_id"] == "sort"
    assert paper_config["pipeline"]["detector"] == "faster_rcnn_or_mot_public_detection"
    assert environment["current_working_directory"]
    assert manifest["paper_id"] == "sort"
    assert manifest["dry_run"] is True
    assert manifest["message"] == "Dry run only. Tracking is not implemented yet."
