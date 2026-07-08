import subprocess
import sys

import pytest

from motlab.evaluation.trackeval_layout import export_sort_run_to_trackeval_layout


TRACKS_CONTENT = "1,1,10.00,20.00,30.00,40.00,0.90,-1,-1,-1\n"


def make_run_dir(tmp_path):
    run_dir = tmp_path / "20260708_101500_123456_sort_mot"
    run_dir.mkdir()
    (run_dir / "tracks.txt").write_text(TRACKS_CONTENT, encoding="utf-8")
    return run_dir


def test_export_trackeval_layout_copies_tracks_and_writes_seqmap(tmp_path):
    run_dir = make_run_dir(tmp_path)

    result = export_sort_run_to_trackeval_layout(
        run_dir=run_dir,
        sequence_name="MOT17-02",
        output_root=tmp_path / "trackeval",
    )

    assert result.run_dir == run_dir
    assert result.sequence_name == "MOT17-02"
    assert result.tracker_name == "sort"
    assert result.tracks_source_path == run_dir / "tracks.txt"
    assert result.tracker_result_path == (
        tmp_path
        / "trackeval"
        / "sort"
        / run_dir.name
        / "trackers"
        / "sort"
        / "data"
        / "MOT17-02.txt"
    )
    assert result.seqmap_path == (
        tmp_path / "trackeval" / "sort" / run_dir.name / "seqmaps" / "MOT17-test.txt"
    )
    assert result.tracker_result_path.read_text(encoding="utf-8") == TRACKS_CONTENT
    assert result.seqmap_path.read_text(encoding="utf-8") == "name\nMOT17-02\n"


def test_export_trackeval_layout_rejects_missing_run_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_dir"):
        export_sort_run_to_trackeval_layout(
            run_dir=tmp_path / "missing",
            sequence_name="MOT17-02",
            output_root=tmp_path / "trackeval",
        )


def test_export_trackeval_layout_rejects_missing_tracks_file(tmp_path):
    run_dir = tmp_path / "run_without_tracks"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="tracks.txt"):
        export_sort_run_to_trackeval_layout(
            run_dir=run_dir,
            sequence_name="MOT17-02",
            output_root=tmp_path / "trackeval",
        )


def test_export_trackeval_layout_rejects_empty_sequence_name(tmp_path):
    run_dir = make_run_dir(tmp_path)

    with pytest.raises(ValueError, match="sequence_name"):
        export_sort_run_to_trackeval_layout(
            run_dir=run_dir,
            sequence_name=" ",
            output_root=tmp_path / "trackeval",
        )


def test_export_trackeval_layout_rejects_existing_files_without_overwrite(tmp_path):
    run_dir = make_run_dir(tmp_path)
    export_sort_run_to_trackeval_layout(
        run_dir=run_dir,
        sequence_name="MOT17-02",
        output_root=tmp_path / "trackeval",
    )

    with pytest.raises(FileExistsError, match="overwrite"):
        export_sort_run_to_trackeval_layout(
            run_dir=run_dir,
            sequence_name="MOT17-02",
            output_root=tmp_path / "trackeval",
        )


def test_export_trackeval_layout_overwrites_existing_files(tmp_path):
    run_dir = make_run_dir(tmp_path)
    first = export_sort_run_to_trackeval_layout(
        run_dir=run_dir,
        sequence_name="MOT17-02",
        output_root=tmp_path / "trackeval",
    )
    first.tracker_result_path.write_text("old\n", encoding="utf-8")

    second = export_sort_run_to_trackeval_layout(
        run_dir=run_dir,
        sequence_name="MOT17-02",
        output_root=tmp_path / "trackeval",
        overwrite=True,
    )

    assert second.tracker_result_path == first.tracker_result_path
    assert second.tracker_result_path.read_text(encoding="utf-8") == TRACKS_CONTENT


def test_cli_export_trackeval_layout_succeeds(tmp_path):
    run_dir = make_run_dir(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "motlab.app.cli_main",
            "export-trackeval-layout",
            "--run-dir",
            str(run_dir),
            "--sequence-name",
            "MOT17-02",
            "--output-root",
            str(tmp_path / "trackeval"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Tracker result path:" in completed.stdout
    assert "Seqmap path:" in completed.stdout
    assert "Tracks source path:" in completed.stdout
    assert (tmp_path / "trackeval" / "sort" / run_dir.name / "seqmaps" / "MOT17-test.txt").exists()
