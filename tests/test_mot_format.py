from motlab.core.types import BoundingBoxTLWH, TrackResult
from motlab.evaluation.mot_format import write_mot_tracking_results


def test_write_mot_tracking_results_sorts_and_formats_rows(tmp_path):
    output_path = tmp_path / "results" / "sort.txt"
    results = [
        TrackResult(
            frame=2,
            track_id=1,
            bbox=BoundingBoxTLWH(left=12, top=22, width=30, height=40),
            confidence=0.8,
        ),
        TrackResult(
            frame=1,
            track_id=2,
            bbox=BoundingBoxTLWH(left=100, top=120, width=30, height=40),
            confidence=0.75,
        ),
        TrackResult(
            frame=1,
            track_id=1,
            bbox=BoundingBoxTLWH(left=10, top=20, width=30, height=40),
            confidence=0.9,
        ),
    ]

    write_mot_tracking_results(results, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert lines == [
        "1,1,10.00,20.00,30.00,40.00,0.90,-1,-1,-1",
        "1,2,100.00,120.00,30.00,40.00,0.75,-1,-1,-1",
        "2,1,12.00,22.00,30.00,40.00,0.80,-1,-1,-1",
    ]


def test_written_mot_tracking_result_values_are_readable_as_csv(tmp_path):
    output_path = tmp_path / "track.txt"
    result = TrackResult(
        frame=3,
        track_id=4,
        bbox=BoundingBoxTLWH(left=1.234, top=2.345, width=3.456, height=4.567),
        confidence=0.987,
    )

    write_mot_tracking_results([result], output_path)

    row = output_path.read_text(encoding="utf-8").strip().split(",")
    assert row[:2] == ["3", "4"]
    assert [float(value) for value in row[2:7]] == [1.23, 2.35, 3.46, 4.57, 0.99]
    assert row[7:] == ["-1", "-1", "-1"]
