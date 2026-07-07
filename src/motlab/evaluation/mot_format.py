"""MOTChallenge tracking result format utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from motlab.core.types import TrackResult


def write_mot_tracking_results(
    results: Iterable[TrackResult],
    output_path: str | Path,
    decimal_places: int = 2,
) -> None:
    """Write tracker results in MOTChallenge text format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(results, key=lambda result: (result.frame, result.track_id))
    lines = [_format_track_result(result, decimal_places=decimal_places) for result in sorted_results]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _format_track_result(result: TrackResult, decimal_places: int) -> str:
    bbox = result.bbox
    float_format = f"{{:.{decimal_places}f}}"
    return ",".join(
        [
            str(result.frame),
            str(result.track_id),
            float_format.format(bbox.left),
            float_format.format(bbox.top),
            float_format.format(bbox.width),
            float_format.format(bbox.height),
            float_format.format(result.confidence),
            "-1",
            "-1",
            "-1",
        ]
    )
