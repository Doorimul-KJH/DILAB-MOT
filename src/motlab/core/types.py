"""Shared MOT data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBoxTLWH:
    """Bounding box in MOTChallenge tlwh format."""

    left: float
    top: float
    width: float
    height: float


@dataclass(frozen=True)
class Detection:
    """One public detection row for a 1-based frame index."""

    frame: int
    bbox: BoundingBoxTLWH
    confidence: float
    class_id: int | None = None
    label: str | None = None


@dataclass(frozen=True)
class TrackResult:
    """One tracker output row for a 1-based frame index."""

    frame: int
    track_id: int
    bbox: BoundingBoxTLWH
    confidence: float
