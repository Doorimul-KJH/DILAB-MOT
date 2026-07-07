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

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("BoundingBoxTLWH width must be greater than 0.")
        if self.height <= 0:
            raise ValueError("BoundingBoxTLWH height must be greater than 0.")


@dataclass(frozen=True)
class Detection:
    """One public detection row for a 1-based frame index."""

    frame: int
    bbox: BoundingBoxTLWH
    confidence: float
    class_id: int | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.frame < 1:
            raise ValueError("Detection frame must be 1-based and greater than or equal to 1.")


@dataclass(frozen=True)
class TrackResult:
    """One tracker output row for a 1-based frame index."""

    frame: int
    track_id: int
    bbox: BoundingBoxTLWH
    confidence: float

    def __post_init__(self) -> None:
        if self.frame < 1:
            raise ValueError("TrackResult frame must be 1-based and greater than or equal to 1.")
        if self.track_id < 1:
            raise ValueError("TrackResult track_id must be greater than or equal to 1.")
