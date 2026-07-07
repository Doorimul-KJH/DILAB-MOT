"""Single-track lifecycle state for SORT."""

from __future__ import annotations

from dataclasses import dataclass, field

from motlab.core.types import BoundingBoxTLWH
from motlab.trackers.sort.kalman import KalmanBoxTracker


@dataclass
class SortTrack:
    """SORT single-track lifecycle wrapper.

    This class manages one track's counters and Kalman motion model. It does
    not implement multi-object association, max_age, or min_hits policies.
    """

    track_id: int
    initial_bbox: BoundingBoxTLWH
    age: int = 0
    hits: int = 1
    hit_streak: int = 1
    time_since_update: int = 0
    kalman: KalmanBoxTracker = field(init=False)

    def __post_init__(self) -> None:
        if self.track_id < 1:
            raise ValueError("SortTrack track_id must be greater than or equal to 1.")
        self.kalman = KalmanBoxTracker(self.initial_bbox)

    def predict(self) -> BoundingBoxTLWH:
        """Predict this track one frame forward."""
        if self.time_since_update > 0:
            self.hit_streak = 0

        predicted_bbox = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        return predicted_bbox

    def update(self, bbox: BoundingBoxTLWH) -> BoundingBoxTLWH:
        """Update this track from an associated detection bbox."""
        updated_bbox = self.kalman.update(bbox)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        return updated_bbox

    def get_state(self) -> BoundingBoxTLWH:
        """Return this track's current bbox estimate."""
        return self.kalman.get_state()
