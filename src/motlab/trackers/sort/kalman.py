"""Kalman motion model wrapper for SORT bounding boxes."""

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter

from motlab.core.types import BoundingBoxTLWH
from motlab.trackers.sort.box_conversion import tlwh_to_z, x_to_tlwh


class KalmanBoxTracker:
    """Single-track SORT Kalman motion model.

    This wrapper only handles motion prediction and measurement update for one
    bounding box. It does not implement SORT track lifecycle management.
    """

    def __init__(self, bbox: BoundingBoxTLWH):
        self.filter = KalmanFilter(dim_x=7, dim_z=4)
        self.filter.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.filter.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )

        self.filter.R[2:, 2:] *= 10.0
        self.filter.P[4:, 4:] *= 1000.0
        self.filter.P *= 10.0
        self.filter.Q[-1, -1] *= 0.01
        self.filter.Q[4:, 4:] *= 0.01

        self.filter.x[:4] = tlwh_to_z(bbox)

    def predict(self) -> BoundingBoxTLWH:
        """Predict one time step and return the predicted bbox."""
        if self.filter.x[2, 0] + self.filter.x[6, 0] <= 0:
            self.filter.x[6, 0] = 0.0
        self.filter.predict()
        return x_to_tlwh(self.filter.x)

    def update(self, bbox: BoundingBoxTLWH) -> BoundingBoxTLWH:
        """Update the motion model from a detection bbox."""
        self.filter.update(tlwh_to_z(bbox))
        return self.get_state()

    def get_state(self) -> BoundingBoxTLWH:
        """Return the current bbox estimate."""
        return x_to_tlwh(self.filter.x)
