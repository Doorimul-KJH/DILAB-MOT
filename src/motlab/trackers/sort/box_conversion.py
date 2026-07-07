"""Bounding-box conversion utilities for SORT."""

from __future__ import annotations

import math

import numpy as np

from motlab.core.types import BoundingBoxTLWH


def tlwh_to_z(bbox: BoundingBoxTLWH) -> np.ndarray:
    """Convert tlwh bbox to SORT measurement [cx, cy, s, r]."""
    center_x = bbox.left + bbox.width / 2.0
    center_y = bbox.top + bbox.height / 2.0
    scale = bbox.width * bbox.height
    aspect_ratio = bbox.width / bbox.height
    return np.array([[center_x], [center_y], [scale], [aspect_ratio]], dtype=float)


def z_to_tlwh(z: np.ndarray) -> BoundingBoxTLWH:
    """Convert SORT measurement [cx, cy, s, r] to tlwh bbox."""
    values = np.asarray(z, dtype=float).reshape(-1)
    if values.size < 4:
        raise ValueError("z must contain at least [cx, cy, s, r].")

    center_x, center_y, scale, aspect_ratio = values[:4]
    if scale <= 0:
        raise ValueError("SORT bbox scale s must be positive.")
    if aspect_ratio <= 0:
        raise ValueError("SORT bbox aspect ratio r must be positive.")

    width = math.sqrt(scale * aspect_ratio)
    height = scale / width
    left = center_x - width / 2.0
    top = center_y - height / 2.0
    return BoundingBoxTLWH(left=left, top=top, width=width, height=height)


def x_to_tlwh(x: np.ndarray) -> BoundingBoxTLWH:
    """Convert SORT Kalman state [cx, cy, s, r, ...] to tlwh bbox."""
    return z_to_tlwh(np.asarray(x, dtype=float).reshape(-1)[:4])
