import numpy as np
import pytest

from motlab.core.types import BoundingBoxTLWH
from motlab.trackers.sort.box_conversion import tlwh_to_z, x_to_tlwh, z_to_tlwh


def assert_bbox_close(actual: BoundingBoxTLWH, expected: BoundingBoxTLWH) -> None:
    assert actual.left == pytest.approx(expected.left)
    assert actual.top == pytest.approx(expected.top)
    assert actual.width == pytest.approx(expected.width)
    assert actual.height == pytest.approx(expected.height)


def test_tlwh_to_z_returns_center_area_and_aspect_ratio_column_vector():
    bbox = BoundingBoxTLWH(left=10, top=20, width=30, height=40)

    z = tlwh_to_z(bbox)

    assert z.shape == (4, 1)
    np.testing.assert_allclose(z, np.array([[25.0], [40.0], [1200.0], [0.75]]))


def test_z_to_tlwh_restores_original_bbox():
    bbox = BoundingBoxTLWH(left=10, top=20, width=30, height=40)

    restored = z_to_tlwh(tlwh_to_z(bbox))

    assert_bbox_close(restored, bbox)


def test_x_to_tlwh_accepts_7d_kalman_state():
    x = np.array([[25.0], [40.0], [1200.0], [0.75], [1.0], [2.0], [3.0]])

    bbox = x_to_tlwh(x)

    assert_bbox_close(bbox, BoundingBoxTLWH(left=10, top=20, width=30, height=40))


@pytest.mark.parametrize("z", [np.array([[25], [40], [0], [0.75]]), np.array([[25], [40], [1200], [0]])])
def test_z_to_tlwh_rejects_non_positive_scale_or_aspect_ratio(z):
    with pytest.raises(ValueError, match="positive"):
        z_to_tlwh(z)
