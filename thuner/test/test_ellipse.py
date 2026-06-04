"""Unit tests for ellipse fitting.

The cartesian branch of ``cv2_ellipse`` is exercised at runtime only by the cpol
cartesian demo, which runs via ``parallel.track`` (spawn subprocesses) and so isn't
captured by coverage. These tests cover both grid types in-process.
"""

from types import SimpleNamespace
import numpy as np

from thuner.attribute import ellipse


def _blob_mask(shape=(40, 50), center=(20, 25), radius=9):
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    mask = np.zeros(shape, dtype=np.int32)
    mask[(rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= radius**2] = 1
    return mask


def test_cv2_ellipse_geographic():
    mask = _blob_mask()
    nlat, nlon = mask.shape
    grid_options = SimpleNamespace(
        name="geographic",
        latitude=list(np.linspace(-12.0, -10.0, nlat)),
        longitude=list(np.linspace(130.0, 132.0, nlon)),
        geographic_spacing=[2.0 / (nlat - 1), 2.0 / (nlon - 1)],
    )
    lat, lon, major, minor, orientation, ecc = ellipse.cv2_ellipse(mask, 1, grid_options)
    assert major >= minor > 0
    assert 0.0 <= ecc < 1.0
    assert 0.0 <= orientation < np.pi
    assert -12.0 <= lat <= -10.0 and 130.0 <= lon <= 132.0


def test_cv2_ellipse_cartesian():
    mask = _blob_mask()
    nlat, nlon = mask.shape
    lon2d, lat2d = np.meshgrid(
        np.linspace(130.0, 132.0, nlon), np.linspace(-12.0, -10.0, nlat)
    )
    grid_options = SimpleNamespace(
        name="cartesian",
        latitude=lat2d,
        longitude=lon2d,
        cartesian_spacing=[2500.0, 2500.0],
    )
    lat, lon, major, minor, orientation, ecc = ellipse.cv2_ellipse(mask, 1, grid_options)
    assert major >= minor > 0
    assert 0.0 <= ecc < 1.0
    assert 0.0 <= orientation < np.pi
    # A near-circular blob -> near-zero eccentricity.
    assert ecc < 0.5
