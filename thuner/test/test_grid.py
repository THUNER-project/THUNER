"""Unit tests for grid coordinate helpers."""

from types import SimpleNamespace
import numpy as np
import pytest

import thuner.grid as grid


def test_get_pixels_geographic_array():
    go = SimpleNamespace(name="geographic", latitude=[10.0, 11.0, 12.0, 13.0],
                         longitude=[100.0, 101.0, 102.0, 103.0, 104.0])
    lats, lons = grid.get_pixels_geographic(np.array([1, 3]), np.array([2, 4]), go)
    assert list(lats) == [11.0, 13.0]
    assert list(lons) == [102.0, 104.0]


def test_get_pixels_geographic_scalar_returns_scalars():
    go = SimpleNamespace(name="geographic", latitude=[10.0, 11.0, 12.0],
                         longitude=[100.0, 101.0, 102.0])
    lat, lon = grid.get_pixels_geographic(2, 1, go)
    assert (lat, lon) == (12.0, 101.0)
    assert np.ndim(lat) == 0 and np.ndim(lon) == 0


def test_get_pixels_geographic_cartesian_uses_2d_coords():
    lat2d = np.arange(12).reshape(3, 4).astype(float)
    lon2d = lat2d + 100
    go = SimpleNamespace(name="cartesian", latitude=lat2d, longitude=lon2d)
    lats, lons = grid.get_pixels_geographic(np.array([0, 2]), np.array([1, 3]), go)
    assert list(lats) == [1.0, 11.0]
    assert list(lons) == [101.0, 111.0]


def test_get_pixels_geographic_shape_mismatch_raises():
    go = SimpleNamespace(name="geographic", latitude=[0.0, 1.0], longitude=[0.0, 1.0])
    with pytest.raises(ValueError):
        grid.get_pixels_geographic(np.array([0, 1]), np.array([0]), go)
