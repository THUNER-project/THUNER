"""Unit tests for the detection preprocessing (grid flattening) functions."""

from types import SimpleNamespace
import numpy as np
import xarray as xr

import thuner.detect.preprocess as preprocess


def _grid():
    altitudes = [1000.0, 2000.0, 3000.0, 4000.0]
    values = np.arange(4 * 2 * 2, dtype=float).reshape(4, 2, 2)
    return xr.DataArray(
        values,
        dims=("altitude", "latitude", "longitude"),
        coords={"altitude": altitudes, "latitude": [0, 1], "longitude": [0, 1]},
    )


def _options(altitudes):
    return SimpleNamespace(detection=SimpleNamespace(altitudes=altitudes))


def test_vertical_max_over_range():
    grid = _grid()
    out = preprocess.vertical_max(grid, _options((1000.0, 3000.0)))
    expected = grid.sel(altitude=slice(1000.0, 3000.0)).max(dim="altitude")
    assert "altitude" not in out.dims
    assert np.array_equal(out.values, expected.values)
    assert out.attrs["flatten_method"] == "vertical_max"
    assert out.attrs["start_altitude"] == 1000.0
    assert out.attrs["end_altitude"] == 3000.0


def test_cross_section_single_altitude():
    grid = _grid()
    out = preprocess.cross_section(grid, _options(2000.0))
    assert "altitude" not in out.dims
    assert np.array_equal(out.values, grid.sel(altitude=2000.0).values)
    assert out.attrs["flatten_method"] == "cross_section"
    assert out.attrs["altitude"] == 2000.0
