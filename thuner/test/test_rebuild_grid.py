"""Tests for rebuilding the per-object processed grids used by visualization.

These run without any data download: they construct default track options and feed
synthetic grids straight into ``detect.rebuild_processed_grid``.
"""

import numpy as np
import xarray as xr
import thuner.default as default
import thuner.detect.detect as detect


def _grid_2d(value, time):
    """A simple 2D (lat, lon) grid with a single time, filled with ``value``."""
    da = xr.DataArray(
        np.full((2, 2), float(value)),
        coords={"latitude": np.array([0.0, 1.0]), "longitude": np.array([0.0, 1.0])},
        dims=["latitude", "longitude"],
        name="reflectivity",
    )
    return da.expand_dims(time=[np.datetime64(time)])


def test_rebuild_grid_uses_each_members_own_dataset():
    """A grouped object whose members come from different datasets must rebuild each
    member panel from its OWN dataset (the ACCESS 1 km vs column-max bug)."""
    track_options = default.access_c_track()  # convective->access_1km, anvil->access_maxcol
    grids = {
        "access_1km": _grid_2d(1.0, "2021-12-02T06:00:00"),
        "access_maxcol": _grid_2d(2.0, "2021-12-02T06:00:00"),
    }
    result = detect.rebuild_processed_grid(grids, track_options, "mcs", 1)

    assert set(result.data_vars) == {"convective_grid", "anvil_grid"}
    # convective is detected from access_1km (1.0), anvil from access_maxcol (2.0).
    assert (result["convective_grid"] == 1.0).all()
    assert (result["anvil_grid"] == 2.0).all()


def test_rebuild_grid_same_dataset_group_is_unchanged():
    """Regression guard: when all members share one dataset (cpol mcs), every member
    resolves to that single grid and is flattened per its own altitude band."""
    track_options = default.track("cpol")  # convective/middle/anvil all from "cpol"
    altitudes = np.array([500.0, 3000.0, 5000.0, 8000.0, 10000.0])
    da = xr.DataArray(
        np.arange(len(altitudes))[:, None, None] * np.ones((len(altitudes), 2, 2)),
        coords={
            "altitude": altitudes,
            "latitude": np.array([0.0, 1.0]),
            "longitude": np.array([0.0, 1.0]),
        },
        dims=["altitude", "latitude", "longitude"],
        name="reflectivity",
    )
    da = da.expand_dims(time=[np.datetime64("2005-11-13T14:00:00")])
    grids = {"cpol": da}

    result = detect.rebuild_processed_grid(grids, track_options, "mcs", 1)

    assert set(result.data_vars) == {"convective_grid", "middle_grid", "anvil_grid"}
    # All members derive from the single "cpol" grid, flattened over altitude.
    for name in result.data_vars:
        assert "altitude" not in result[name].dims
