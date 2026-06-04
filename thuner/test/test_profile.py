"""Unit tests for thuner.attribute.profile.

Mask-based profiling (``from_masks``) and the error branches are not exercised by any
demo, so they are covered here on a small synthetic pressure-level dataset (with
``setup_interp`` mocked).
"""

import types
import numpy as np
import xarray as xr
import pytest

import thuner.attribute.profile as profile
import thuner.attribute.utils as attribute_utils

_G = 9.80665


def _profile_dataset(with_geopotential=True):
    times = np.array(["2005-11-13T13:50:00", "2005-11-13T14:00:00"], dtype="datetime64[ns]")
    lats = [0.0, 1.0, 2.0]
    lons = [10.0, 11.0, 12.0]
    pressure = [1000.0, 850.0, 500.0]
    temp_1d = np.array([300.0, 290.0, 250.0])
    temp = np.broadcast_to(temp_1d[None, :, None, None], (2, 3, 3, 3)).astype(float)
    data = {"temperature": (("time", "pressure", "latitude", "longitude"), temp.copy())}
    if with_geopotential:
        geo_1d = np.array([0.0, 1500.0, 5500.0]) * _G  # altitude 0/1500/5500 m
        geo = np.broadcast_to(geo_1d[None, :, None, None], (2, 3, 3, 3)).astype(float)
        data["geopotential"] = (("time", "pressure", "latitude", "longitude"), geo.copy())
    ds = xr.Dataset(
        data,
        coords={"time": times, "pressure": pressure, "latitude": lats, "longitude": lons},
    )
    return ds, lats, lons, times


def _object_tracks(lats, lons, current_time):
    mask = xr.DataArray(
        np.array([[0, 0, 0], [0, 5, 5], [0, 0, 0]]),
        dims=("latitude", "longitude"),
        coords={"latitude": lats, "longitude": lons},
    )
    return types.SimpleNamespace(
        name="mcs", times=[current_time], masks=[mask], matched_masks=[mask]
    )


def _mock_setup(monkeypatch, ds, core_attributes, current_time):
    monkeypatch.setattr(
        attribute_utils, "setup_interp",
        lambda **kwargs: ("mcs", ["temperature"], ds, core_attributes, current_time),
    )


def test_from_masks_interpolates_to_grid_altitudes(monkeypatch):
    ds, lats, lons, times = _profile_dataset()
    current_time = times[-1]
    core_attributes = {"latitude": [1.0], "longitude": [11.0], "universal_id": [5]}
    _mock_setup(monkeypatch, ds, core_attributes, current_time)
    grid_options = types.SimpleNamespace(altitude=[1000.0, 3000.0])
    object_tracks = _object_tracks(lats, lons, current_time)

    result = profile.from_masks(None, None, object_tracks, grid_options, "era5_pl", [0])
    # One profile (id 5) sampled at the two requested altitudes.
    assert result["altitude"] == [1000.0, 3000.0]
    assert result["universal_id"] == [5, 5]
    # Linear interp of temperature(altitude) at 1000 m and 3000 m.
    assert result["temperature"][0] == pytest.approx(300 - (1000 / 1500) * 10)
    assert result["temperature"][1] == pytest.approx(290 - (1500 / 4000) * 40)


def test_from_masks_requires_pressure_and_geopotential(monkeypatch):
    ds, lats, lons, times = _profile_dataset(with_geopotential=False)
    current_time = times[-1]
    core_attributes = {"latitude": [1.0], "longitude": [11.0], "universal_id": [5]}
    _mock_setup(monkeypatch, ds, core_attributes, current_time)
    grid_options = types.SimpleNamespace(altitude=[1000.0, 3000.0])
    object_tracks = _object_tracks(lats, lons, current_time)
    with pytest.raises(ValueError):
        profile.from_masks(None, None, object_tracks, grid_options, "era5_pl", [0])


def test_from_masks_requires_id(monkeypatch):
    ds, lats, lons, times = _profile_dataset()
    current_time = times[-1]
    core_attributes = {"latitude": [1.0], "longitude": [11.0]}  # no id/universal_id
    _mock_setup(monkeypatch, ds, core_attributes, current_time)
    grid_options = types.SimpleNamespace(altitude=[1000.0, 3000.0])
    object_tracks = _object_tracks(lats, lons, current_time)
    with pytest.raises(ValueError):
        profile.from_masks(None, None, object_tracks, grid_options, "era5_pl", [0])


def test_default_matched_and_unmatched():
    matched = profile.default("era5_pl", matched=True)
    unmatched = profile.default("era5_pl", matched=False)
    matched_names = [a.name for a in matched.attributes[0].attributes]
    unmatched_names = [a.name for a in unmatched.attributes[0].attributes]
    assert "universal_id" in matched_names and "id" not in matched_names
    assert "id" in unmatched_names and "universal_id" not in unmatched_names
