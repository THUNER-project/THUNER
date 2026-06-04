"""Unit tests for thuner.attribute.tag.

The mask-based tagging (``from_masks``) is not yet exercised by any demo, so it is
covered here on small synthetic datasets. ``setup_interp`` (which loads the tagging
dataset from the input records) is mocked; the interpolation/averaging logic and the
real ``get_nearest_points`` then run on the synthetic data.
"""

import types
import numpy as np
import xarray as xr
import pytest

import thuner.attribute.tag as tag
import thuner.attribute.utils as attribute_utils


def _cape_dataset():
    times = np.array(["2005-11-13T13:50:00", "2005-11-13T14:00:00"], dtype="datetime64[ns]")
    lats = [0.0, 1.0, 2.0]
    lons = [10.0, 11.0, 12.0]
    space = np.arange(9.0).reshape(3, 3)  # cape varies in space, constant in time
    cape = np.stack([space, space])
    ds = xr.Dataset(
        {"cape": (("time", "latitude", "longitude"), cape)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    return ds, lats, lons, times


def test_from_masks_averages_over_object(monkeypatch):
    ds, lats, lons, times = _cape_dataset()
    current_time = times[-1]
    core_attributes = {"latitude": [1.0], "longitude": [11.0], "universal_id": [5]}
    monkeypatch.setattr(
        attribute_utils, "setup_interp",
        lambda **kwargs: ("mcs", ["cape"], ds, core_attributes, current_time),
    )
    mask = xr.DataArray(
        np.array([[0, 0, 0], [0, 5, 5], [0, 0, 0]]),
        dims=("latitude", "longitude"),
        coords={"latitude": lats, "longitude": lons},
    )
    object_tracks = types.SimpleNamespace(
        name="mcs", times=[current_time], masks=[mask], matched_masks=[mask]
    )
    result = tag.from_masks(None, None, object_tracks, "era5_sl", [0])
    assert result["universal_id"] == [5]
    assert result["time_offset"] == [0]
    # object 5 spans cape cells with values 4 and 5 -> mean 4.5
    assert result["cape"] == [pytest.approx(4.5)]


def test_from_masks_requires_id(monkeypatch):
    ds, lats, lons, times = _cape_dataset()
    current_time = times[-1]
    core_attributes = {"latitude": [1.0], "longitude": [11.0]}  # no id/universal_id
    monkeypatch.setattr(
        attribute_utils, "setup_interp",
        lambda **kwargs: ("mcs", ["cape"], ds, core_attributes, current_time),
    )
    object_tracks = types.SimpleNamespace(name="mcs", times=[current_time])
    with pytest.raises(ValueError):
        tag.from_masks(None, None, object_tracks, "era5_sl", [0])


def test_default_matched_and_unmatched():
    matched = tag.default("era5_sl", matched=True)
    unmatched = tag.default("era5_sl", matched=False)
    matched_names = [a.name for a in matched.attributes[0].attributes]
    unmatched_names = [a.name for a in unmatched.attributes[0].attributes]
    assert "universal_id" in matched_names and "id" not in matched_names
    assert "id" in unmatched_names and "universal_id" not in unmatched_names
