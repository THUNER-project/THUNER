"""Unit tests for synthetic ground-truth → detected-object matching."""

from types import SimpleNamespace
import numpy as np
import pandas as pd
import xarray as xr
import pytest

from thuner.data.synthetic.truth import match_truth_to_masks, NO_MATCH

LATS = [0.0, 1.0, 2.0, 3.0]
LONS = [10.0, 11.0, 12.0, 13.0]
TIME = np.datetime64("2005-11-13T00:00:00", "ns")
GEOGRAPHIC = SimpleNamespace(name="geographic")


def _mask(label_at):
    """A single-time mask DataArray; label_at maps (lat_idx, lon_idx) -> uid."""
    arr = np.zeros((1, len(LATS), len(LONS)), dtype=int)
    for (i, j), uid in label_at.items():
        arr[0, i, j] = uid
    return xr.DataArray(
        arr,
        dims=("time", "latitude", "longitude"),
        coords={"time": [TIME], "latitude": LATS, "longitude": LONS},
    )


def _truth(centres):
    """centres: list of (lat, lon) for the truth object centres."""
    return pd.DataFrame(
        {
            "time": [TIME] * len(centres),
            "id": list(range(len(centres))),
            "latitude": [lat for lat, _ in centres],
            "longitude": [lon for _, lon in centres],
        }
    )


def test_match_single_object_geographic():
    da = _mask({(1, 1): 7, (1, 2): 7})  # object 7 spans cells (1,11) and (1,12)
    truth = _truth([(1.0, 11.0), (3.0, 13.0)])  # inside object 7, then background
    out = match_truth_to_masks(truth, {"convective_universal_id": [da]}, GEOGRAPHIC)
    assert list(out["convective_universal_id"]) == [7, NO_MATCH]


def test_match_grouped_union():
    convective = _mask({(1, 1): 3})  # mcs 3 convective part
    middle = _mask({(2, 2): 3})  # mcs 3 middle part (same group uid)
    truth = _truth([(1.0, 11.0), (2.0, 12.0), (0.0, 10.0)])
    sources = {"mcs_universal_id": [convective, middle]}
    out = match_truth_to_masks(truth, sources, GEOGRAPHIC)
    assert list(out["mcs_universal_id"]) == [3, 3, NO_MATCH]


def test_match_interleaved_members_raises():
    convective = _mask({(1, 1): 3})
    middle = _mask({(1, 1): 5})  # different uid at the SAME cell -> interleaved
    truth = _truth([(1.0, 11.0)])
    sources = {"mcs_universal_id": [convective, middle]}
    with pytest.raises(ValueError):
        match_truth_to_masks(truth, sources, GEOGRAPHIC)


def test_match_cartesian():
    # Cartesian: 2D curvilinear lat/lon coords over (y, x).
    lon2d, lat2d = np.meshgrid(LONS, LATS)
    arr = np.zeros((1, len(LATS), len(LONS)), dtype=int)
    arr[0, 1, 1] = 9
    da = xr.DataArray(
        arr,
        dims=("time", "y", "x"),
        coords={
            "time": [TIME],
            "latitude": (("y", "x"), lat2d),
            "longitude": (("y", "x"), lon2d),
        },
    )
    truth = _truth([(1.0, 11.0), (3.0, 13.0)])
    out = match_truth_to_masks(truth, {"echo_id": [da]}, SimpleNamespace(name="cartesian"))
    assert list(out["echo_id"]) == [9, NO_MATCH]
