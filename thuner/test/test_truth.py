"""Unit tests for synthetic ground-truth → detected-object matching."""

from types import SimpleNamespace
import numpy as np
import pandas as pd
import xarray as xr

from thuner.analyze.synthetic import match_truth_to_masks, NO_MATCH
from thuner.data.synthetic.objects import SyntheticObject, EllipsoidObject

LATS = [0.0, 1.0, 2.0, 3.0]
LONS = [10.0, 11.0, 12.0, 13.0]
LON2D, LAT2D = np.meshgrid(LONS, LATS)
TIME = np.datetime64("2005-11-13T00:00:00", "ns")
# grid_options stand-ins carry the coordinate arrays the masks share (to_mask reads them).
GEOGRAPHIC = SimpleNamespace(
    name="geographic", latitude=np.array(LATS), longitude=np.array(LONS)
)
CARTESIAN = SimpleNamespace(name="cartesian", latitude=LAT2D, longitude=LON2D)


def _mask(label_at):
    """A single-time geographic mask; label_at maps (lat_idx, lon_idx) -> uid."""
    arr = np.zeros((1, len(LATS), len(LONS)), dtype=np.uint32)
    for (i, j), uid in label_at.items():
        arr[0, i, j] = uid
    return xr.DataArray(
        arr,
        dims=("time", "latitude", "longitude"),
        coords={"time": [TIME], "latitude": LATS, "longitude": LONS},
    )


def _cartesian_mask(label_at):
    """A single-time Cartesian mask with 2D curvilinear lat/lon coords over (y, x)."""
    arr = np.zeros((1, len(LATS), len(LONS)), dtype=np.uint32)
    for (i, j), uid in label_at.items():
        arr[0, i, j] = uid
    return xr.DataArray(
        arr,
        dims=("time", "y", "x"),
        coords={
            "time": [TIME],
            "latitude": (("y", "x"), LAT2D),
            "longitude": (("y", "x"), LON2D),
        },
    )


def _point(obj_id, lat, lon):
    """A bare object (no extent), whose truth mask is the nearest grid cell."""
    return SyntheticObject(
        id=obj_id,
        time=str(TIME),
        center_latitude=lat,
        center_longitude=lon,
        direction=0.0,
        speed=0.0,
    )


def _ellipse(obj_id, lat, lon, major, minor):
    """A circular/elliptical truth object centred at (lat, lon)."""
    return EllipsoidObject(
        id=obj_id,
        time=str(TIME),
        center_latitude=lat,
        center_longitude=lon,
        direction=0.0,
        speed=0.0,
        major=major,
        minor=minor,
    )


def _truth_and_objects(objects):
    """Build a truth table aligned row-for-row with ``objects`` (as the replay would)."""
    truth = pd.DataFrame(
        {
            "time": [TIME] * len(objects),
            "id": [obj.id for obj in objects],
            "latitude": [obj.center_latitude for obj in objects],
            "longitude": [obj.center_longitude for obj in objects],
        }
    )
    return truth, objects


def test_match_point_geographic():
    da = _mask({(1, 1): 7, (1, 2): 7})  # object 7 spans cells (1,11) and (1,12)
    truth, objects = _truth_and_objects(
        [_point(0, 1.0, 11.0), _point(1, 3.0, 13.0)]  # in object 7, then background
    )
    sources = {"convective_universal_id": [da]}
    out = match_truth_to_masks(truth, sources, objects, GEOGRAPHIC)
    assert list(out["convective_universal_id"]) == ["7", NO_MATCH]


def test_match_grouped_union():
    convective = _mask({(1, 1): 3})  # mcs 3 convective part
    middle = _mask({(2, 2): 3})  # mcs 3 middle part (same group uid)
    truth, objects = _truth_and_objects(
        [_point(0, 1.0, 11.0), _point(1, 2.0, 12.0), _point(2, 0.0, 10.0)]
    )
    sources = {"mcs_universal_id": [convective, middle]}
    out = match_truth_to_masks(truth, sources, objects, GEOGRAPHIC)
    assert list(out["mcs_universal_id"]) == ["3", "3", NO_MATCH]


def test_match_interleaved_members_records_set():
    # Different uids at the same cell (e.g. members of two different groups). The old
    # centre-based matcher raised here; mask overlap records both uids instead.
    convective = _mask({(1, 1): 3})
    middle = _mask({(1, 1): 5})
    truth, objects = _truth_and_objects([_point(0, 1.0, 11.0)])
    sources = {"mcs_universal_id": [convective, middle]}
    out = match_truth_to_masks(truth, sources, objects, GEOGRAPHIC)
    assert list(out["mcs_universal_id"]) == ["3 5"]


def test_match_ellipse_spans_multiple_objects():
    # An ellipse wide enough to cover two detected objects is ambiguous: both uids recorded.
    da = _mask({(1, 1): 4, (1, 2): 8})
    truth, objects = _truth_and_objects(
        [_ellipse(0, 1.0, 11.5, major=200.0, minor=200.0)]
    )
    sources = {"convective_universal_id": [da]}
    out = match_truth_to_masks(truth, sources, objects, GEOGRAPHIC)
    assert list(out["convective_universal_id"]) == ["4 8"]


def test_match_ellipse_single_object():
    # A small ellipse covering only one detected object is an unambiguous match.
    da = _mask({(1, 1): 4, (3, 3): 8})
    truth, objects = _truth_and_objects(
        [_ellipse(0, 1.0, 11.0, major=20.0, minor=20.0)]
    )
    sources = {"convective_universal_id": [da]}
    out = match_truth_to_masks(truth, sources, objects, GEOGRAPHIC)
    assert list(out["convective_universal_id"]) == ["4"]


def test_match_cartesian():
    da = _cartesian_mask({(1, 1): 9})
    truth, objects = _truth_and_objects([_point(0, 1.0, 11.0), _point(1, 3.0, 13.0)])
    sources = {"echo_id": [da]}
    out = match_truth_to_masks(truth, sources, objects, CARTESIAN)
    assert list(out["echo_id"]) == ["9", NO_MATCH]
