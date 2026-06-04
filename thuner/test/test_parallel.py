"""Unit tests for the pure stitching helpers in ``thuner.parallel``.

The demo integration tests exercise the happy path of stitching, but never the
error branches (mismatched masks/times) or the small-domain branches of
``get_time_intervals``; those are covered here on hand-built objects.
"""

import numpy as np
import pandas as pd
import xarray as xr
import pytest

import thuner.parallel as parallel


def _mask_dataarray(values):
    return xr.DataArray(np.array(values), dims=("latitude", "longitude"))


def test_match_dataarray_relabels_objects():
    da_1 = _mask_dataarray([[1, 1, 0], [0, 2, 2]])
    da_2 = _mask_dataarray([[5, 5, 0], [0, 7, 7]])  # same regions, relabelled
    assert parallel.match_dataarray(da_1, da_2) == {1: 5, 2: 7}


def test_match_dataarray_binary_mismatch_returns_empty():
    da_1 = _mask_dataarray([[1, 0], [0, 0]])
    da_2 = _mask_dataarray([[1, 1], [0, 0]])  # different footprint
    assert parallel.match_dataarray(da_1, da_2) == {}


def test_match_dataarray_ambiguous_region_raises():
    da_1 = _mask_dataarray([[1, 1], [0, 0]])
    da_2 = _mask_dataarray([[3, 4], [0, 0]])  # one region -> two ids
    with pytest.raises(ValueError):
        parallel.match_dataarray(da_1, da_2)


def _mask_dataset(values, time="2005-11-13T14:00:00", name="mcs"):
    da = xr.DataArray(np.array(values), dims=("latitude", "longitude"))
    return xr.Dataset({name: da}, coords={"time": np.datetime64(time)})


def test_match_dataset_time_mismatch_raises():
    ds_1 = _mask_dataset([[1, 0]], time="2005-11-13T14:00:00")
    ds_2 = _mask_dataset([[1, 0]], time="2005-11-13T14:10:00")
    with pytest.raises(ValueError):
        parallel.match_dataset(ds_1, ds_2)


def test_match_dataset_mask_name_mismatch_raises():
    ds_1 = _mask_dataset([[1, 0]], name="mcs")
    ds_2 = _mask_dataset([[1, 0]], name="convective")
    with pytest.raises(ValueError):
        parallel.match_dataset(ds_1, ds_2)


def test_apply_mapping_remaps_ids():
    mask = xr.Dataset({"mcs": (("y", "x"), np.array([[0, 1, 2], [1, 2, 0]]))})
    out = parallel.apply_mapping({1: 10, 2: 20}, mask)
    expected = np.array([[0, 10, 20], [10, 20, 0]])
    assert np.array_equal(out["mcs"].values, expected)


def test_apply_mapping_empty_is_noop():
    mask = xr.Dataset({"mcs": (("y", "x"), np.array([[0, 1], [2, 0]]))})
    out = parallel.apply_mapping({}, mask)
    assert np.array_equal(out["mcs"].values, mask["mcs"].values)


def test_get_mapping_missing_object_returns_empty():
    assert parallel.get_mapping({}, "mcs", 0) == {}


def test_get_mapping_returns_interval_mapping():
    index = pd.MultiIndex.from_tuples(
        [(0, 1), (0, 2), (1, 1)], names=["interval", "original_id"]
    )
    id_dicts = {"mcs": pd.DataFrame({"universal_id": [10, 11, 12]}, index=index)}
    assert parallel.get_mapping(id_dicts, "mcs", 0) == {1: 10, 2: 11}


def _times(n):
    base = np.datetime64("2005-11-13T00:00:00")
    return [base + np.timedelta64(10 * i, "m") for i in range(n)]


def test_get_time_intervals_small_domain_uses_one_process():
    intervals, num_processes = parallel.get_time_intervals(_times(5), 4)
    assert num_processes == 1
    assert len(intervals) == 1


def test_get_time_intervals_splits_and_covers_domain():
    times = _times(24)
    intervals, num_processes = parallel.get_time_intervals(times, 4)
    assert len(intervals) >= 2
    # The union of intervals spans the full time domain.
    assert intervals[0][0] == str(pd.Timestamp(times[0]))
    assert intervals[-1][1] == str(pd.Timestamp(times[-1]))
