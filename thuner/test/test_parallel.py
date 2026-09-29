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
import thuner.option as option
import thuner.data as data
import thuner.data.synthetic as synthetic


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
    times = _times(5)
    intervals, interval_times, num_processes = parallel.get_time_intervals(times, 4)
    assert num_processes == 1
    assert len(intervals) == 1
    # The single interval carries the whole domain.
    assert list(interval_times[0]) == list(times)


def test_get_time_intervals_splits_and_covers_domain():
    times = _times(24)
    intervals, interval_times, num_processes = parallel.get_time_intervals(times, 4)
    assert len(intervals) >= 2
    # The union of intervals spans the full time domain.
    assert intervals[0][0] == str(pd.Timestamp(times[0]))
    assert intervals[-1][1] == str(pd.Timestamp(times[-1]))
    # Each interval's explicit time list lines up with its (start, end) boundaries.
    assert len(interval_times) == len(intervals)
    for (start, end), itimes in zip(intervals, interval_times):
        assert str(pd.Timestamp(itimes[0])) == start
        assert str(pd.Timestamp(itimes[-1])) == end
    # Adjacent intervals overlap (share boundary times) so there are no gaps, and the
    # union of all intervals recovers every time in the domain exactly.
    for earlier, later in zip(interval_times, interval_times[1:]):
        assert set(earlier) & set(later)
    covered = sorted(set().union(*(set(it) for it in interval_times)))
    assert covered == list(times)


def _geo_grid():
    """A small geographic grid with a single altitude level."""
    lat = np.round(np.arange(-12.0, -8.0, 0.1), 5).tolist()
    lon = np.round(np.arange(130.0, 134.0, 0.1), 5).tolist()
    go = option.grid.GridOptions(
        name="geographic", latitude=lat, longitude=lon, geographic_spacing=[0.1, 0.1]
    )
    go.altitude = [3000.0]
    return go


def test_parallel_synthetic_intervals_match_serial():
    """The parallel data path reproduces a serial synthetic run, interval by interval.

    ``anchor_synthetic_generators`` records the run's full time grid and
    ``get_interval_data_options`` deep-copies it into each interval. Each interval's
    generator -- including one whose interval starts mid-run -- must then render exactly
    what a single serial pass renders at the same times, so the per-interval tracks stitch
    as if the run was never split. Uses the random generator so RNG-driven spawns (the
    state most sensitive to where stepping begins) are exercised.
    """
    go = _geo_grid()
    times = _times(13)  # enough that get_time_intervals actually splits the domain

    generator = synthetic.RandomEllipseGenerator(
        seed=7, spawn_rate=15.0, initial_count=2
    )
    data_options = option.data.DataOptions(
        datasets=[synthetic.SyntheticOptions(generator=generator)]
    )

    # Serial reference: one generator stepped over every time, in order.
    serial_gen = generator.model_copy(deep=True)
    serial_fields = {t: serial_gen.step(t, go) for t in times}

    # Parallel path: anchor the run's time grid, split into intervals, then drive each
    # interval's independently deep-copied generator just as a worker would.
    parallel.anchor_synthetic_generators(data_options, times)
    assert data_options.datasets[0].generator.run_times is not None
    intervals, interval_times, _ = parallel.get_time_intervals(times, 2)
    assert len(intervals) >= 2  # the split actually happened

    for interval, itimes in zip(intervals, interval_times):
        interval_options = parallel.get_interval_data_options(data_options, interval)
        gen = interval_options.datasets[0].generator
        assert gen.run_times is not None  # survived the per-interval deep copy
        for t in itimes:
            ds = gen.step(t, go)
            xr.testing.assert_allclose(
                ds["reflectivity"], serial_fields[t]["reflectivity"]
            )
