"""Unit tests for the synthetic scene generators (stepping, lifecycle, culling)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import thuner.option as option
import thuner.data.synthetic as synthetic

START = "2005-11-13T00:00:00"


def _grid():
    """A small geographic grid around (-10, 132) with a single altitude level."""
    lat = np.round(np.arange(-12.0, -8.0, 0.05), 5).tolist()
    lon = np.round(np.arange(130.0, 134.0, 0.05), 5).tolist()
    go = option.grid.GridOptions(
        name="geographic", latitude=lat, longitude=lon, geographic_spacing=[0.05, 0.05]
    )
    go.altitude = [3000.0]
    return go


def _times(n, step_min=10):
    start = np.datetime64(START)
    return [start + np.timedelta64(step_min * i, "m") for i in range(n)]


def _cell(**kw):
    params = dict(
        time=START,
        center_latitude=-10.0,
        center_longitude=132.0,
        direction=0.0,
        speed=0.0,
        major=40.0,
        minor=16.0,
        orientation=0.0,
    )
    params.update(kw)
    return synthetic.EllipsoidObject(**params)


def _peak_lat_lon(ds):
    field = ds["reflectivity"].isel(time=0).max("altitude")
    i, j = np.unravel_index(np.nanargmax(field.values), field.shape)
    return field.latitude.values[i], field.longitude.values[j]


def test_ground_truth_matches_rendered_positions():
    """The replayed ground truth must agree with where objects are actually rendered."""
    go = _grid()
    obj = _cell(speed=20.0, direction=np.pi / 2)  # moving east; no fade, no death
    times = _times(3)
    truth = synthetic.FixedGenerator(objects=[obj]).ground_truth(times, go).sort_index()

    # Step a fresh generator over the same times (incremental, like a real run).
    gen = synthetic.FixedGenerator(objects=[obj])
    rendered = [_peak_lat_lon(gen.step(t, go)) for t in times]

    assert np.allclose([r[0] for r in rendered], truth["latitude"].values, atol=0.06)
    assert np.allclose([r[1] for r in rendered], truth["longitude"].values, atol=0.06)


def test_lifetime_culls_object():
    go = _grid()
    obj = _cell(life_time=15.0)  # dies between the +10 and +20 minute steps
    truth = synthetic.FixedGenerator(objects=[obj]).ground_truth(_times(3), go)
    assert len(truth.index.get_level_values("time").unique()) == 2  # start and +10 only


def test_out_of_domain_culled_unless_buffered():
    go = _grid()  # longitude up to ~134
    obj = _cell(center_longitude=133.9, speed=300.0, direction=np.pi / 2)  # races east
    truth = synthetic.FixedGenerator(objects=[obj]).ground_truth(_times(3), go)
    assert len(truth.index.get_level_values("time").unique()) == 1  # gone by +10

    buffered = synthetic.FixedGenerator(objects=[obj], domain_buffer=500.0)
    truth_b = buffered.ground_truth(_times(3), go)
    assert len(truth_b.index.get_level_values("time").unique()) >= 2  # buffer keeps it


def test_fade_scales_intensity_in_truth_and_render():
    go = _grid()
    obj = _cell(fade_in_time=10.0)  # ramps 0 -> 1 over the first 10 minutes
    peak = obj.intensity
    times = _times(3, step_min=5)  # 0, 5, 10 minutes
    truth = synthetic.FixedGenerator(objects=[obj]).ground_truth(times, go).sort_index()
    intensity = truth["intensity"].values
    assert intensity[0] == pytest.approx(0.0, abs=0.01)  # nothing at birth
    assert intensity[1] == pytest.approx(0.5 * peak, abs=0.6)  # half way in
    assert intensity[2] == pytest.approx(peak, abs=0.6)  # fully faded in

    gen = synthetic.FixedGenerator(objects=[obj])
    fields = [gen.step(t, go) for t in times]
    assert np.all(
        np.isnan(fields[0]["reflectivity"].values)
    )  # scale 0 -> nothing drawn
    assert float(np.nanmax(fields[2]["reflectivity"].values)) == pytest.approx(
        peak, abs=1.0
    )


def test_random_generator_deterministic_and_consistent():
    go = _grid()
    times = _times(6)
    kw = dict(seed=3, spawn_rate=20.0, initial_count=2)

    # Same seed -> identical scene; different seed -> different scene.
    truth = synthetic.RandomEllipseGenerator(**kw).ground_truth(times, go)
    truth_again = synthetic.RandomEllipseGenerator(**kw).ground_truth(times, go)
    pd.testing.assert_frame_equal(truth, truth_again)
    assert len(truth) > 0
    other = synthetic.RandomEllipseGenerator(**{**kw, "seed": 99}).ground_truth(
        times, go
    )
    assert not truth.equals(other)

    # A render run (step) reproduces the ground-truth re-run exactly (RNG determinism).
    gen = synthetic.RandomEllipseGenerator(**kw)
    rows = []
    for time in times:
        gen.step(time, go)
        rows.extend(obj.ground_truth() for obj in gen._live)
    rendered = pd.DataFrame(rows).set_index(["time", "id"]).sort_index()
    pd.testing.assert_frame_equal(rendered, truth)


def test_run_times_fast_forward_matches_serial_random():
    """A run_times-anchored generator started mid-run reproduces the serial scene.

    Simulates a parallel worker whose interval begins partway through the run: its first
    ``step`` is at a later time, so the fast-forward must rebuild the exact state a serial
    run would have there -- RNG-driven spawns, motion and culling included -- before
    rendering. We compare the rendered field (what the tracker actually sees) frame by
    frame over the worker's tail of the run.
    """
    go = _grid()
    times = _times(6)
    run_times = [str(pd.Timestamp(t)) for t in times]
    kw = dict(seed=3, spawn_rate=20.0, initial_count=2)

    serial = synthetic.RandomEllipseGenerator(**kw)
    serial_fields = [serial.step(t, go) for t in times]

    # Worker interval starts at times[3]; the first step must fast-forward through 0..3.
    worker = synthetic.RandomEllipseGenerator(run_times=run_times, **kw)
    for k in range(3, len(times)):
        ds = worker.step(times[k], go)
        xr.testing.assert_allclose(ds["reflectivity"], serial_fields[k]["reflectivity"])


def test_run_times_fast_forward_matches_serial_fixed():
    """Fixed-generator motion is a chain of geodesic hops, so a fast-forwarded worker must
    replay every step to land where a serial run did. A single jump to the interval's
    start would diverge -- this guards that the fast-forward steps, not jumps."""
    go = _grid()
    times = _times(6)
    run_times = [str(pd.Timestamp(t)) for t in times]
    obj = _cell(speed=25.0, direction=np.pi / 3)  # moving north-east

    serial = synthetic.FixedGenerator(objects=[obj])
    serial_fields = [serial.step(t, go) for t in times]

    worker = synthetic.FixedGenerator(objects=[obj], run_times=run_times)
    for k in range(3, len(times)):
        ds = worker.step(times[k], go)
        xr.testing.assert_allclose(ds["reflectivity"], serial_fields[k]["reflectivity"])


def test_random_generator_round_trips_through_options_union():
    gen = synthetic.RandomEllipseGenerator(seed=1, spawn_rate=5.0)
    options = synthetic.SyntheticOptions(generator=gen)
    restored = synthetic.SyntheticOptions.model_validate(options.model_dump())
    assert isinstance(restored.generator, synthetic.RandomEllipseGenerator)
    assert restored.generator.seed == 1
