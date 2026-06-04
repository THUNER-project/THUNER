"""Unit tests for the synthetic scene generators (stepping, lifecycle, culling)."""

import numpy as np
import pytest

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
    assert np.all(np.isnan(fields[0]["reflectivity"].values))  # scale 0 -> nothing drawn
    assert float(np.nanmax(fields[2]["reflectivity"].values)) == pytest.approx(peak, abs=1.0)
