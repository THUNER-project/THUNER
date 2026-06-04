"""Unit tests for synthetic object geometry, ground truth and rendering."""

import numpy as np
import xarray as xr
import pytest

from thuner.data.synthetic.objects import EllipsoidObject


def _ellipsoid(**kwargs):
    """An EllipsoidObject with sensible defaults, overridable per test."""
    params = dict(
        time="2005-11-13T00:00:00",
        center_latitude=-10.0,
        center_longitude=132.0,
        direction=0.0,
        speed=0.0,
        major=10.0,
        minor=6.0,
        orientation=0.0,
    )
    params.update(kwargs)
    return EllipsoidObject(**params)


def test_ground_truth_infers_eccentricity_and_rounds():
    # major=10, minor=6 -> eccentricity = sqrt(1 - (6/10)^2) = sqrt(0.64) = 0.8.
    truth = _ellipsoid(major=10.0, minor=6.0, orientation=1.23456).ground_truth()
    assert truth["orientation"] == 1.2346  # 4 dp
    assert truth["eccentricity"] == pytest.approx(0.8, abs=1e-4)  # inferred, 4 dp
    assert "horizontal_radius" not in truth  # old field is gone
    # major/minor are rounded to 1 dp.
    rounded = _ellipsoid(major=10.04, minor=6.06).ground_truth()
    assert rounded["major"] == 10.0 and rounded["minor"] == 6.1


def test_axes_validator_rejects_minor_greater_than_major():
    with pytest.raises(ValueError):
        _ellipsoid(major=10.0, minor=12.0)


def test_axes_validator_rejects_nonpositive_axis():
    with pytest.raises(ValueError):
        _ellipsoid(major=10.0, minor=0.0)


def _grid_dataset(lat, lon, alt=3000.0):
    """A minimal generator-style dataset carrying broadcast LON/LAT/ALT coords."""
    shape = (1, 1, len(lat), len(lon))
    ds = xr.Dataset(
        {"reflectivity": (["time", "altitude", "latitude", "longitude"], np.zeros(shape))},
        coords={
            "time": [np.datetime64("2005-11-13T00:00:00", "ns")],
            "altitude": [alt],
            "latitude": lat,
            "longitude": lon,
        },
    )
    LON, LAT, ALT = xr.broadcast(ds.time, ds.longitude, ds.latitude, ds.altitude)[1:]
    ds["LON"], ds["LAT"], ds["ALT"] = LON, LAT, ALT
    return ds


def test_render_blob_centered_and_elliptical():
    lat = np.arange(-11.0, -9.0, 0.02)
    lon = np.arange(131.0, 133.0, 0.02)
    ds = _grid_dataset(lat, lon)
    # Full major axis (30 km) along east, minor (10 km) along north -> wider in longitude.
    obj = _ellipsoid(major=30.0, minor=10.0, orientation=0.0, intensity=50.0)
    rendered = obj.render(ds, grid_options=None)
    field = rendered["reflectivity"].isel(time=0, altitude=0)

    # Peak is the intensity and sits at the object centre.
    assert float(field.max()) == pytest.approx(50.0, abs=1.0)
    i, j = np.unravel_index(np.nanargmax(field.values), field.shape)
    assert lat[i] == pytest.approx(-10.0, abs=0.05)
    assert lon[j] == pytest.approx(132.0, abs=0.05)

    # The lit footprint spans wider in longitude than latitude (major along east).
    # render leaves untouched cells at their original 0, so isolate the blob with > 0.
    lit = field > 0
    lit_lon = lon[lit.any("latitude").values]
    lit_lat = lat[lit.any("longitude").values]
    assert np.ptp(lit_lon) > np.ptp(lit_lat)


def test_render_flat_is_uniform_and_tighter_than_gaussian():
    lat = np.arange(-11.0, -9.0, 0.02)
    lon = np.arange(131.0, 133.0, 0.02)
    kw = dict(major=30.0, minor=10.0, orientation=0.0, intensity=50.0)
    flat = (
        _ellipsoid(style="flat", **kw)
        .render(_grid_dataset(lat, lon), grid_options=None)["reflectivity"]
        .isel(time=0, altitude=0)
    )
    gauss = (
        _ellipsoid(style="gaussian", **kw)
        .render(_grid_dataset(lat, lon), grid_options=None)["reflectivity"]
        .isel(time=0, altitude=0)
    )
    # Flat fill is uniform at the intensity (no Gaussian falloff). Untouched cells stay
    # at their original 0, so the blob is the > 0 region.
    flat_vals = flat.values[flat.values > 0]
    assert flat_vals.size > 0 and np.allclose(flat_vals, 50.0)
    # Flat footprint (ellipse edge, distance<=1) is tighter than the Gaussian's 5% one.
    flat_lon = lon[(flat > 0).any("latitude").values]
    gauss_lon = lon[(gauss > 0).any("latitude").values]
    assert np.ptp(flat_lon) < np.ptp(gauss_lon)
