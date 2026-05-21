"""Utilities for writing data to disk."""

from pathlib import Path
import numpy as np
import xarray as xr


def pin_datetime_encoding(data_object: xr.Dataset | xr.DataArray):
    """Force a stable CF time encoding on every datetime variable."""

    encoding = {
        "units": "seconds since 1970-01-01",
        "dtype": "int64",
        "_FillValue": None,
    }

    if isinstance(data_object, xr.Dataset):
        names = [data_object[name] for name in data_object.variables]
    else:
        names = []
        if np.issubdtype(data_object.dtype, np.datetime64):
            names.append(data_object)
        names += [data_object.coords[name] for name in data_object.coords]
    for var in names:
        if np.issubdtype(var.dtype, np.datetime64):
            var.encoding = dict(encoding)


def zarr_group_exists(store_path, group):
    """Return True if a group already exists inside the zarr DirectoryStore."""
    candidate = Path(store_path) / group
    if not candidate.exists():
        return False
    return any(candidate.iterdir())


def write_interval_reached(next_time, object_tracks, object_options):
    """Check if the write interval has been reached."""

    try:
        last_write_time = object_tracks._last_write_time
    except TypeError:
        last_write_time = object_tracks._last_write_time

    # Initialise _last_write_time if not already done
    if last_write_time is None:
        # Assume write_interval units of hours, so drop minutes from next_time
        try:
            object_tracks._last_write_time = next_time.astype("datetime64[h]")
        except TypeError:
            object_tracks._last_write_time = next_time.astype("datetime64[h]")
        last_write_time = next_time.astype("datetime64[h]")

    # Check if write interval reached; if so, write masks to file
    time_diff = next_time - last_write_time
    try:
        write_interval = object_options.write_interval
    except AttributeError:
        write_interval = object_options["write_interval"]
    write_interval = np.timedelta64(write_interval, "h")

    return time_diff >= write_interval
