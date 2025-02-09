"""Utilities for writing data to disk."""

import numpy as np


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
