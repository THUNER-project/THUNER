"""Utilities for writing data to disk."""

import numpy as np


def write_interval_reached(time, object_tracks, object_options):
    """Check if the write interval has been reached."""

    # Initialise last_write_time if not already done
    if object_tracks["last_write_time"] is None:
        # Assume write_interval units of hours, so drop minutes from current_time
        object_tracks["last_write_time"] = time.astype("datetime64[h]")

    # Check if write interval reached; if so, write masks to file
    last_write_time = object_tracks["last_write_time"]
    time_diff = time - last_write_time
    try:
        write_interval = object_options.write_interval
    except AttributeError:
        write_interval = object_options["write_interval"]
    write_interval = np.timedelta64(write_interval, "h")

    return time_diff >= write_interval
