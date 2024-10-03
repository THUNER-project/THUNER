"""Test setup."""

from pathlib import Path
import os
import numpy as np
import datetime
import multiprocessing
import thor.data as data
import thor.data.dispatch as dispatch
import thor.grid as grid
import thor.track as track
import thor.option as option
import thor.visualize as visualize
from thor.log import setup_logger

logger = setup_logger(__name__)

# Suppress the "wayland" plugin warning
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def align_to_nearest_hour(time):
    """Align a datetime to the nearest hour."""
    dt = np.datetime64(time).astype(datetime.datetime)
    if dt.minute >= 30:
        dt = dt.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    else:
        dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt


def generate_hourly_intervals(start, end):
    """Generate a list of hourly intervals from start to end."""
    intervals = []

    # Convert np.datetime64 to datetime
    start_dt = np.datetime64(start).astype(datetime.datetime)
    end_dt = np.datetime64(end).astype(datetime.datetime)

    current = align_to_nearest_hour(start_dt)

    while current < end_dt:
        next_hour = current + datetime.timedelta(hours=1)
        # Convert datetime back to np.datetime64
        intervals.append([np.datetime64(current), np.datetime64(next_hour)])
        current = next_hour

    return intervals


def process_interval(i, time_interval, output_parent):
    print(f"Interval {i}: {time_interval}")

    start = str(time_interval[0].astype("datetime64[s]"))
    end = str(time_interval[1].astype("datetime64[s]"))

    output_directory = output_parent / f"interval_{i}"
    options_directory = output_directory / "options"

    # Create the data_options dictionary
    converted_options = {"save": True, "load": False, "parent_converted": None}
    cpol_options = data.aura.cpol_data_options(
        start=start, end=end, converted_options=converted_options
    )
    # Restrict the ERA5 data to a smaller region containing the CPOL radar
    lon_range = [129, 133]
    lat_range = [-14, -10]
    era5_pl_options = data.era5.data_options(
        start=start, end=end, latitude_range=lat_range, longitude_range=lon_range
    )
    era5_sl_options = data.era5.data_options(
        start=start,
        end=end,
        data_format="single-levels",
        latitude_range=lat_range,
        longitude_range=lon_range,
    )
    data_options = option.consolidate_options(
        [cpol_options, era5_pl_options, era5_sl_options]
    )

    dispatch.check_data_options(data_options)
    data.option.save_data_options(data_options, options_directory)

    altitude = list(np.arange(0, 25e3 + 500, 500))
    altitude = [float(alt) for alt in altitude]
    grid_options = grid.create_options(name="geographic", altitude=altitude)
    grid.check_options(grid_options)
    grid.save_grid_options(grid_options, options_directory)

    # Create the track_options dictionary
    track_options = option.mcs(dataset="cpol")
    option.save_track_options(track_options, options_directory)

    # Create the display_options dictionary
    visualize_options = {
        obj: visualize.option.runtime_options(obj, save=True, style="presentation")
        for obj in ["mcs"]
    }
    visualize_options["middle_echo"] = visualize.option.runtime_options(
        "middle_echo", save=True, style="presentation", figure_types=["mask"]
    )
    # visualize.option.save_display_options(visualize_options, options_directory)
    visualize_options = None

    # Track
    times = data.utils.generate_times(data_options["cpol"])
    tracks = track.simultaneous_track(
        times,
        data_options,
        grid_options,
        track_options,
        visualize_options,
        output_directory=output_directory,
        parallel="thread",
    )


if __name__ == "__main__":

    # Parent directory for saving outputs
    base_local = Path.home() / "THOR_output"
    start = "2005-11-13T13:00:09"
    end = "2005-11-13T17:00:00"

    intervals = generate_hourly_intervals(start, end)
    output_parent = base_local / "runs/parallel_demo"

    with multiprocessing.Pool() as pool:
        results = []
        for i, time_interval in enumerate(intervals):
            results.append(
                pool.apply_async(process_interval, (i, time_interval, output_parent))
            )

        for result in results:
            try:
                result.get()  # Wait for the result and handle exceptions
            except Exception as exc:
                print(f"Generated an exception: {exc}")
