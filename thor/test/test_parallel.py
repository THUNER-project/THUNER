"""Test setup."""

import shutil
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
import thor.parallel as parallel
from thor.log import setup_logger
import thor.analyze as analyze

logger = setup_logger(__name__)

# Suppress the "wayland" plugin warning
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def setup(start, end, options_directory, grid_type="geographic"):

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

    altitude = list(np.arange(0, 20e3 + 500, 500))
    altitude = [float(alt) for alt in altitude]
    grid_options = grid.create_options(name=grid_type, altitude=altitude)
    grid.check_options(grid_options)
    grid.save_grid_options(grid_options, options_directory)

    # Create the track_options dictionary
    track_options = option.mcs(dataset="cpol")
    option.check_options(track_options)
    option.save_track_options(track_options, options_directory)
    visualize_options = None

    return data_options, grid_options, track_options, visualize_options


def process_interval(i, time_interval, output_parent):
    print(f"Interval {i}: {time_interval}")

    start = time_interval[0]
    end = time_interval[1]

    output_directory = output_parent / f"interval_{i}"
    options_directory = output_directory / "options"

    all_options = setup(start, end, options_directory)
    data_options, grid_options, track_options, visualize_options = all_options

    # Track
    times = data.utils.generate_times(data_options["cpol"])
    tracks = track.simultaneous_track(
        times,
        data_options,
        grid_options,
        track_options,
        visualize_options,
        output_directory=output_directory,
        parallel=True,
    )


if __name__ == "__main__":

    # Parent directory for saving outputs
    base_local = Path.home() / "THOR_output"
    start = "2005-11-13T14:00"
    end = "2005-11-13T18:00"

    intervals = parallel.generate_time_intervals(start, end)
    output_parent = base_local / "runs/cpol_demo_parallel"
    if output_parent.exists():
        shutil.rmtree(output_parent)

    setup(start, end, output_parent / "options")

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

    parallel.stitch_run(output_parent, intervals)
    analysis_options = analyze.mcs.analysis_options()
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent)

    figure_options = visualize.option.horizontal_attribute_options(
        "mcs_velocity_analysis", style="presentation"
    )
    start_time = np.datetime64("2005-11-13T12:00")
    end_time = np.datetime64("2005-11-13T22:00")
    visualize.attribute.mcs_series(
        output_parent, start_time, end_time, figure_options, parallel=True
    )
