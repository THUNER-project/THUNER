"""Test setup."""

import shutil
from pathlib import Path
import os
import numpy as np
from multiprocessing import get_context
import thuner.data as data
import thuner.data.dispatch as dispatch
import thuner.grid as grid
import thuner.option as option
import thuner.visualize as visualize
import thuner.parallel as parallel
from thuner.parallel import initialize_process
from thuner.log import setup_logger, logging_listener
import thuner.analyze as analyze

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
    track_options = option.default_track_options(dataset="cpol")
    track_options.to_yaml(options_directory / "track.yml")
    visualize_options = None

    return data_options, grid_options, track_options, visualize_options


def test_parallel():
    """Test parallel tracking."""
    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    start = "2005-11-13T18:00"
    end = "2005-11-13T20:00"

    intervals = parallel.get_time_intervals(start, end)
    output_parent = base_local / "runs/cpol_demo_parallel"
    if output_parent.exists():
        shutil.rmtree(output_parent)

    all_options = setup(start, end, output_parent / "options")
    data_options, grid_options, track_options, visualize_options = all_options
    # Note travis crashes at era5 subset step if processes > 1. But can still test
    # parallel tracking with processes=1.
    with logging_listener(), get_context("spawn").Pool(
        initializer=initialize_process, processes=4
    ) as pool:
        results = []
        for i, time_interval in enumerate(intervals):
            args = [i, time_interval, data_options.model_copy(), grid_options.copy()]
            args += [track_options, visualize_options]
            args += [output_parent, "cpol"]
            args = tuple(args)
            results.append(pool.apply_async(parallel.track_interval, args))
        pool.close()
        pool.join()
        parallel.check_results(results)

    parallel.stitch_run(output_parent, intervals)
    analysis_options = analyze.mcs.AnalysisOptions()
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent)

    figure_options = visualize.option.horizontal_attribute_options(
        "mcs_velocity_analysis", style="presentation"
    )
    start_time = np.datetime64("2005-11-13T18:00")
    end_time = np.datetime64("2005-11-13T20:00")
    visualize.attribute.mcs_series(
        output_parent, start_time, end_time, figure_options, parallel_figure=True
    )


if __name__ == "__main__":
    test_parallel()
