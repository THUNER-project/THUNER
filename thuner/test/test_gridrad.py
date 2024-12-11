"""Test GridRad tracking."""

import os
from multiprocessing import get_context
from pathlib import Path
import shutil
import numpy as np
import thuner.data as data
import thuner.data.dispatch as dispatch
import thuner.grid as grid
import thuner.option as option
import thuner.analyze as analyze
import thuner.parallel as parallel
import thuner.visualize as visualize
import thuner.log as log
import thuner.track as track

notebook_name = "gridrad_demo.ipynb"

logger = log.setup_logger(__name__)


def test_gridrad():
    # Parent directory for saving outputs
    logger.info("Setting up GridRad test.")
    base_local = Path.home() / "THUNER_output"
    start = "2010-01-20T21:00:00"
    end = "2010-01-20T23:00:00"
    event_start = "2010-01-20"

    period = parallel.get_period(start, end)
    intervals = parallel.get_time_intervals(start, end, period=period)

    output_parent = base_local / "runs/gridrad_demo"
    if output_parent.exists():
        shutil.rmtree(output_parent)
    options_directory = output_parent / "options"

    # Create the data_options dictionary
    converted_options = {"save": True, "load": False, "parent_converted": None}
    gridrad_options = data.gridrad.gridrad_data_options(
        start=start,
        end=end,
        converted_options=converted_options,
        event_start=event_start,
    )

    lon_range = [-102, -89]
    lat_range = [27, 39]
    era5_pl_options = data.era5.data_options(
        start=start, end=end, longitude_range=lon_range, latitude_range=lat_range
    )
    kwargs = {"start": start, "end": end, "data_format": "single-levels"}
    kwargs.update({"longitude_range": lon_range, "latitude_range": lat_range})
    era5_sl_options = data.era5.data_options(**kwargs)

    data_options = option.consolidate_options(
        [gridrad_options, era5_pl_options, era5_sl_options]
    )

    dispatch.check_data_options(data_options)
    data.option.save_data_options(data_options, options_directory=options_directory)

    # Create the grid_options dictionary using the first file in the cpol dataset
    grid_options = grid.create_options(
        name="geographic", regrid=False, altitude_spacing=None, geographic_spacing=None
    )
    grid.check_options(grid_options)
    grid.save_grid_options(grid_options, options_directory=options_directory)

    # Create the track_options dictionary
    track_options = option.default_track_options(dataset="gridrad")
    # Modify the default options for gridrad. Because grids so large we now use a distinct
    # global flow box for each object.
    track_options.levels[1].objects[0].tracking.global_flow_margin = 70
    track_options.levels[1].objects[0].tracking.unique_global_flow = False
    track_options.to_yaml(options_directory / "track.yml")

    # Create the display_options dictionary
    visualize_options = {
        obj: visualize.option.runtime_options(obj, save=True, style="presentation")
        for obj in ["mcs"]
    }
    visualize_options = None

    num_processes = int(0.75 * os.cpu_count())
    logger.info(f"Using {num_processes} processes for tracking.")
    with get_context("spawn").Pool(
        initializer=parallel.initialize_process, processes=num_processes
    ) as pool:
        results = []
        for i, time_interval in enumerate(intervals):
            args = [i, time_interval, data_options.model_copy(), grid_options.copy()]
            args += [track_options, visualize_options]
            args += [output_parent, "gridrad"]
            args = tuple(args)
            results.append(pool.apply_async(parallel.track_interval, args))
        pool.close()
        pool.join()
        parallel.check_results(results)

    parallel.stitch_run(output_parent, intervals, cleanup=True)

    analysis_options = analyze.mcs.AnalysisOptions()
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent)
    figure_options = visualize.option.horizontal_attribute_options(
        "mcs_velocity_analysis", style="presentation", attributes=["velocity", "offset"]
    )
    start_time = np.datetime64("2010-01-20T18:00")
    end_time = np.datetime64(np.datetime64("2010-01-21T03:30"))
    args = [output_parent, start_time, end_time, figure_options]
    kwargs = {"parallel_figure": True, "dt": 5400, "by_date": False}
    visualize.attribute.mcs_series(*args, **kwargs)


if __name__ == "__main__":
    test_gridrad()
