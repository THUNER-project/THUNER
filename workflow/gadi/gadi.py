"""Test GridRad tracking."""

import concurrent.futures
from pathlib import Path
import shutil
import numpy as np
import thor.data as data
import thor.data.dispatch as dispatch
import thor.grid as grid
import thor.option as option
import thor.track as track
import thor.analyze as analyze
import thor.parallel as parallel
import thor.visualize as visualize
from thor.log import setup_logger

logger = setup_logger(__name__)


def gridrad():
    # Parent directory for saving outputs
    base_local = Path("/scratch/w40/esh563/THOR_output")
    start = "2010-01-20T18:00:00"
    end = "2010-01-21T03:30:00"
    event_start = "2010-01-20"

    period = parallel.get_period(start, end)
    intervals = parallel.get_time_intervals(start, end, period=period)

    output_parent = base_local / "runs/gridrad_demo"
    if output_parent.exists():
        shutil.rmtree(output_parent)
    options_directory = output_parent / "options"

    # Create the data_options dictionary
    gridrad_parent = base_local / "input_data/raw"
    converted_options = {"save": True, "load": False, "parent_converted": None}
    gridrad_options = data.gridrad.gridrad_data_options(
        start=start,
        end=end,
        converted_options=converted_options,
        event_start=event_start,
        parent_local=gridrad_parent,
    )

    era5_parent = "/g/data/rt52"

    era5_pl_options = data.era5.data_options(
        start=start, end=end, parent_local=era5_parent
    )
    args_dict = {"start": start, "end": end, "data_format": "single-levels"}
    args_dict.update({"parent_local": era5_parent})
    era5_sl_options = data.era5.data_options(**args_dict)

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
    track_options = option.mcs(
        dataset="gridrad",
        global_flow_margin=70,
        unique_global_flow=False,
    )

    option.check_options(track_options)
    option.save_track_options(track_options, options_directory=options_directory)

    # Create the display_options dictionary
    visualize_options = None

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i, time_interval in enumerate(intervals):
            args = [i, time_interval, data_options.copy(), grid_options.copy()]
            args += [track_options.copy(), visualize_options]
            args += [output_parent, "gridrad"]
            futures.append(executor.submit(parallel.track_interval, *args))
        parallel.check_futures(futures)
    parallel.stitch_run(output_parent, intervals, cleanup=True)

    analysis_options = analyze.mcs.analysis_options()
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent)
    figure_options = visualize.option.horizontal_attribute_options(
        "mcs_velocity_analysis", style="presentation", attributes=["velocity", "offset"]
    )
    start_time = np.datetime64("2010-01-20T18:00")
    end_time = np.datetime64(np.datetime64("2010-01-21T03:30"))
    args = [output_parent, start_time, end_time, figure_options]
    args_dict = {"parallel_figure": True, "dt": 5400, "by_date": False}
    visualize.attribute.mcs_series(*args, **args_dict)
