"""Test GridRad tracking."""

from multiprocessing import get_context
import time
from pathlib import Path
import shutil
import numpy as np
import thor.data as data
import thor.data.dispatch as dispatch
import thor.grid as grid
import thor.option as option
import thor.analyze as analyze
import thor.parallel as parallel
import thor.visualize as visualize
from thor.log import setup_logger, logging_listener

logger = setup_logger(__name__)


def gridrad(start, end, event_start, base_local=None):
    # Parent directory for saving outputs
    if base_local is None:
        base_local = Path("/scratch/w40/esh563/THOR_output")

    period = parallel.get_period(start, end)
    intervals = parallel.get_time_intervals(start, end, period=period)

    output_parent = base_local / f"runs/dev/gridrad_{event_start.replace('-', '')}"

    if output_parent.exists():
        shutil.rmtree(output_parent)
    options_directory = output_parent / "options"

    # Create the data_options dictionary
    gridrad_parent = str(base_local / "input_data/raw")
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
    kwargs = {"start": start, "end": end, "data_format": "single-levels"}
    kwargs.update({"parent_local": era5_parent})
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
    track_options.levels[1].objects[0].tracking.global_flow_margin = 70
    track_options.levels[1].objects[0].tracking.unique_global_flow = False
    track_options.to_yaml(options_directory / "track.yml")

    # Create the display_options dictionary
    visualize_options = None

    # 8 processes a good choice for a GADI job with 32 GB of memory, 7 cores
    # Each process can use up to 4 GB of memory - mainly in storing gridrad files
    num_processes = 8
    kwargs = {"initializer": parallel.initialize_process, "processes": num_processes}
    with logging_listener(), get_context("spawn").Pool(**kwargs) as pool:
        results = []
        for i, time_interval in enumerate(intervals):
            args = [i, time_interval, data_options.copy(), grid_options.copy()]
            args += [track_options.model_copy(), visualize_options]
            args += [output_parent, "gridrad"]
            args = tuple(args)
            # Stagger job for smoother execution
            time.sleep(1)
            results.append(pool.apply_async(parallel.track_interval, args))
        pool.close()
        pool.join()
        parallel.check_results(results)

    parallel.stitch_run(output_parent, intervals, cleanup=True)

    analysis_options = analyze.mcs.analysis_options()
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent)
    figure_options = visualize.option.horizontal_attribute_options(
        "mcs_velocity_analysis", style="gadi", attributes=["velocity", "offset"]
    )
    args = [output_parent, start, end, figure_options]
    kwargs = {"parallel_figure": True, "dt": 7200, "by_date": False}
    kwargs.update({"num_processes": num_processes})
    visualize.attribute.mcs_series(*args, **kwargs)


if __name__ == "__main__":
    year = 2010
    event_directories = data.gridrad.get_event_directories(year)
    for event_directory in event_directories[:50]:
        start, end, event_start = data.gridrad.get_event_times(event_directory)
        gridrad(start, end, event_start)
