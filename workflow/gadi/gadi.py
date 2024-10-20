"""Test GridRad tracking."""

from multiprocessing import get_context
import time
import os
import concurrent.futures
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


def download_data(parent_local="/scratch/w40/esh563/THOR_output/input_data/raw"):
    # Load list of urls from file

    parent_remote = "https://data.rda.ucar.edu"

    # Load list of urls from file
    urls = []
    with open("./extracted_urls.txt", "r") as f:
        urls = [line.strip() for line in f]
    # Download data

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for url in urls:
            time.sleep(0.1)
            data.utils.download(url, parent_remote, parent_local)
        parallel.check_futures(futures)


def gridrad():
    # Parent directory for saving outputs
    base_local = Path("/scratch/w40/esh563/THOR_output")
    start = "2010-01-21T12:00:00"
    end = "2010-01-21T16:00:00"
    event_start = "2010-01-21"

    period = parallel.get_period(start, end)
    intervals = parallel.get_time_intervals(start, end, period=period)

    output_parent = base_local / "runs/gridrad_demo"
    # if output_parent.exists():
    #     shutil.rmtree(output_parent)
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
    track_options = option.default_track_options(dataset="gridrad")
    track_options.levels[1].objects[0].tracking.global_flow_margin = 70
    track_options.levels[1].objects[0].tracking.unique_global_flow = False
    track_options.to_yaml(options_directory / "track.yml")

    # Create the display_options dictionary
    visualize_options = None

    # num_processes = 4
    # with logging_listener(), get_context("spawn").Pool(
    #     initializer=parallel.initialize_process, processes=num_processes
    # ) as pool:
    #     results = []
    #     for i, time_interval in enumerate(intervals):
    #         args = [i, time_interval, data_options.copy(), grid_options.copy()]
    #         args += [track_options.copy(), visualize_options]
    #         args += [output_parent, "gridrad"]
    #         args = tuple(args)
    #         time.sleep(1)
    #         results.append(pool.apply_async(parallel.track_interval, args))
    #     pool.close()
    #     pool.join()
    #     parallel.check_results(results)

    # parallel.stitch_run(output_parent, intervals, cleanup=True)

def plot(output_parent):
    
    analysis_options = analyze.mcs.analysis_options()
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent)
    figure_options = visualize.option.horizontal_attribute_options(
        "mcs_velocity_analysis", style="gadi", attributes=["velocity", "offset"]
    )
    start_time = np.datetime64("2010-01-21T15:50")
    end_time = np.datetime64(np.datetime64("2010-01-21T16:00"))
    args = [output_parent, start_time, end_time, figure_options]
    args_dict = {"parallel_figure": True, "dt": 5400, "by_date": False}
    visualize.attribute.mcs_series(*args, **args_dict)


if __name__ == "__main__":
    # gridrad()
    plot(Path("/scratch/w40/esh563/THOR_output/runs/gridrad_demo"))


