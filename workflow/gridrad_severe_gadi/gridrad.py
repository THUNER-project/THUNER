"""Test GridRad tracking."""

import subprocess
import argparse
from multiprocessing import get_context
import time
from pathlib import Path
import shutil
import thuner.data as data
import thuner.data.dispatch as dispatch
import thuner.grid as grid
import thuner.option as option
import thuner.analyze as analyze
import thuner.parallel as parallel
import thuner.visualize as visualize
from thuner.log import setup_logger, logging_listener
import thuner.config as config

logger = setup_logger(__name__)


def gridrad(start, end, event_start, base_local=None):

    if base_local is None:
        base_local = config.get_outputs_directory()

    period = parallel.get_period(start, end)
    intervals = parallel.get_time_intervals(start, end, period=period)

    event_start_str = event_start.replace("-", "")
    output_parent = base_local / f"runs/gridrad_severe/gridrad_{event_start_str}"

    # Check if tar.gz exists; if so return
    tar_file = f"{output_parent}.tar.gz"
    if Path(tar_file).exists():
        logger.info(f"Event {event_start} already tracked.")
        return

    if output_parent.exists():
        shutil.rmtree(output_parent)
    output_parent.mkdir(parents=True, exist_ok=True)
    options_directory = output_parent / "options"
    options_directory.mkdir(parents=True, exist_ok=True)

    # Create the data_options dictionary
    gridrad_parent = str(base_local / "input_data/raw")
    era5_parent = "/g/data/rt52"

    # Create and save the dataset options
    times_dict = {"start": start, "end": end}
    gridrad_dict = {"event_start": event_start, "parent_local": gridrad_parent}
    gridrad_options = data.option.GridRadSevereOptions(**times_dict, **gridrad_dict)
    era5_dict = {"data_format": "pressure-levels", "parent_local": era5_parent}
    era5_pl_options = data.option.ERA5Options(**times_dict, parent_local=era5_parent)
    era5_dict["data_format"] = "single-levels"
    era5_sl_options = data.option.ERA5Options(**times_dict, **era5_dict)
    data_options = data.option.DataOptions(
        datasets=[gridrad_options, era5_pl_options, era5_sl_options]
    )
    data_options.to_yaml(options_directory / "data.yml")
    gridrad_options = data_options.dataset_by_name("gridrad")

    # Create the grid_options dictionary
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
    # Each tracking process can use up to 4 GB of memory - mainly storing gridrad data
    num_processes = 8
    kwargs = {"initializer": parallel.initialize_process, "processes": num_processes}
    with logging_listener(), get_context("spawn").Pool(**kwargs) as pool:
        results = []
        for i, time_interval in enumerate(intervals):
            args = [i, time_interval, data_options.model_copy(), grid_options.copy()]
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

    analysis_options = analyze.mcs.AnalysisOptions()
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent)

    figure_name = f"mcs_gridrad_{event_start.replace('-', '')}"
    figure_options = visualize.option.horizontal_attribute_options(
        figure_name, style="gadi", attributes=["velocity", "offset"]
    )
    args = [output_parent, start, end, figure_options]
    kwargs = {"parallel_figure": True, "dt": 7200, "by_date": False}
    # Halving the number of processes used for figure creation appears to be a good
    # rule of thumb. Even with rasterization etc, the largest, most complex figures can
    # still consume nearly 6 GB during plt.savefig!
    kwargs.update({"num_processes": int(num_processes / 2)})
    visualize.attribute.mcs_series(*args, **kwargs)

    # Tar and compress the output directory
    output_parent = Path(output_parent)
    tar_file = f"{output_parent}.tar.gz"
    # Remove the tar file if it already exists
    if Path(tar_file).exists():
        Path(tar_file).unlink()
    command = f"tar -czvf {tar_file} -C {output_parent.parent} {output_parent.name}"
    subprocess.run(command, shell=True, text=True)


if __name__ == "__main__":

    # Parse input arguments
    parser = argparse.ArgumentParser(description="Track GridRad event on GADI")
    parser.add_argument("event_directory", type=str, help="Directory of event files")
    args = parser.parse_args()
    event_directory = Path(args.event_directory)

    start, end, event_start = data.gridrad.get_event_times(event_directory)
    try:
        gridrad(start, end, event_start)
    except Exception as e:
        logger.error(f"Error tracking event {str(event_start)}: {e}")
