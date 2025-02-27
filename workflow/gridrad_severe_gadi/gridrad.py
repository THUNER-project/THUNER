"""Test GridRad tracking."""

import subprocess
import argparse
from pathlib import Path
import shutil
import pandas as pd
import thuner.data as data
import thuner.default as default
import thuner.option as option
import thuner.analyze as analyze
import thuner.parallel as parallel
import thuner.visualize as visualize
from thuner.log import setup_logger
import thuner.config as config

logger = setup_logger(__name__)


def gridrad(start, end, event_start, base_local=None):

    if base_local is None:
        base_local = config.get_outputs_directory()

    year = pd.to_datetime(start).year

    event_start_str = event_start.replace("-", "")
    output_parent = base_local / f"runs/gridrad_severe/{year}/gridrad_{event_start_str}"

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
    gridrad_options = data.gridrad.GridRadSevereOptions(**times_dict, **gridrad_dict)
    era5_dict = {"data_format": "pressure-levels", "parent_local": era5_parent}
    era5_pl_options = data.era5.ERA5Options(**times_dict, parent_local=era5_parent)
    era5_dict["data_format"] = "single-levels"
    era5_sl_options = data.era5.ERA5Options(**times_dict, **era5_dict)
    datasets = [gridrad_options, era5_pl_options, era5_sl_options]
    data_options = option.data.DataOptions(datasets=datasets)
    data_options.to_yaml(options_directory / "data.yml")

    # Create the grid_options dictionary
    kwargs = {"name": "geographic", "regrid": False, "altitude_spacing": None}
    kwargs.update({"geographic_spacing": None})
    grid_options = option.grid.GridOptions(**kwargs)
    grid_options.to_yaml(options_directory / "grid.yml")

    # Create the track_options dictionary
    track_options = default.track(dataset="gridrad")
    track_options.levels[1].objects[0].tracking.global_flow_margin = 70
    track_options.levels[1].objects[0].tracking.unique_global_flow = False
    track_options.to_yaml(options_directory / "track.yml")

    # Create the display_options dictionary
    visualize_options = None

    # 8 processes a good choice for a GADI job with 32 GB of memory, 7 cores
    # Each tracking process can use up to 4 GB of memory - mainly storing gridrad data
    num_processes = 8
    times = data.utils.generate_times(data_options.dataset_by_name("gridrad"))
    args = [times, data_options, grid_options, track_options, visualize_options]
    kwargs = {"output_directory": output_parent, "num_processes": num_processes}
    parallel.track(*args, **kwargs)

    analysis_options = analyze.mcs.AnalysisOptions()
    analyze.mcs.process_velocities(output_parent, profile_dataset="era5_pl")
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent, analysis_options)

    figure_name = f"mcs_gridrad_{event_start.replace('-', '')}"
    kwargs = {"style": "gadi", "attributes": ["velocity", "offset"]}
    kwargs.update({"name": figure_name})
    figure_options = option.visualize.HorizontalAttributeOptions(**kwargs)

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
