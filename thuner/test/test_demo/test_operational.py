import shutil
import glob
import xarray as xr
from pathlib import Path
import glob
import numpy as np
import thuner.data as data
import thuner.option as option
import thuner.track.track as track
import thuner.visualize as visualize
import thuner.analyze as analyze
import thuner.default as default
import thuner.attribute as attribute
import thuner.parallel as parallel
import thuner.utils as utils
import thuner.config as config


def test_operational():
    # # Operational Radar
    # ## Setup
    # Set a flag for whether or not to remove existing output directories
    remove_existing_outputs = True
    # Specify the local base directory for saving outputs
    base_local = config.get_outputs_directory()
    output_parent = base_local / "runs/operational/geographic"
    options_directory = output_parent / "options"
    visualize_directory = output_parent / "visualize"
    # Remove the output parent directory if it already exists
    if output_parent.exists() and remove_existing_outputs:
        shutil.rmtree(output_parent)
    # # Download the demo data
    # remote_directory = "s3://thuner-storage/THUNER_output/input_data/raw/operational"
    # data.get_demo_data(base_local, remote_directory)
    # remote_directory = "s3://thuner-storage/THUNER_output/input_data/raw/"
    # remote_directory += "era5_monthly_10S_129E_14S_133E"
    # data.get_demo_data(base_local, remote_directory)
    # ## Geographic Coordinates
    # Create the dataset options
    start = "2021-10-14T16:00:00"
    # Note the CPOL times are usually a few seconds off the 10 m interval, so add 30 seconds
    end = "2021-10-14T17:00:00"
    times_dict = {"start": start, "end": end}
    weights_filepaths = glob.glob(
        str(output_parent / "regridder_weights/operational*.nc")
    )
    for weights_filepath in weights_filepaths:
        Path(weights_filepath).unlink(missing_ok=True)
    weights_filepath = str(output_parent / "regridder_weights/operational.nc")
    operational_ensemble_options = data.aura.OperationalEnsembleOptions(
        **times_dict,
        weights_filepath=weights_filepath,
        radars=[3, 4, 40, 54, 69, 71, 96],
    )
    operational_ensemble_options.converted_options.save = True
    datasets = [operational_ensemble_options]
    data_options = option.data.DataOptions(datasets=datasets)
    data_options.to_json(options_directory / "data.json")
    # Create the grid_options
    grid_options = option.grid.GridOptions()
    grid_options.to_json(options_directory / "grid.json")
    # Create the track_options
    track_options = default.track.track(
        dataset_name="operational", profile_dataset=None, tag_dataset=None
    )
    track_options.levels[1].objects[0].tracking.unique_global_flow = False
    track_options.levels[1].objects[0].tracking.global_flow_margin = 70
    track_options.levels[1].objects[0].revalidate()
    track_options.to_json(options_directory / "track.json")
    step = np.timedelta64(data_options.datasets[0].timestep, "m")
    times = np.arange(np.datetime64(start), np.datetime64(end) + step, step)
    track.track(
        times=times,
        data_options=data_options,
        grid_options=grid_options,
        track_options=track_options,
        visualize_options=None,
        output_directory=output_parent,
    )
    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_json(options_directory / "analysis.json")
    analyze.utils.smooth_flow_velocities("mcs", output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    style = "presentation"
    attribute_handlers = default.visualize.grouped_attribute_handlers(
        output_parent, style
    )
    figure_options = option.visualize.GroupedHorizontalAttributeOptions(
        name="mcs_attributes",
        object_name="mcs",
        style=style,
        attribute_handlers=attribute_handlers,
    )
    visualize.attribute.series(
        output_directory=output_parent,
        start_time=start,
        end_time=end,
        figure_options=figure_options,
        dataset_name="operational",
        parallel_figure=False,
        by_date=False,
        num_processes=4,
    )


if __name__ == "__main__":
    test_operational()
