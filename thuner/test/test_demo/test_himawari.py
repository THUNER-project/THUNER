import shutil
import numpy as np
import thuner.data as data
import thuner.option as option
import thuner.track.track as track
import thuner.visualize as visualize
import thuner.analyze as analyze
import thuner.default as default
import thuner.parallel as parallel
import thuner.utils as utils
import thuner.config as config


def test_himawari():
    # # Himawari
    # This tutorial/demo provides a quick and dirty example of how THUNER can be applied to [Himawari](https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f8433_0020_1861_5916) observations.
    # ## Setup
    # Set a flag for whether or not to remove existing output directories
    remove_existing_outputs = True
    # Specify the local base directory for saving outputs
    base_local = config.get_outputs_directory()
    output_parent = base_local / "runs/himawari/"
    options_directory = output_parent / "options"
    visualize_directory = output_parent / "visualize"
    # Remove the output parent directory if it already exists
    if output_parent.exists() and remove_existing_outputs:
        shutil.rmtree(output_parent)
    # Run the cell below to get the demo data for this tutorial, if you haven't already.
    # Download the demo data
    remote_directory = (
        "s3://thuner-storage/THUNER_output/input_data/raw/satellite-products"
    )
    data.get_demo_data(base_local, remote_directory)
    # ## Options
    # Create the dataset options
    start = "2023-01-01T00:00:00"
    # Note the CPOL times are usually a few seconds off the 10 m interval, so add 30 seconds
    # to ensure we capture 19:00:00
    end = "2023-01-02T00:00:00"
    times_dict = {"start": start, "end": end}
    himawari_options = data.himawari.HimawariOptions(**times_dict)
    data_options = option.data.DataOptions(datasets=[himawari_options])
    data_options.to_yaml(options_directory / "data.yml")
    # Setup a grid over New Guinea.
    # Note the demo data contains the full disk, so vary the lat/lon as you like!
    spacing = [0.025, 0.025]
    latitude = np.arange(-10, 0 + spacing[0], spacing[0])
    longitude = np.arange(130, 150 + spacing[1], spacing[1])
    altitude = None
    grid_options = option.grid.GridOptions(
        name="geographic", latitude=latitude, longitude=longitude, altitude=altitude
    )
    grid_options.to_yaml(options_directory / "grid.yml")
    # Create the track_options
    track_options = default.satellite_track(dataset_name="himawari")
    track_options.to_yaml(options_directory / "track.yml")
    # ## Track
    times = utils.generate_times(data_options.dataset_by_name("himawari").filepaths)
    args = [times, data_options, grid_options, track_options]
    parallel.track(
        *args, output_directory=output_parent, dataset_name="himawari", num_processes=2
    )
    # track.track(*args, output_directory=output_parent)
    # ## Analyze/Visualize
    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_yaml(options_directory / "analysis.yml")
    core_filepath = output_parent / "attributes/anvil/core.csv"
    analyze.utils.smooth_flow_velocities(core_filepath, output_parent)
    analyze.utils.quality_control("anvil", output_parent, analysis_options)
    style = "presentation"
    attribute_handlers = default.detected_attribute_handlers(output_parent, style)
    kwargs = {"name": "himawari_anvil", "object_name": "anvil", "style": style}
    kwargs.update({"attribute_handlers": attribute_handlers})
    figure_options = option.visualize.HorizontalAttributeOptions(**kwargs)
    args = [output_parent, start, end, figure_options, "himawari"]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.series(*args, **args_dict)


if __name__ == "__main__":
    test_himawari()
