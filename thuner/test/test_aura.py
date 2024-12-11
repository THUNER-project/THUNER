"""Test setup."""

from pathlib import Path
import shutil
import os
import numpy as np
import thuner.data as data
import thuner.data.dispatch as dispatch
import thuner.grid as grid
import thuner.track as track
import thuner.option as option
import thuner.visualize as visualize
from thuner.log import setup_logger
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

    mcs_vis_options = visualize.option.runtime_options(
        "mcs", save=True, style="presentation", figure_types=["mask", "match"]
    )
    visualize_options = {"mcs": mcs_vis_options}
    visualize.option.save_display_options(visualize_options, options_directory)
    return data_options, grid_options, track_options, visualize_options


def test_cpol_with_runtime_figures_geographic():
    """
    Test cpol download and tracking.
    """

    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    start = "2005-11-13T15:00:00"
    end = "2005-11-13T16:00:00"

    output_directory = base_local / "runs/cpol_demo_geographic_with_runtime_figures"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    options_directory = output_directory / "options"

    options = setup(start, end, options_directory, grid_type="geographic")
    data_options, grid_options, track_options, visualize_options = options

    # Test in geographic coordinates
    times = data.utils.generate_times(data_options.dataset_by_name("cpol"))
    track.simultaneous_track(
        times,
        data_options,
        grid_options,
        track_options,
        visualize_options,
        output_directory=output_directory,
    )


def test_cpol_with_runtime_figures_cartesian():
    """
    Test cpol download and tracking.
    """

    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    start = "2005-11-13T15:00:00"
    end = "2005-11-13T16:00:00"

    # Test in Cartesian coordinates
    output_directory = base_local / "runs/cpol_demo_cartesian_with_runtime_figures"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    options_directory = output_directory / "options"

    options = setup(start, end, options_directory, grid_type="cartesian")
    data_options, grid_options, track_options, visualize_options = options

    times = data.utils.generate_times(data_options.dataset_by_name("cpol"))
    track.simultaneous_track(
        times,
        data_options,
        grid_options,
        track_options,
        visualize_options,
        output_directory=output_directory,
    )


def test_cpol_geographic():
    """
    Test cpol download and tracking.
    """

    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    start = "2005-11-13T14:00:00"
    end = "2005-11-13T16:00:00"

    output_directory = base_local / "runs/cpol_demo_geographic"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    options_directory = output_directory / "options"

    options = setup(start, end, options_directory, grid_type="geographic")
    data_options, grid_options, track_options, visualize_options = options
    visualize_options = None

    # Test tracking in geographic coordinates
    times = data.utils.generate_times(data_options.dataset_by_name("cpol"))
    track.simultaneous_track(
        times,
        data_options,
        grid_options,
        track_options,
        visualize_options,
        output_directory=output_directory,
    )

    analysis_options = analyze.mcs.AnalysisOptions()
    analyze.mcs.process_velocities(output_directory)
    analyze.mcs.quality_control(output_directory, analysis_options)
    analyze.mcs.classify_all(output_directory)
    figure_options = visualize.option.horizontal_attribute_options(
        "mcs_velocity_analysis", style="presentation"
    )
    visualize.attribute.mcs_series(
        output_directory, start, end, figure_options, parallel_figure=True
    )


def test_cpol_cartesian():
    """
    Test cpol download and tracking.
    """

    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    start = "2005-11-13T14:00:00"
    end = "2005-11-13T16:00:00"

    # Test tracking in Cartesian coordinates
    output_directory = base_local / "runs/cpol_demo_cartesian"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    options_directory = output_directory / "options"

    options = setup(start, end, options_directory, grid_type="cartesian")
    data_options, grid_options, track_options, visualize_options = options
    visualize_options = None

    times = data.utils.generate_times(data_options.dataset_by_name("cpol"))
    track.simultaneous_track(
        times,
        data_options,
        grid_options,
        track_options,
        visualize_options,
        output_directory=output_directory,
    )

    analysis_options = analyze.mcs.AnalysisOptions()
    analyze.mcs.process_velocities(output_directory)
    analyze.mcs.quality_control(output_directory, analysis_options)
    analyze.mcs.classify_all(output_directory)
    figure_options = visualize.option.horizontal_attribute_options(
        "mcs_velocity_analysis", style="presentation"
    )
    visualize.attribute.mcs_series(
        output_directory, start, end, figure_options, parallel_figure=True
    )


if __name__ == "__main__":
    # test_cpol_geographic()
    # test_cpol_cartesian()
    test_cpol_with_runtime_figures_geographic()
    test_cpol_with_runtime_figures_cartesian()
