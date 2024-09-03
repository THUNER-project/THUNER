"""Test setup."""

from pathlib import Path
import shutil
import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import thor.data as data
import thor.data.dispatch as dispatch
import thor.grid as grid
import thor.track as track
import thor.option as option
import thor.visualize as visualize
from thor.log import setup_logger

logger = setup_logger(__name__)


def test_cpol_with_runtime_figures():
    """
    Test cpol download and tracking.
    """

    # Parent directory for saving outputs
    base_local = Path.home() / "THOR_output"
    start = "2005-11-13T15:00:00"
    end = "2005-11-13T16:00:00"

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
    data.option.save_data_options(data_options, filename="cpol_era5")

    grid_options = grid.create_options(name="geographic")
    grid.check_options(grid_options)
    grid.save_grid_options(grid_options, filename="cpol_geographic")

    # Create the track_options dictionary
    track_options = option.mcs(dataset="cpol")
    option.check_options(track_options)
    option.save_track_options(track_options, filename="gridrad_mcs")

    # Create the display_options dictionary
    cell_vis_options = visualize.option.runtime_options(
        "cell", save=True, style="presentation", figure_types=["mask", "match"]
    )
    anvil_vis_options = visualize.option.runtime_options(
        "anvil", save=True, style="presentation", figure_types=["mask", "match"]
    )
    mcs_vis_options = visualize.option.runtime_options(
        "mcs", save=True, style="presentation", figure_types=["mask", "match"]
    )
    visualize_options = {
        "cell": cell_vis_options,
        "anvil": anvil_vis_options,
        "mcs": mcs_vis_options,
    }
    visualize.option.save_display_options(visualize_options, filename="runtime_mcs")

    # Test in geographic coordinates s
    output_directory = base_local / "runs/cpol_demo_geographic_with_runtime_figures"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    times = data.utils.generate_times(data_options["cpol"])
    track.simultaneous_track(
        times,
        data_options,
        grid_options,
        track_options,
        visualize_options,
        output_directory=output_directory,
    )

    # Test in Cartesian coordinates
    grid_options = grid.create_options(name="cartesian", regrid=True)
    grid.check_options(grid_options)
    grid.save_grid_options(grid_options, filename="cpol_cartesian")

    output_directory = base_local / "runs/cpol_demo_cartesian_with_runtime_figures"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    times = data.utils.generate_times(data_options["cpol"])
    track.simultaneous_track(
        times,
        data_options,
        grid_options,
        track_options,
        visualize_options,
        output_directory=output_directory,
    )


def test_cpol():
    """
    Test cpol download and tracking.
    """

    # Parent directory for saving outputs
    base_local = Path.home() / "THOR_output"
    start = "2005-11-13T13:00:00"
    end = "2005-11-13T19:00:00"

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
    data.option.save_data_options(data_options, filename="cpol_era5")

    grid_options = grid.create_options(name="geographic")
    grid.check_options(grid_options)
    grid.save_grid_options(grid_options, filename="cpol_geographic")

    # Create the track_options dictionary
    track_options = option.mcs(dataset="cpol")
    option.check_options(track_options)
    option.save_track_options(track_options, filename="gridrad_mcs")

    visualize_options = None

    # Test tracking in geographic coordinates s
    output_directory = base_local / "runs/cpol_demo_geographic"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    times = data.utils.generate_times(data_options["cpol"])
    track.simultaneous_track(
        times,
        data_options,
        grid_options,
        track_options,
        visualize_options,
        output_directory=output_directory,
    )

    # Test tracking in Cartesian coordinates
    grid_options = grid.create_options(name="cartesian", regrid=True)
    grid.check_options(grid_options)
    grid.save_grid_options(grid_options, filename="cpol_cartesian")

    output_directory = base_local / "runs/cpol_demo_cartesian"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    times = data.utils.generate_times(data_options["cpol"])
    track.simultaneous_track(
        times,
        data_options,
        grid_options,
        track_options,
        visualize_options,
        output_directory=output_directory,
    )
