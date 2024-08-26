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
import thor.tag as tag
import thor.visualize as visualize
from thor.log import setup_logger

logger = setup_logger(__name__)


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
    era5_pl_options = data.era5.data_options(start=start, end=end)
    era5_sl_options = data.era5.data_options(
        start=start, end=end, data_format="single-levels"
    )
    data_options = option.consolidate_options(
        [cpol_options, era5_pl_options, era5_sl_options]
    )

    dispatch.check_data_options(data_options)
    data.option.save_data_options(data_options, filename="cpol_era5")

    grid_options = grid.create_options(name="geographic")
    grid.check_options(grid_options)
    grid.save_grid_options(grid_options, filename="cpol_geographic")

    # Create the tag_options dictionary
    era5_pl_tag_options = data.era5.tag_options()
    era5_sl_tag_options = data.era5.tag_options(dataset="era5_sl")
    tag_options = option.consolidate_options([era5_pl_tag_options, era5_sl_tag_options])
    tag.save_tag_options(tag_options, filename="era5")

    # Create the track_options dictionary
    track_options = option.mcs(dataset="cpol", tags=["era5_pl", "era5_sl"])
    option.save_track_options(track_options, filename="cpol_mcs")

    # Create the display_options dictionary
    visualize_options = {
        obj: visualize.option.runtime_options(obj, save=True, style="presentation")
        for obj in ["cell", "anvil", "mcs"]
    }
    visualize_options["middle_cloud"] = visualize.option.runtime_options(
        "middle_cloud", save=True, style="presentation", figure_types=["mask"]
    )
    visualize.option.save_display_options(visualize_options, filename="runtime_mcs")

    # Test tracking in geographic coordinates
    output_directory = base_local / "runs/cpol_demo_geographic"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    times = data.utils.generate_times(data_options["cpol"])
    track.simultaneous_track(
        times,
        data_options,
        grid_options,
        track_options,
        tag_options,
        visualize_options,
        output_directory=output_directory,
    )

    # Test tracking in Cartesian coordinates
    grid_options = grid.create_options(name="cartesian", regrid=True)
    grid.check_options(grid_options)
    grid.save_grid_options(grid_options, filename="cpol_cartesian")

    output_directory = base_local / "runs/cpol_cartesian_demo"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    times = data.utils.generate_times(data_options["cpol"])
    tracks = track.simultaneous_track(
        times,
        data_options,
        grid_options,
        track_options,
        tag_options,
        visualize_options,
        output_directory=base_local / "runs/cpol_cartesian_demo",
    )
