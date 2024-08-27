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
    end = "2005-11-13T14:00:00"

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
    # Disable tracking/matching for cell and anvil objects
    track_options[0]["cell"]["tracking"] = {"method": None}
    track_options[0]["anvil"]["tracking"] = {"method": None}
    option.check_options(track_options)
    option.save_track_options(track_options, filename="gridrad_mcs")

    # Create the display_options dictionary
    cell_vis_options = visualize.option.runtime_options(
        "cell", save=True, style="presentation", figure_types=["mask"]
    )
    anvil_vis_options = visualize.option.runtime_options(
        "anvil", save=True, style="presentation", figure_types=["mask"]
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

    output_directory = base_local / "runs/cpol_demo_cartesian"
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
        output_directory=output_directory,
    )
