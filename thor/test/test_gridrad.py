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


def test_gridrad():
    """
    Test gridrad download and tracking.
    """

    # Parent directory for saving outputs
    base_local = Path("/home/ewan/THOR_output")
    start = "2010-01-20T18:00:00"
    end = "2010-01-20T19:00:00"

    # Create the data_options dictionary
    converted_options = {"save": True, "load": False, "parent_converted": None}
    gridrad_options = data.gridrad.gridrad_data_options(
        start=start, end=end, converted_options=converted_options
    )
    era5_pl_options = data.era5.data_options(start=start, end=end)
    era5_sl_options = data.era5.data_options(
        start=start, end=end, data_format="single-levels"
    )
    data_options = option.consolidate_options(
        [gridrad_options, era5_pl_options, era5_sl_options]
    )

    dispatch.check_data_options(data_options)
    data.option.save_data_options(data_options, filename="gridrad_era5")

    # Create the grid_options dictionary using the first file in the cpol dataset
    grid_options = grid.create_options(
        name="geographic", regrid=False, altitude_spacing=None, geographic_spacing=None
    )
    grid.check_options(grid_options)
    grid.save_grid_options(grid_options, filename="gridrad_geographic")

    # Create the tag_options dictionary
    era5_pl_tag_options = data.era5.tag_options()
    era5_sl_tag_options = data.era5.tag_options(dataset="era5_sl")
    tag_options = option.consolidate_options([era5_pl_tag_options, era5_sl_tag_options])
    tag.save_tag_options(tag_options, filename="era5")

    # Create the track_options dictionary
    track_options = option.mcs(
        dataset="gridrad",
        tags=["era5_pl", "era5_sl"],
        global_flow_margin=70,
        unique_global_flow=False,
    )
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

    output_directory = base_local / "runs/gridrad_demo"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    times = data.utils.generate_times(data_options["gridrad"])
    tracks = track.simultaneous_track(
        times,
        data_options,
        grid_options,
        track_options,
        tag_options,
        visualize_options,
        output_directory=base_local / "runs/gridrad_demo",
    )
