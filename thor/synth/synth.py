"""
Module for generating synthetic reflectivity data for testing. 
"""

import numpy as np
import xarray as xr


def dataset(
    grid_options,
    start_time="2020-01-01T00:00:00",
    end_time="2020-01-01T03:00:00",
    time_step=10,
):
    """
    Generate synthetic reflectivity data for testing.

    Parameters
    ----------
    grid_options : dict
        Dictionary containing the grid options.

    Returns
    -------
    dataset : dict
        Dictionary containing the synthetic reflectivity data.
    """

    lat = np.array(grid_options["latitude"])
    lon = np.array(grid_options["longitude"])
    alt = np.array(grid_options["altitude"])
    time = np.arange(
        np.datetime64(start_time),
        np.datetime64(end_time),
        np.timedelta64(time_step),
    )

    ds_values = np.ones((len(alt), len(lat), len(lon))) * np.nan
    ds = xr.DataArray(
        ds_values,
        coords=[("altitude", alt), ("latitude", lat), ("longitude", lon)],
    )
    return ds


def cell(lat, lon, alt, time, radius, intensity, eccentricity, orientation):

    # Generate synthetic reflectivity data for a cell object using gaussian

    


def anvil():
    pass
