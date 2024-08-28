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


def add_reflectivity(
    ds,
    lat_center,
    lon_center,
    alt_center,
    radius,
    intensity,
    eccentricity,
    orientation,
):
    # Create a meshgrid for the coordinates
    lon, lat, alt = xr.broadcast(ds.longitude, ds.latitude, ds.altitude)

    # Calculate the rotated coordinates
    x_rot = (lon - lon_center) * np.cos(orientation)
    x_rot += (lat - lat_center) * np.sin(orientation)
    y_rot = -(lon - lon_center) * np.sin(orientation)
    y_rot += (lat - lat_center) * np.cos(orientation)

    # Calculate the distance from the center for each point in the grid, considering eccentricity
    distance = np.sqrt(
        (x_rot / radius) ** 2
        + (y_rot / (radius * eccentricity)) ** 2
        + (alt - alt_center) ** 2
    )

    # Apply a Gaussian function to create an elliptical pattern
    reflectivity = intensity * np.exp(-(distance**2) / 2)
    reflectivity = reflectivity.where(reflectivity >= 0.05 * intensity, np.nan)
    reflectivity = reflectivity.transpose(*ds.dims)

    # Add the generated data to the ds DataArray
    ds.values = xr.where(~np.isnan(reflectivity), reflectivity, ds)
    return ds
