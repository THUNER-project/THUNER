"""Deal with thor grid objects"""

from pathlib import Path
import yaml
import inspect
import numpy as np
from pyproj import Geod
from thor.utils import almost_equal, pad
from thor.log import setup_logger
from thor.option import save_options


logger = setup_logger(__name__)


def create_options(
    name="geographic",
    timestep=None,
    start_latitude=None,
    end_latitude=None,
    start_longitude=None,
    end_longitude=None,
    central_latitude=None,
    central_longitude=None,
    projection=None,
    start_x=None,
    end_x=None,
    start_y=None,
    end_y=None,
    start_alt=0,
    end_alt=25e3,
    cartesian_spacing=[500, 2500, 2500],
    geographic_spacing=[500, 0.025, 0.025],
    regrid=True,
    save=False,
    **kwargs,
):
    """
    Generate grid options dictionary.

    Parameters
    ----------
    timestep : int, optional
        Time step for the dataset; default is None.
    start_latitude : float, optional
        Starting latitude for the dataset; default is None.
    end_latitude : float, optional
        Ending latitude for the dataset; default is None.
    start_longitude : float, optional
        Starting longitude for the dataset; default is None.
    end_longitude : float, optional
        Ending longitude for the dataset; default is None.
    central_latitude : float, optional
        Central latitude for the dataset; default is None.
    central_longitude : float, optional
        Central longitude for the dataset; default is None.
    projection : str, optional
        Projection for the dataset; default is None.
    start_x : float, optional
        Starting x-coordinate for the dataset; default is None.
    end_x : float, optional
        Ending x-coordinate for the dataset; default is None.
    start_y : float, optional
        Starting y-coordinate for the dataset; default is None.
    end_y : float, optional
        Ending y-coordinate for the dataset; default is None.
    start_alt : float, optional
        Starting z-coordinate for the dataset; default is 0.
    end_alt : float, optional
        Ending z-coordinate for the dataset; default is 25e3.
    cartesian_spacing : list, optional
        Spacing for the cartesian grid [z, y, x] in metres; default is
        [500, 2500, 2500].
    geographic_spacing : list, optional
        Spacing for the geographic grid [z, lat, lon] in metres and
        degrees; default is [500, 0.025, 0.025].
    regrid : bool, optional
        Whether to regrid the dataset; default is True.
    save : bool, optional
        Whether to save the dataset; default is False.
    **kwargs

    Returns
    -------
    options : dict
        Dictionary containing the grid options.
    """

    options = {
        "name": name,
        "timestep": timestep,
        "start_latitude": start_latitude,
        "end_latitude": end_latitude,
        "start_longitude": start_longitude,
        "end_longitude": end_longitude,
        "central_latitude": central_latitude,
        "central_longitude": central_longitude,
        "projection": projection,
        "start_x": start_x,
        "end_x": end_x,
        "start_y": start_y,
        "end_y": end_y,
        "start_alt": start_alt,
        "end_alt": end_alt,
        "cartesian_spacing": cartesian_spacing,
        "geographic_spacing": geographic_spacing,
        "regrid": regrid,
        "save": save,
    }

    for key, value in kwargs.items():
        options[key] = value

    if save:
        filepath = Path(__file__).parent / "option/default/grid.yaml"
        with open(filepath, "w") as outfile:
            yaml.dump(
                options,
                outfile,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    return options


def save_grid_options(
    grid_options, filename=None, options_directory=None, append_time=False
):
    if options_directory is None:
        options_directory = Path(__file__).parent / "options/grid_options"
    if filename is None:
        filename = "grid_options"
        append_time = True
    logger.debug(f"Saving grid options to {options_directory / filename}")
    save_options(grid_options, filename, options_directory, append_time=append_time)


def check_options(options):
    """
    Check the input options.

    Parameters
    ----------
    options : dict
        Dictionary containing the input options.


    Returns
    -------
    options : dict
        Dictionary containing the input options.
    """

    for key in inspect.getfullargspec(create_options).args:
        if key not in options.keys():
            raise ValueError(f"Missing required key {key}")

    return options


def new_geographic_grid(latitudes, longitudes, grid_options):
    """
    Get the geographic grid.

    Parameters
    ----------
    longitudes : numpy.ndarray
        Array of longitudes, ascending.
    latitudes : numpy.ndarray
        Array of latitudes, ascending.
    grid_options : dict
        Dictionary containing the grid options.

    Returns
    -------
    tuple
        The geographic grid as a tuple of (lons, lats).
    """

    if grid_options["name"] != "geographic":
        raise ValueError("grid_options['name'] must be 'geographic'.")

    if grid_options["geographic_spacing"] is None:
        raise ValueError("grid_options['geographic_spacing'] must be defined.")

    spacing = grid_options["geographic_spacing"]

    altitude = np.arange(
        grid_options["start_alt"], grid_options["end_alt"] + spacing[0], spacing[0]
    )

    new_latitudes = np.arange(
        np.ceil(latitudes.max(axis=1).min() / spacing[1]) * spacing[1],
        np.floor(latitudes.min(axis=1).max() / spacing[1]) * spacing[1],
        spacing[1],
    )

    new_longitudes = np.arange(
        np.ceil(longitudes.max(axis=0).min() / spacing[2]) * spacing[2],
        np.floor(longitudes.min(axis=0).max() / spacing[2]) * spacing[2],
        spacing[2],
    )

    return altitude, new_latitudes, new_longitudes


def get_area_elements(latitudes, longitudes):

    geod = Geod(ellps="WGS84")
    d_lon = longitudes[1:] - longitudes[:-1]
    d_lat = latitudes[1:] - latitudes[:-1]

    distance = np.vectorize(
        lambda lon1, lat1, lon2, lat2: geod.inv(lon1, lat1, lon2, lat2)[2]
    )

    if almost_equal(d_lon, 5) and almost_equal(d_lat, 5):

        dx = distance(longitudes[2], latitudes, longitudes[0], latitudes) / 2
        dy = distance(longitudes[0], latitudes[2:], longitudes[0], latitudes[:-2]) / 2
        dy = pad(dy)

        areas = dx * dy
        areas = np.tile(areas, (len(longitudes), 1)).T
    else:
        logger.warning("Irregular lat/lon grid. May be slow to calculate areas.")
        LONS, LATS = np.meshgrid(longitudes, latitudes)
        dx = distance(
            LONS[1:-1, 2:], LATS[1:-1, 1:-1], LONS[1:-1, :-2], LATS[1:-1, 1:-1]
        )
        dx = dx / 2
        dy = distance(
            LONS[1:-1, 1:-1], LATS[2:, 1:-1], LONS[1:-1, 1:-1], LATS[:-2, 1:-1]
        )
        dy = dy / 2
        areas = dx * dy
        areas = np.apply_along_axis(pad, axis=0, arr=areas)
        areas = np.apply_along_axis(pad, axis=1, arr=areas)

    return areas, dx, dy
