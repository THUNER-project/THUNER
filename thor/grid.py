"""Deal with thor grid objects"""

from pathlib import Path
import yaml
import inspect
import numpy as np
from thor.utils import almost_equal, pad, geodesic_distance
from thor.config import get_outputs_directory
from thor.log import setup_logger
from thor.option import save_options


logger = setup_logger(__name__)


def create_options(
    name="geographic",
    altitude=None,
    latitude=None,
    longitude=None,
    x=None,
    y=None,
    projection=None,
    altitude_spacing=500,
    cartesian_spacing=[2500, 2500],
    geographic_spacing=[0.025, 0.025],
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

    if altitude is None:
        altitude = list(np.arange(0, 25e3 + altitude_spacing, altitude_spacing))

    options = {
        "name": name,
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "x": x,
        "y": y,
        "projection": projection,
        "altitude_spacing": altitude_spacing,
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
        options_directory = get_outputs_directory() / "options/grid_options"
    if filename is None:
        filename = grid_options["name"]
        append_time = True
    save_options(grid_options, filename, options_directory, append_time=append_time)


def check_spacing(array, dx):
    """Check if array equally spaced."""
    if not almost_equal(np.diff(array)):
        raise ValueError("Grid not equally spaced.")
    elif not almost_equal(list(np.diff(array)) + [dx]):
        raise ValueError("Grid spacing does not match prescribed gridlengths.")


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

    if options["name"] == "cartesian":
        [x, y, z] = [options[var] for var in ["x", "y", "altitude"]]
        spacing = options["cartesian_spacing"]
    elif options["name"] == "geographic":
        [x, y, z] = [options[var] for var in ["longitude", "latitude", "altitude"]]
        spacing = options["geographic_spacing"]
    [dx, dy, dz] = [spacing[1], spacing[0], options["altitude_spacing"]]

    if x is None or y is None or z is None:
        raise ValueError("Missing required key x, y, or z.")
    else:
        [check_spacing(v, dv) for v, dv in zip([x, y, z], [dx, dy, dz])]

    for key in inspect.getfullargspec(create_options).args:
        if key not in options.keys():
            raise ValueError(f"Missing required key {key}")

    return options


def new_geographic_grid(latitudes, longitudes, dlat, dlon):
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

    min_lat = np.floor(latitudes.min() / dlat) * dlat
    max_lat = np.ceil(latitudes.max() / dlat) * dlat
    min_lon = np.floor(longitudes.min() / dlon) * dlon
    max_lon = np.ceil(longitudes.max() / dlon) * dlon
    new_latitudes = np.arange(min_lat, max_lat + dlat, dlat)
    new_longitudes = np.arange(min_lon, max_lon + dlon, dlon)

    return list(new_latitudes), list(new_longitudes)


def get_cell_areas(latitudes, longitudes):
    """Get cell areas in km^2."""

    d_lon = longitudes[1:] - longitudes[:-1]
    d_lat = latitudes[1:] - latitudes[:-1]

    if almost_equal(d_lon, 5) and almost_equal(d_lat, 5):

        dx = geodesic_distance(longitudes[2], latitudes, longitudes[0], latitudes) / 2
        dy = (
            geodesic_distance(
                longitudes[0], latitudes[2:], longitudes[0], latitudes[:-2]
            )
            / 2
        )
        dy = pad(dy)

        areas = dx * dy
        areas = np.tile(areas, (len(longitudes), 1)).T
    else:
        logger.warning("Irregular lat/lon grid. May be slow to calculate areas.")
        LONS, LATS = np.meshgrid(longitudes, latitudes)
        dx = geodesic_distance(
            LONS[1:-1, 2:], LATS[1:-1, 1:-1], LONS[1:-1, :-2], LATS[1:-1, 1:-1]
        )
        dx = dx / 2
        dy = geodesic_distance(
            LONS[1:-1, 1:-1], LATS[2:, 1:-1], LONS[1:-1, 1:-1], LATS[:-2, 1:-1]
        )
        dy = dy / 2
        areas = dx * dy
        areas = np.apply_along_axis(pad, axis=0, arr=areas)
        areas = np.apply_along_axis(pad, axis=1, arr=areas)

    return areas / 1e6
