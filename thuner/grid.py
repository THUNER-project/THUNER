"""Deal with thuner grid objects"""

import yaml
import inspect
import numpy as np
from pyproj import Geod, Proj, Transformer
from pathlib import Path
from pydantic import Field, model_validator
from thuner.utils import almost_equal, pad
from thuner.config import get_outputs_directory
from thuner.log import setup_logger
import thuner.option as option
import thuner.utils as utils


logger = setup_logger(__name__)
grid_name_message = "Grid name must be 'cartesian' or 'geographic'."

_summary = {
    "timestep": "Time step for the dataset.",
    "start_latitude": "Starting latitude for the dataset.",
    "end_latitude": "Ending latitude for the dataset.",
    "start_longitude": "Starting longitude for the dataset.",
    "end_longitude": "Ending longitude for the dataset.",
    "central_latitude": "Central latitude for the dataset.",
    "central_longitude": "Central longitude for the dataset.",
    "projection": "Projection for the dataset.",
    "start_x": "Starting x-coordinate for the dataset.",
    "end_x": "Ending x-coordinate for the dataset.",
    "start_y": "Starting y-coordinate for the dataset.",
    "end_y": "Ending y-coordinate for the dataset.",
    "start_alt": "Starting z-coordinate for the dataset.",
    "end_alt": "Ending z-coordinate for the dataset.",
    "cartesian_spacing": "Spacing for the cartesian grid [z, y, x] in metres.",
    "geographic_spacing": "Spacing for the geographic grid [z, lat, lon] in metres and degrees.",
    "regrid": "Whether to regrid the dataset.",
    "save": "Whether to save the dataset.",
    "altitude_spacing": "Spacing for the altitude grid in metres.",
    "x": "x-coordinates for the dataset.",
    "y": "y-coordinates for the dataset.",
    "altitude": "z-coordinates for the dataset.",
    "latitude": "latitudes for the dataset.",
    "longitude": "longitudes for the dataset.",
    "shape": "Shape of the dataset.",
}


class GridOptions(option.BaseOptions):
    """Class for grid options."""

    name: str = "geographic"
    altitude: list[float] | None = Field(None, description=_summary["altitude"])
    latitude: list[float] | None = Field(None, description=_summary["latitude"])
    longitude: list[float] | None = Field(None, description=_summary["longitude"])
    central_latitude: float | None = Field(
        None, description=_summary["central_latitude"]
    )
    central_longitude: float | None = Field(
        None, description=_summary["central_longitude"]
    )
    x: list[float] | None = Field(None, description=_summary["x"])
    y: list[float] | None = Field(None, description=_summary["y"])
    projection: str | None = Field(None, description=_summary["projection"])
    altitude_spacing: float | None = Field(
        500, description=_summary["altitude_spacing"]
    )
    cartesian_spacing: list[float] | None = Field(
        [2500, 2500], description=_summary["cartesian_spacing"]
    )
    geographic_spacing: list[float] | None = Field(
        [0.025, 0.025], description=_summary["geographic_spacing"]
    )
    shape: tuple[int, int] | None = Field(None, description=_summary["shape"])
    regrid: bool = Field(True, description=_summary["regrid"])
    save: bool = Field(False, description=_summary["save"])

    @model_validator(mode="after")
    def _check_altitude(cls, values):
        if values.altitude is None and values.altitude_spacing is not None:
            spacing = values.altitude_spacing
            altitude = list(np.arange(0, 20e3 + spacing, spacing))
            altitude = [float(alt) for alt in altitude]
            values.altitude = altitude
            logger.warning("altitude not specified. Using default altitudes.")
        elif values.altitude_spacing is None and values.altitude is None:
            message = "altitude_spacing not specified. Will attempt to infer from"
            message += "input."
            logger.warning(message)
        return values

    @model_validator(mode="after")
    def _check_shape(cls, values):
        latitude, longitude = values.latitude, values.longitude
        if values.shape is None and (latitude is not None and longitude is not None):
            values.shape = (len(latitude), len(longitude))
        if values.shape is None and (values.x is not None and values.y is not None):
            values.shape = (len(values.y), len(values.x))
        else:
            logger.warning("shape not specified. Will attempt to infer from input.")


def create_options(
    name="geographic",
    altitude=None,
    latitude=None,
    longitude=None,
    central_latitude=None,
    central_longitude=None,
    x=None,
    y=None,
    projection=None,
    altitude_spacing=500,
    cartesian_spacing=[2500, 2500],
    geographic_spacing=[0.025, 0.025],
    shape=None,
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
        Ending z-coordinate for the dataset; default is 20e3.
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

    if regrid and altitude is None:
        altitude = list(np.arange(0, 20e3 + altitude_spacing, altitude_spacing))
        altitude = [float(alt) for alt in altitude]
    if shape is None and (latitude is not None and longitude is not None):
        shape = (len(latitude), len(longitude))
    if shape is None and (x is not None and y is not None):
        shape = (len(y), len(x))
    else:
        logger.warning("Shape not specified. Will attempt to infer from input.")

    options = {
        "name": name,
        "latitude": latitude,
        "longitude": longitude,
        "central_latitude": central_latitude,
        "central_longitude": central_longitude,
        "altitude": altitude,
        "x": x,
        "y": y,
        "projection": projection,
        "altitude_spacing": altitude_spacing,
        "cartesian_spacing": cartesian_spacing,
        "geographic_spacing": geographic_spacing,
        "shape": shape,
        "regrid": regrid,
        "save": save,
    }

    for key, value in kwargs.items():
        options[key] = value

    if save:
        filepath = get_outputs_directory() / "option/default/grid.yml"
        dump_options = {"default_flow_style": False, "allow_unicode": True}
        dump_options.update({"sort_keys": False})
        with open(filepath, "w") as outfile:
            yaml.dump(options, outfile, **dump_options)

    return options


def cartesian_to_geographic_lcc(grid_options, x, y):
    """Get latitude, longitude coordinates from cartesian coordinates."""

    if grid_options.name != "cartesian":
        raise ValueError("Grid name must be 'cartesian'.")
    if grid_options.latitude is not None and grid_options.longitude is not None:
        logger.warning("Latitude and longitude already specified.")
    if grid_options.central_latitude is None or grid_options.central_longitude is None:
        message = "Central latitude and longitude must be specified."
        raise ValueError(message)

    # Define the LCC projection
    lcc_proj = Proj(
        proj="lcc",
        lat_1=grid_options.central_latitude,
        lat_2=grid_options.central_latitude,
        lat_0=grid_options.central_latitude,
        lon_0=grid_options.central_longitude,
    )

    # Create a transformer to convert from LCC to geographic coordinates
    transformer = Transformer.from_proj(lcc_proj, Proj(proj="latlong", datum="WGS84"))

    # Transform the Cartesian coordinates to geographic coordinates
    longitude, latitude = transformer.transform(x, y)
    return longitude, latitude


def geographic_to_cartesian_lcc(grid_options, latitude, longitude):
    """Get x, y coordinates from geographic coordinates."""

    if grid_options.name != "geographic":
        raise ValueError("Grid name must be 'geographic'.")
    if grid_options.x is not None and grid_options.y is not None:
        logger.warning("x and y already specified.")
    if grid_options.central_latitude is None or grid_options.central_longitude is None:
        grid_options.central_latitude = np.mean(grid_options.latitude)
        grid_options.central_longitude = np.mean(grid_options.longitude)

    # Define the LCC projection
    lcc_proj = Proj(
        proj="lcc",
        lat_1=grid_options.central_latitude,
        lat_2=grid_options.central_latitude,
        lat_0=grid_options.central_latitude,
        lon_0=grid_options.central_longitude,
    )

    # Create a transformer to convert from geographic to LCC coordinates
    transformer = Transformer.from_proj(Proj(proj="latlong", datum="WGS84"), lcc_proj)

    # Transform the geographic coordinates to Cartesian coordinates
    x, y = transformer.transform(longitude, latitude)
    return x, y


def check_spacing(array, dx):
    """Check if array equally spaced."""
    if not almost_equal(np.diff(array)):
        raise ValueError("Grid not equally spaced.")
    elif not almost_equal(list(np.diff(array)) + [dx]):
        raise ValueError("Grid spacing does not match prescribed gridlengths.")


def new_geographic_grid(lats, lons, dlat, dlon):
    """
    Get the geographic grid.
    """

    min_lat = np.floor(lats.min() / dlat) * dlat
    max_lat = np.ceil(lats.max() / dlat) * dlat
    min_lon = np.floor(lons.min() / dlon) * dlon
    max_lon = np.ceil(lons.max() / dlon) * dlon
    new_lats = np.arange(min_lat, max_lat + dlat, dlat)
    new_lons = np.arange(min_lon, max_lon + dlon, dlon)

    return list(new_lats), list(new_lons)


def get_cell_areas(grid_options):
    """
    Get the cell areas.

    Parameters
    ----------
    grid_options : dict
        Dictionary containing the grid grid_options.

    Returns
    -------
    numpy.ndarray
        The cell areas in km^2.
    """

    if grid_options.name == "cartesian":
        area = np.prod(grid_options.cartesian_spacing) / 1e6  # Convert to km^2
        cell_areas = np.ones((len(grid_options.y), len(grid_options.x))) * area
        cell_areas = cell_areas.astype(np.float32)
        return cell_areas
    elif grid_options.name == "geographic":
        return get_geographic_cell_areas(grid_options.latitude, grid_options.longitude)
    else:
        raise ValueError(grid_name_message)


def get_coordinate_names(grid_options):
    """Get the names of the horizontal coordinates."""
    if grid_options.name == "cartesian":
        return ["y", "x"]
    elif grid_options.name == "geographic":
        return ["latitude", "longitude"]
    else:
        raise ValueError(grid_name_message)


def get_distance(row_1, col_1, row_2, col_2, grid_options):
    """Get the distance in meters between two grid cells."""
    row_coords, col_coords = get_horizontal_coordinates(grid_options)
    row_coord_1, col_coord_1 = [row_coords[row_1], col_coords[col_1]]
    row_coord_2, col_coord_2 = [row_coords[row_2], col_coords[col_2]]

    if grid_options.name == "cartesian":
        return np.sqrt(
            (row_coord_2 - row_coord_1) ** 2 + (col_coord_2 - col_coord_1) ** 2
        )
    elif grid_options.name == "geographic":
        return geodesic_distance(col_coord_1, row_coord_1, col_coord_2, row_coord_2)


def get_geographic_cell_areas(lats, lons):
    """Get cell areas in km^2."""

    lats, lons = np.array(lats), np.array(lons)
    d_lon = lons[1:] - lons[:-1]
    d_lat = lats[1:] - lats[:-1]

    if almost_equal(d_lon, 5) and almost_equal(d_lat, 5):

        dx = geodesic_distance(lons[2], lats, lons[0], lats) / 2
        dy = geodesic_distance(lons[0], lats[2:], lons[0], lats[:-2]) / 2
        dy = pad(dy)

        areas = dx * dy
        areas = np.tile(areas, (len(lons), 1)).T
    else:
        logger.warning("Irregular lat/lon grid. May be slow to calculate areas.")
        LONS, LATS = np.meshgrid(lons, lats)
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
    areas = (areas / 1e6).astype(np.float32)  # Convert to km^2
    return areas


def get_horizontal_coordinates(grid_options):
    """
    Get the coordinates for the grid.

    Parameters
    ----------
    grid_options : dict
        Dictionary containing the grid grid_options.

    Returns
    -------
    tuple
        The coordinates as a tuple of (lats, lons, alts).
    """

    if grid_options.name == "cartesian":
        [col_coords, row_coords] = [getattr(grid_options, var) for var in ["x", "y"]]
    elif grid_options.name == "geographic":
        [col_coords, row_coords] = [
            getattr(grid_options, var) for var in ["longitude", "latitude"]
        ]

    return np.array(row_coords), np.array(col_coords)


def get_horizontal_spacing(grid_options):
    """
    Get the grid spacing.

    Parameters
    ----------
    grid_options : dict
        Dictionary containing the grid grid_options.

    Returns
    -------
    tuple
        The grid spacing as a tuple of (dlat, dlon, dz).
    """

    if grid_options.name == "cartesian":
        return grid_options.cartesian_spacing
    elif grid_options.name == "geographic":
        return grid_options.geographic_spacing
    else:
        raise ValueError(grid_name_message)


def pixel_to_cartesian_vector(row, col, vector, grid_options):
    """
    Convert a vector from gridcell, i.e. "pixel", coordinates to cartesian coordinates.
    Note that while mathematically a vector does not have a "start" position, if the
    grid is in geographic coordinates, the "location" of the vector in pixel coordinates
    determines what distances it corresponds to in cartesian coordinates.

    Parameters
    ----------
    row : int
        The start row of the vector.
    col : int
        The start column of the vector
    vector : tuple
        The vector to be converted, represented as a tuple of (delta_row, delta_col)
        in pixel coordinates.
    grid_options : dict
        The grid grid_options dictionary.

    Returns
    -------
    tuple
        The converted vector in cartesian coordinates, represented as a tuple
        (delta_y, delta_x), both in units of metres.
    """

    if grid_options.name == "cartesian":
        return vector * grid_options.cartesian_spacing
    elif grid_options.name == "geographic":
        grid_options.geographic_spacing
        lats = grid_options.latitude
        lons = grid_options.longitude
        start_lat = lats[row]
        start_lon = lons[col]
        end_lat = start_lat + vector[0] * grid_options.geographic_spacing[0]
        end_lon = start_lon + vector[1] * grid_options.geographic_spacing[1]
        return geographic_to_cartesian_displacement(
            start_lat, start_lon, end_lat, end_lon
        )
    else:
        raise ValueError(grid_name_message)


geod = Geod(ellps="WGS84")
geodesic_inverse = np.vectorize(
    lambda lon1, lat1, lon2, lat2: geod.inv(lon1, lat1, lon2, lat2)
)
geodesic_forward = np.vectorize(
    lambda lon, lat, direction, distance: geod.fwd(lon, lat, direction, distance)
)
geodesic_distance = lambda lon1, lat1, lon2, lat2: geodesic_inverse(
    lon1, lat1, lon2, lat2
)[2]


def geographic_to_cartesian_displacement(start_lat, start_lon, end_lat, end_lon):
    """
    Calculate the y and x displacements in metres in cartesian coordinates between two
    points on the Earth's surface given in geographic coordinates.
    """
    direction, backward_direction, distance = geodesic_inverse(
        start_lon, start_lat, end_lon, end_lat
    )
    y_displacement = distance * np.cos(np.radians(direction))
    x_displacement = distance * np.sin(np.radians(direction))
    return y_displacement, x_displacement


def get_pixels_geographic(rows, cols, grid_options):
    """Get the geographic coordinates of the gridcells, i.e. "pixels" at rows, cols."""
    if np.array(rows).shape != np.array(cols).shape:
        raise ValueError("row and col must have the same shape.")
    scalar_input = np.isscalar(rows) and np.isscalar(cols)
    rows = np.array([rows]).flatten()
    cols = np.array([cols]).flatten()
    latitudes = grid_options.latitude
    longitudes = grid_options.longitude
    if grid_options.name == "cartesian":
        # lats, lons are 2D arrays
        lats = [latitudes[row, col] for row, col in zip(rows, cols)]
        lons = [longitudes[row, col] for row, col in zip(rows, cols)]
    elif grid_options.name == "geographic":
        # lats, lons are 1D arrays
        lats, lons = [latitudes[row] for row in rows], [longitudes[col] for col in cols]
    else:
        raise ValueError(grid_name_message)
    if scalar_input:
        lats = lats[0]
        lons = lons[0]
    return lats, lons
