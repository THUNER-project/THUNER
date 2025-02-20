"""
Module for generating synthetic reflectivity data for testing. This module a work in 
progress; core functions are very slow. 

"""

import numpy as np
import copy
import xarray as xr
from pyproj import Geod
from thuner.log import setup_logger
from pydantic import Field
import thuner.data.utils as utils
import thuner.grid as grid
from thuner.utils import BaseDatasetOptions

logger = setup_logger(__name__)
geod = Geod(ellps="WGS84")


class SyntheticOptions(BaseDatasetOptions):
    """
    Class for managing the options for the synthetic dataset.
    """

    name: str = Field("synthetic")
    start: str = Field("2005-11-13T00:00:00")
    end: str = Field("2005-11-14T00:00:00")
    fields: list[str] = Field(["reflectivity"])
    use: str = Field("track")
    starting_objects: list[dict] | None = Field(None)


def create_object(
    time,
    center_latitude,
    center_longitude,
    direction,
    speed,
    horizontal_radius=20,
    alt_center=3e3,
    alt_radius=1e3,
    intensity=50,
    eccentricity=0.4,
    orientation=np.pi / 4,
):
    """
    Create a dictionary containing the object properties.

    Parameters
    ----------
    time : str
        The time at which the object has the properties in the dictionary.
    center_latitude : float
        The latitude of the center of the object.
    center_longitude : float
        The longitude of the center of the object.
    direction : float
        The direction the object is moving in radians clockwise from north.
    speed : float
        The speed the object is moving in metres per second.
    horizontal_radius : float, optional
        The horizontal radius of the object in km; default is 20.

    Returns
    -------
    object_dict : dict
        Dictionary containing the object properties.
    """

    object_dict = {
        "time": time,
        "center_latitude": center_latitude,
        "center_longitude": center_longitude,
        "horizontal_radius": horizontal_radius,
        "alt_center": alt_center,
        "alt_radius": alt_radius,
        "intensity": intensity,
        "eccentricity": eccentricity,
        "orientation": orientation,
        "direction": direction,
        "speed": speed,
    }
    return object_dict


def update_dataset(time, input_record, tracks, dataset_options, grid_options):
    """
    Update an aura dataset.

    Parameters
    ----------
    time : datetime64
        The time of the dataset.
    object_tracks : dict
        Dictionary containing the object tracks.
    dataset_options : dict
        Dictionary containing the dataset options.
    grid_options : dict
        Dictionary containing the grid options.

    Returns
    -------
    dataset : object
        The updated dataset.
    """
    utils.log_dataset_update(logger, dataset_options.name, time)

    latitude = grid_options.latitude
    longitude = grid_options.longitude
    missing_geographic = (latitude is None) or (longitude is None)
    x = grid_options.x
    y = grid_options.y
    missing_cartesian = (x is None) or (y is None)
    if grid_options.name == "cartesian" and missing_geographic:
        X, Y = np.meshgrid(grid_options.x, grid_options.y)
        LON, LAT = grid.cartesian_to_geographic_lcc(grid_options, X, Y)
        grid_options.latitude = LAT
        grid_options.longitude = LON
    if grid_options.name == "geographic" and missing_cartesian:
        LON, LAT = np.meshgrid(longitude, latitude)
        X, Y = grid.geographic_to_cartesian_lcc(grid_options, LAT, LON)
        grid_options.x = X
        grid_options.y = Y

    if input_record.synthetic_objects is None:
        input_record.synthetic_objects = dataset_options.starting_objects

    updated_objects = copy.deepcopy(input_record.synthetic_objects)
    for i in range(len(input_record.synthetic_objects)):
        updated_objects[i] = update_object(time, input_record.synthetic_objects[i])
    input_record.synthetic_objects = updated_objects

    if input_record.synthetic_base_dataset is None:
        input_record.synthetic_base_dataset = create_dataset(time, grid_options)
    ds = copy.deepcopy(input_record.synthetic_base_dataset)
    ds["time"] = np.array([np.datetime64(time)])

    for object in input_record.synthetic_objects:
        ds = add_reflectivity(ds, **object)

    input_record.dataset = ds


def update_object(time, obj):
    """
    Update object based on the difference between time and the object time.
    """
    time_diff = np.datetime64(time) - np.datetime64(obj["time"])
    time_diff = time_diff.astype("timedelta64[s]").astype(float)
    distance = time_diff * obj["speed"]
    args = [obj["center_longitude"], obj["center_latitude"]]
    args += [np.rad2deg(obj["direction"]), distance]
    new_lon, new_lat = geod.fwd(*args)[0:2]
    obj["center_latitude"] = new_lat
    obj["center_longitude"] = new_lon
    obj["time"] = time

    return obj


def create_dataset(time, grid_options):
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

    dims = grid.get_coordinate_names(grid_options)
    if dims == ["latitude", "longitude"]:
        alt_dims = ["y", "x"]
    elif dims == ["y", "x"]:
        alt_dims = ["latitude", "longitude"]
    else:
        raise ValueError("Invalid grid options")

    time = np.array([np.datetime64(time)]).astype("datetime64[ns]")
    meridional_dim = np.array(getattr(grid_options, dims[0]))
    zonal_dim = np.array(getattr(grid_options, dims[1]))
    alt = np.array(grid_options.altitude)

    # Create ds
    ds_values = np.ones((1, len(alt), len(meridional_dim), len(zonal_dim))) * np.nan
    coords = {"time": time, "altitude": alt}
    coords.update({dims[0]: meridional_dim, dims[1]: zonal_dim})
    variables_dict = {
        "reflectivity": (["time", "altitude", dims[0], dims[1]], ds_values),
        alt_dims[0]: ([dims[0], dims[1]], getattr(grid_options, alt_dims[0])),
        alt_dims[1]: ([dims[0], dims[1]], getattr(grid_options, alt_dims[1])),
    }
    ds = xr.Dataset(variables_dict, coords=coords)
    ds["reflectivity"].attrs.update({"long_name": "reflectivity", "units": "dBZ"})

    cell_areas = grid.get_cell_areas(grid_options)
    ds["gridcell_area"] = (dims, cell_areas)
    ds["gridcell_area"].attrs.update(
        {"units": "km^2", "standard_name": "area", "valid_min": 0}
    )
    LON, LAT, ALT = xr.broadcast(ds.time, ds.longitude, ds.latitude, ds.altitude)[1:]
    ds["LON"], ds["LAT"], ds["ALT"] = LON, LAT, ALT
    return ds


def add_reflectivity(
    ds,
    center_latitude,
    center_longitude,
    horizontal_radius,
    alt_center,
    alt_radius,
    intensity,
    eccentricity,
    orientation,
    **kwargs,
):
    """
    Add elliptical/gaussian synthetic reflectivity data to emulate cells, anvils etc.
    This needs to be made much more efficient.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to add the reflectivity to.
    lat_center : float
        The latitude of the center of the ellipse.
    lon_center : float
        The longitude of the center of the ellipse.
    horizontal_radius : float
        The horizontal "radius" of the ellipse in km. Note this is only an approximate
        radius, as the object dimenensions are defined in geographic coordinates.
    """

    LON, LAT, ALT = ds.LON, ds.LAT, ds.ALT

    # lon, lat, alt = xr.broadcast(ds.time, ds.longitude, ds.latitude, ds.altitude)[1:]

    # Calculate the rotated coordinates
    lon_rotated = (LON - center_longitude) * np.cos(orientation)
    lon_rotated += (LAT - center_latitude) * np.sin(orientation)
    lat_rotated = -(LON - center_longitude) * np.sin(orientation)
    lat_rotated += (LAT - center_latitude) * np.cos(orientation)

    # Convert horizontal_radius to approximate lat/lon radius.
    horizontal_radius = horizontal_radius / 111.32

    # Calculate the distance from the center for each point in the grid, considering eccentricity
    distance = np.sqrt(
        (lon_rotated / horizontal_radius) ** 2
        + (lat_rotated / (horizontal_radius * eccentricity)) ** 2
        + ((ALT - alt_center) / alt_radius) ** 2
    )

    # Apply a Gaussian function to create an elliptical pattern
    reflectivity = intensity * np.exp(-(distance**2) / 2)
    reflectivity = reflectivity.where(reflectivity >= 0.05 * intensity, np.nan)
    reflectivity = reflectivity.transpose(*ds.dims)

    # Add the generated data to the ds DataArray
    ds["reflectivity"].values = xr.where(
        ~np.isnan(reflectivity), reflectivity, ds["reflectivity"]
    )
    return ds


def convert_synthetic():
    pass
