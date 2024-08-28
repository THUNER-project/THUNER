"""
Module for generating synthetic reflectivity data for testing. 
"""

import numpy as np
import copy
import xarray as xr
from thor.log import setup_logger
from thor.config import get_outputs_directory
import thor.data.option as option
from scipy.stats import vonmises
import thor.data.utils as utils
import thor.grid as grid
from pyproj import Geod


logger = setup_logger(__name__)
geod = Geod(ellps="WGS84")


def synthetic_data_options(
    start="2005-11-13T00:00:00",
    end="2005-11-14T00:00:00",
    use="track",
    fields=None,
    deque_length=2,
    starting_objects=None,
    regeneration_options=None,
):
    """
    Generate CPOL radar data options dictionary.

    Parameters
    ----------
    name : str, optional
        The name of the dataset; default is "cpol".
    start : str, optional
        The start time of the dataset; default is "2005-11-13T00:00:00".
    end : str, optional
        The end time of the dataset; default is "2005-11-14T00:00:00".
    save_options : bool, optional
        Whether to save the data options; default is False.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    options : dict
        Dictionary containing the input options.
    """

    if fields is None:
        fields = ["reflectivity"]

    options = {
        "name": "synthetic",
        "start": start,
        "end": end,
        "deque_length": deque_length,
        "fields": fields,
        "use": use,
        "starting_objects": starting_objects,
        "regeneration_options": regeneration_options,
    }

    return options


def create_object_dictionary(
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
        The direction the object is moving in radians counterclockwise from east.
    speed : float
        The speed the object is moving in metres per second.
    horizontal_radius : float, optional
        The horizontal radius of the object in km; default is 10.

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
    utils.log_dataset_update(logger, dataset_options["name"], time)

    if "objects" not in input_record.keys():
        input_record["objects"] = dataset_options["starting_objects"]

    updated_objects = copy.deepcopy(input_record["objects"])
    for object in input_record["objects"]:
        updated_objects.append(update_object(time, object))
    input_record["objects"] = updated_objects

    ds = create_dataset(grid_options)
    for object in input_record["objects"]:
        ds = add_reflectivity(ds, **object)

    input_record["dataset"] = ds


def update_object(time, object):
    """
    Update object based on the difference between time and the object time.
    """
    time_diff = np.datetime64(time) - np.datetime64(object["time"])
    time_diff = time_diff.astype("timedelta64[s]").astype(int)
    distance = time_diff * object["speed"]
    new_lat, new_lon = geod.fwd(
        object["center_latitude"],
        object["center_longitude"],
        np.rad2deg(object["direction"]),
        distance,
    )[0:2]
    object["center_latitude"] = new_lat
    object["center_longitude"] = new_lon
    object["time"] = time
    return object


def create_dataset(grid_options):
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

    ds_values = np.ones((len(alt), len(lat), len(lon))) * np.nan
    ds = xr.Dataset(
        {"reflectivity": (["altitude", "latitude", "longitude"], ds_values)},
        coords={"altitude": alt, "latitude": lat, "longitude": lon},
    )
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

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to add the reflectivity to.
    lat_center : float
        The latitude of the center of the ellipse.
    lon_center : float
        The longitude of the center of the ellipse.
    horizontal_radius : float
        The horizontal radius of the ellipse in km.
    """

    # Create a meshgrid for the coordinates
    lon, lat, alt = xr.broadcast(ds.longitude, ds.latitude, ds.altitude)

    # Calculate the rotated coordinates
    x_rot = (lon - center_longitude) * np.cos(orientation)
    x_rot += (lat - center_latitude) * np.sin(orientation)
    y_rot = -(lon - center_longitude) * np.sin(orientation)
    y_rot += (lat - center_latitude) * np.cos(orientation)

    # Convert horizontal_radius to approximate lat/lon radius.
    horizontal_radius = horizontal_radius / 111.32

    # Calculate the distance from the center for each point in the grid, considering eccentricity
    distance = np.sqrt(
        (x_rot / horizontal_radius) ** 2
        + (y_rot / (horizontal_radius * eccentricity)) ** 2
        + ((alt - alt_center) / alt_radius) ** 2
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


# from scipy.stats import vonmises

# # Parameters
# mean_direction = np.pi / 2  # Mean direction (mu)
# sigma = 2*np.pi/1e3
# concentration = 1 / sigma  # Concentration parameter (kappa)

# # Create a von Mises distribution
# distribution = vonmises(kappa=concentration, loc=mean_direction)

# # Sample from the distribution
# samples = distribution.rvs(size=1000)
# samples = samples % (2 * np.pi)

# # Plot the distribution
# theta = np.linspace(0, 2 * np.pi, 1000)
# pdf = distribution.pdf(theta)

# plt.figure(figsize=(8, 4))
# plt.plot(theta, pdf, label='von Mises PDF')
# plt.hist(samples, bins=50, density=True, alpha=0.6, label='Samples')
# plt.axvline(mean_direction, color='r', linestyle='--', label='Mean Direction')
# plt.xticks(np.linspace(0, 2 * np.pi, 9), labels=['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$', r'$2\pi$'])
# plt.xlabel('Direction (radians)')
# plt.ylabel('Density')
# plt.title('von Mises Distribution')
# plt.legend()
# plt.show()
