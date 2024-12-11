"General utilities for the thuner package."
from datetime import datetime
import yaml
from pathlib import Path
from thuner.config import get_outputs_directory
import json
import hashlib
import numpy as np
import pandas as pd
from numba import njit, int32, float32
from numba.typed import List
from scipy.interpolate import interp1d
import os
import platform
from pathlib import Path
from thuner.log import setup_logger


logger = setup_logger(__name__)


class SingletonBase:
    """
    Base class for implementing singletons in python. See for instance the classic
    "Gang of Four" design pattern book for more information on the "singleton" pattern.
    The idea is that only one instance of a "singleton" class can exist at one time,
    making these useful for storing program state.

    Gamma et al. (1995), Design Patterns: Elements of Reusable Object-Oriented Software.

    Note however that if processes are created with, e.g., the multiprocessing module
    different processes will have different instances of the singleton. We can avoid
    this by explicitly passing the singleton instance to the processes.
    """

    # The base class now keeps track of all instances of singleton classes
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super(SingletonBase, cls).__new__(cls)
            instance._initialize(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    def _initialize(self, *args, **kwargs):
        """
        Initialize the singleton instance. This method should be overridden by subclasses.
        """
        pass


def format_string_list(strings):
    """
    Format a list of strings into a human-readable string.

    Parameters
    ----------
    strings : list of str
        List of strings to be formatted.

    Returns
    -------
    formatted_string : str
        The formatted string.
    """
    if len(strings) > 1:
        formatted_string = ", ".join(strings[:-1]) + " or " + strings[-1]
        return formatted_string
    elif strings:
        return strings[0]
    else:
        raise ValueError("strings must be an iterable of strings'.")


def create_hidden_directory(path):
    """Create a hidden directory."""
    if not Path(path).name.startswith("."):
        hidden_path = Path(path).parent / f".{Path(path).name}"
    else:
        hidden_path = Path(path)
    if hidden_path.exists() and hidden_path.is_file():
        message = f"{hidden_path} exists, but is a file, not a directory."
        raise FileExistsError(message)
    hidden_path.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Windows":
        os.system(f'attrib +h "{hidden_path}"')
    else:
        os.makedirs(hidden_path, exist_ok=True)
    return hidden_path


def hash_dictionary(dictionary):
    params_str = json.dumps(dictionary, sort_keys=True)
    hash_obj = hashlib.sha256()
    hash_obj.update(params_str.encode("utf-8"))
    return hash_obj.hexdigest()


def drop_time(time):
    """Drop the time component of a datetime64 object."""
    return time.astype("datetime64[D]").astype("datetime64[s]")


def almost_equal(numbers, decimal_places=5):
    """Check if all numbers are equal to a certain number of decimal places."""
    rounded_numbers = [round(num, decimal_places) for num in numbers]
    return len(set(rounded_numbers)) == 1


def pad(array, left_pad=1, right_pad=1, kind="linear"):
    """Pad an array by extrapolating."""
    x = np.arange(len(array))
    f = interp1d(x, array, kind=kind, fill_value="extrapolate")
    return f(np.arange(-left_pad, len(array) + right_pad))


def print_keys(dictionary, indent=0):
    """Print the keys of a nested dictionary."""
    for key, value in dictionary.items():
        print("\t".expandtabs(4) * indent + str(key))
        if isinstance(value, dict):
            print_keys(value, indent + 1)


def check_component_options(component_options):
    """Check options for converted datasets and masks."""

    if not isinstance(component_options, dict):
        raise TypeError("component_options must be a dictionary.")
    if "save" not in component_options:
        raise KeyError("save key not found in component_options.")
    if "load" not in component_options:
        raise KeyError("load key not found in component_options.")
    if not isinstance(component_options["save"], bool):
        raise TypeError("save key must be a boolean.")
    if not isinstance(component_options["load"], bool):
        raise TypeError("load key must be a boolean.")


def time_in_dataset_range(time, dataset):
    """Check if a time is in a dataset."""

    if dataset is None:
        return False

    condition = time >= dataset.time.values.min() and time <= dataset.time.values.max()
    return condition


def get_hour_interval(time, interval=6):
    if 24 % interval != 0:
        raise ValueError("Interval must be a divisor of 24")
    hour = time.astype("M8[h]").item().hour
    start_hour = hour // interval * interval
    start = np.datetime64(time, "h") - np.timedelta64(hour - start_hour, "h")
    end = start + np.timedelta64(interval, "h")
    return start, end


def format_time(time, filename_safe=True, day_only=False):
    """Format a np.datetime64 object as a string, truncating to seconds."""
    time_seconds = pd.DatetimeIndex([time]).round("s")[0]
    if day_only:
        time_str = time_seconds.strftime("%Y-%m-%d")
    else:
        time_str = time_seconds.strftime("%Y-%m-%dT%H:%M:%S")
    if filename_safe:
        time_str = time_str.replace(":", "").replace("-", "").replace("T", "_")
    return time_str


def now_str(filename_safe=True):
    """Return the current time as a string."""
    return format_time(datetime.now(), filename_safe=filename_safe, day_only=False)


def get_time_interval(current_grid, previous_grid):
    """Get the time interval between two grids."""
    if previous_grid is not None:
        time_interval = current_grid.time.values - previous_grid.time.values
        time_interval = time_interval.astype("timedelta64[s]").astype(int)
        return time_interval
    else:
        return None


use_numba = True


def conditional_jit(use_numba=True, *jit_args, **jit_kwargs):
    """
    A decorator that applies Numba's JIT compilation to a function if use_numba is True.
    Otherwise, it returns the original function. It also adjusts type aliases based on the
    usage of Numba.
    """

    def decorator(func):
        if use_numba:
            # Define type aliases for use with Numba
            globals()["int32"] = int32
            globals()["float32"] = float32
            globals()["List"] = List
            return njit(*jit_args, **jit_kwargs)(func)
        else:
            # Define type aliases for use without Numba
            globals()["int32"] = int
            globals()["float32"] = float
            globals()["List"] = list
            return func

    return decorator


@conditional_jit(use_numba=use_numba)
def meshgrid_numba(x, y):
    """
    Create a meshgrid-like pair of arrays for x and y coordinates.
    This function mimics the behaviour of np.meshgrid but is compatible with Numba.
    """
    m, n = len(y), len(x)
    X = np.empty((m, n), dtype=x.dtype)
    Y = np.empty((m, n), dtype=y.dtype)

    for i in range(m):
        X[i, :] = x
    for j in range(n):
        Y[:, j] = y

    return X, Y


@conditional_jit(use_numba=use_numba)
def numba_boolean_assign(array, condition, value=np.nan):
    """
    Assign a value to an array based on a boolean condition.
    """
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if condition[i, j]:
                array[i, j] = value
    return array


@conditional_jit(use_numba=use_numba)
def equirectangular(lat1_radians, lon1_radians, lat2_radians, lon2_radians):
    """
    Calculate the equirectangular distance between two points
    on the earth, where lat and lon are expressed in radians.
    """

    # Equirectangular approximation formula
    dlat = lat2_radians - lat1_radians
    dlon = lon2_radians - lon1_radians
    avg_lat = (lat1_radians + lat2_radians) / 2
    r = 6371e3  # Radius of Earth in metres
    x = dlon * np.cos(avg_lat)
    y = dlat
    return np.sqrt(x**2 + y**2) * r


@conditional_jit(use_numba=use_numba)
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in metres between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371e3  # Radius of earth in metres
    return c * r


def save_options(options, filename=None, options_directory=None, append_time=False):
    """Save the options to a yml file."""
    if filename is None:
        filename = now_str()
        append_time = False
    else:
        filename = Path(filename).stem
    if append_time:
        filename += f"_{now_str()}"
    filename += ".yml"
    if options_directory is None:
        options_directory = get_outputs_directory() / "options"
    if not options_directory.exists():
        options_directory.mkdir(parents=True)
    filepath = options_directory / filename
    logger.debug("Saving options to %s", options_directory / filename)
    with open(filepath, "w") as outfile:
        yaml.dump(
            options,
            outfile,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def new_angle(angles):
    """
    Get the angle between the two angles that are farthest apart. All angles are
    provided/returned in radians.
    """
    if len(angles) == 0:
        return 0
    sorted_angles = np.sort(angles)
    gaps = np.diff(sorted_angles)
    circular_gap = 2 * np.pi - (sorted_angles[-1] - sorted_angles[0])
    gaps = np.append(gaps, circular_gap)
    max_gap_index = np.argmax(gaps)
    if max_gap_index == len(gaps) - 1:
        # Circular gap case
        angle1 = sorted_angles[-1]
        angle2 = sorted_angles[0] + 2 * np.pi
    else:
        angle1 = sorted_angles[max_gap_index]
        angle2 = sorted_angles[max_gap_index + 1]
    return (angle1 + angle2) / 2 % (2 * np.pi)


def circular_mean(angles, weights=None):
    """
    Calculate a weighted circular mean. Based on the scipy.stats.circmean function.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html
    """
    if weights is None:
        weights = np.ones_like(angles)
    angles, weights = np.array(angles), np.array(weights)
    total_weight = np.sum(weights)
    # Convert the angles to complex numbers of unit length
    complex_numbers = np.exp(1j * angles)
    # Get the angle of the weighted sum of the complex numbers
    return np.angle(np.sum(weights * complex_numbers)) % (2 * np.pi)


def circular_variance(angles, weights=None):
    """
    Calculate a weighted circular variance. Based on the scipy.stats.circvar function.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circvar.html
    """
    if weights is None:
        weights = np.ones_like(angles)
    angles, weights = np.array(angles), np.array(weights)
    # Convert the angles to complex numbers of unit length
    complex_numbers = np.exp(1j * angles)
    total_weight = np.sum(weights)
    if total_weight == 0:
        return np.nan
    complex_sum = np.sum(weights * complex_numbers / total_weight)
    return 1 - np.abs(complex_sum)
