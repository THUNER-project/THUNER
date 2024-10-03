"General utilities for the thor package."
import concurrent.futures
from datetime import datetime
import json
import hashlib
import numpy as np
import pandas as pd
from numba import njit, int32, float32
from numba.typed import List
from scipy.interpolate import interp1d
from pyproj import Geod
from thor.log import setup_logger


logger = setup_logger(__name__)


def check_futures(futures):
    """Check the status of the futures."""
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            logger.error("Generated an exception: %s", exc)


def hash_dictionary(dictionary):
    params_str = json.dumps(dictionary, sort_keys=True)
    hash_obj = hashlib.sha256()
    hash_obj.update(params_str.encode("utf-8"))
    return hash_obj.hexdigest()


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
