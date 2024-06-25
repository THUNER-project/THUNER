"General utilities for the thor package."

from datetime import datetime
import numpy as np
from scipy.interpolate import interp1d


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
    """
    Drop the time component of a datetime64 object.

    Parameters
    ----------
    time : np.datetime64
        Datetime object.

    Returns
    -------
    date : np.datetime64
        Date object.
    """
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


def now_str():
    """Return the current time as a filename friendly string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def format_time(time):
    """Format a datetime64 object as a filename safe string."""
    time_seconds = time.astype("datetime64[s]")
    return time_seconds.astype(datetime).strftime("%Y%m%d_%H%M%S")
