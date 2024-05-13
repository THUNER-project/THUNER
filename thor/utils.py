"General utilities for the thor package."

import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime


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
    rounded_numbers = [round(num, decimal_places) for num in numbers]
    return len(set(rounded_numbers)) == 1


def pad(array, left_pad=1, right_pad=1, kind="linear"):
    x = np.arange(len(array))
    f = interp1d(x, array, kind=kind, fill_value="extrapolate")
    return f(np.arange(-1, len(array) + 1))


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
