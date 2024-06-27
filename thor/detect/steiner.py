"""
Classify reflectivity using Steiner (1995) scheme. Based on code by Valentin Louf,
in turn adapted from Py-ART code. The method has been extended to cope with
irregular grids, and geographic coordinates. In geographic coordinates, we determine
the points within the background and convective radii using the Haversine distance.
Note these radii are at most 11 km, so Haversine is accurate enough.
"""

import numpy as np
from itertools import product
from thor.log import setup_logger
from numba import njit, int32, float32
from numba.typed import List


logger = setup_logger(__name__)

use_numba = False


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
            globals()["int32"] = np.int32
            globals()["float32"] = np.float32
            globals()["List"] = list
            return func

    return decorator


@conditional_jit(use_numba=use_numba)
def steiner_scheme(
    reflectivity,
    x,
    y,
    radius_option=1,
    delta_Z_option=0,
    background_radius=11e3,
    dBZ_threshold=42,
    use_dBZ_threshold=True,
    coordinates="geographic",
):
    reflectivity = reflectivity.astype(np.float32)
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    if x.ndim == 1 and y.ndim == 1:
        X, Y = meshgrid_numba(x, y)
    elif x.ndim == 2 and y.ndim == 2:
        X, Y = x, y
    else:
        raise ValueError("x and y must both be one or two dimensional.")

    II, JJ = meshgrid_numba(
        np.arange(reflectivity.shape[1], dtype=float32),
        np.arange(reflectivity.shape[0], dtype=float32),
    )
    II = numba_boolean_assign(II, np.isnan(reflectivity))
    JJ = numba_boolean_assign(JJ, np.isnan(reflectivity))

    classification = np.zeros_like(reflectivity, dtype=int32)

    for i, j in product(range(reflectivity.shape[1]), range(reflectivity.shape[0])):
        if np.isnan(reflectivity[j, i]) or (classification[j, i] != 0):
            continue

        reflectivity_background = values_within_radius(
            reflectivity, X, Y, j, i, background_radius, coordinates
        )
        if len(reflectivity_background) == 0:
            mean_background_reflectivity = np.inf
        else:
            reflectivity_background = reflectivity_background[
                ~np.isnan(reflectivity_background)
            ]
            mean_background_reflectivity = 10 * np.log10(
                np.nanmean(10.0 ** (reflectivity_background / 10))
            )
        convective_radius = get_convective_radius(
            mean_background_reflectivity, radius_option
        )
        II_radius = values_within_radius(II, X, Y, j, i, convective_radius, coordinates)
        JJ_radius = values_within_radius(JJ, X, Y, j, i, convective_radius, coordinates)
        II_radius = II_radius.astype(int32)
        JJ_radius = JJ_radius.astype(int32)

        if use_dBZ_threshold and (reflectivity[j, i] >= dBZ_threshold):
            for ii, jj in zip(II_radius, JJ_radius):
                classification[jj, ii] = 2
        else:
            delta_Z_threshold = get_delta_Z_threshold(
                mean_background_reflectivity, delta_Z_option
            )
            if reflectivity[j, i] - mean_background_reflectivity >= delta_Z_threshold:
                for ii, jj in zip(II_radius, JJ_radius):
                    classification[jj, ii] = 2
            else:
                classification[j, i] = 1
    return classification


@conditional_jit(use_numba=use_numba)
def get_convective_radius(background_reflectivity, radius_option=1):
    """
    Return the convective radius based on the background reflectivity.
    Steiner et al. (1995) considered multiple tresholds. Previous work
    (Louf et al., 2019, Short et al., 2023, Short & Lane, 2023) used thresholds
    beginning at 25 dBZ, i.e. threshold_option=1.

    Parameters:
    ----------
    background_reflectivity: float
        Background reflectivity in dBZ.
    radius_option: int
        Radius option. Default is 1.

    Returns:
    -------
    convective_radius: float
        Convective radius in metres.
    """

    # Define thresholds and radii for different area relations
    base_thresholds = np.arange(20, 35 + 5, 5)
    convective_radii = np.arange(1e3, 5e3 + 1e3, 1e3)

    # Select the appropriate set of thresholds based on threshold_option
    thresholds = base_thresholds + radius_option * 5

    # Get the convective radius from background reflectivity and thresholds
    for i, threshold in enumerate(thresholds):
        if background_reflectivity < threshold:
            return convective_radii[i]
    return convective_radii[-1]


@conditional_jit(use_numba=use_numba)
def get_delta_Z_threshold(background_reflectivity, delta_Z_option=0):
    """
    Return the relevant delta_Z threshold based on the background reflectivity.

    Parameters:
    ----------
    background_reflectivity: float
        Background reflectivity in dBZ.
    threshold_option: float
        Threshold option. Default is 0 to match the Steiner et al. (1995) function.

    Returns:
    -------
    delta_Z_threshold: float
        delta_Z_threshold in dB.
    """

    delta_Z_threshold = 10 + delta_Z_option * 4
    if (background_reflectivity >= 0.0) and (background_reflectivity < 42.43):
        delta_Z_threshold -= background_reflectivity**2 / 180.0
    else:
        delta_Z_threshold = 0.0

    return delta_Z_threshold


@conditional_jit(use_numba=use_numba)
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in metres between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371e3  # Radius of earth in metres
    return c * r


@conditional_jit(use_numba=use_numba)
def values_within_radius(array, X, Y, j, i, radius, coordinates="geographic"):
    """
    Efficiently find the values of an array at locations within a radius of a point.
    We mask values where the array is nan.
    """

    if coordinates == "geographic":
        y_cond = haversine(Y[j, i], X[j, i], Y[:, i], X[j, i]) <= radius
        x_cond = haversine(Y[j, i], X[j, i], Y[j, i], X[j, :]) <= radius
    elif coordinates == "cartesian":
        y_cond = (Y[:, i] - Y[j, i]) ** 2 <= radius**2
        x_cond = (X[j, :] - X[j, i]) ** 2 <= radius**2
    else:
        raise ValueError("Invalid coordinates. Must be 'geographic' or 'cartesian'.")

    boxes = List()
    for arr in [array, X, Y]:
        arr_box = arr[y_cond, :]
        arr_box = arr_box[:, x_cond]
        boxes.append(arr_box)
    [array_box, X_box, Y_box] = boxes

    if coordinates == "geographic":
        radius_cond = haversine(Y_box, X_box, Y[j, i], X[j, i]) <= radius
    elif coordinates == "cartesian":
        radius_cond = np.sqrt((Y_box - Y[j, i]) ** 2 + (X_box - X[j, i]) ** 2) <= radius
    radius_cond = radius_cond.flatten() & ~np.isnan(array_box.flatten())
    return array_box.flatten()[radius_cond]


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
