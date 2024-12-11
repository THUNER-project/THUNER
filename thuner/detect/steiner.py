"""
Classify reflectivity using Steiner (1995) scheme. Based on code by Valentin Louf,
in turn adapted from Py-ART code. The method has been extended to cope with
irregular grids, and geographic coordinates. In geographic coordinates, we determine
the points within the background and convective radii using the Haversine distance.
Note these radii are at most 11 km, so Haversine is accurate enough.
"""

import numpy as np
from thuner.log import setup_logger
from numba import int32, float32, njit
from numba.typed import List
from thuner.utils import meshgrid_numba, numba_boolean_assign, equirectangular

logger = setup_logger(__name__)

use_numba = True


def conditional_jit(*jit_args, use_numba=True, **jit_kwargs):
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
    X,
    Y,
    radius_option=1,
    delta_Z_option=0,
    background_radius=11e3,
    dBZ_threshold=42,
    use_dBZ_threshold=True,
    coordinates="geographic",
):
    reflectivity = reflectivity.astype(float32)
    X = X.astype(float32)
    Y = Y.astype(float32)
    X_rad = X
    Y_rad = Y

    if coordinates == "geographic":
        X_rad = np.radians(X)
        Y_rad = np.radians(Y)

    II, JJ = meshgrid_numba(
        np.arange(reflectivity.shape[1], dtype=float32),
        np.arange(reflectivity.shape[0], dtype=float32),
    )
    II = numba_boolean_assign(II, np.isnan(reflectivity))
    JJ = numba_boolean_assign(JJ, np.isnan(reflectivity))

    classification = np.zeros_like(reflectivity, dtype=int32)

    for j, i in np.ndindex(reflectivity.shape):
        if np.isnan(reflectivity[j, i]) or (classification[j, i] != 0):
            continue

        reflectivity_background = values_within_radius(
            reflectivity, X, Y, j, i, background_radius, X_rad, Y_rad, coordinates
        )
        condition = ~np.isnan(reflectivity_background)
        reflectivity_background = reflectivity_background[condition]
        mean_background_reflectivity = 10 * np.log10(
            np.nanmean(10.0 ** (reflectivity_background / 10))
        )
        convective_radius = get_convective_radius(
            mean_background_reflectivity, radius_option
        )

        II_radius = values_within_radius(
            II, X, Y, j, i, convective_radius, X_rad, Y_rad, coordinates
        )
        JJ_radius = values_within_radius(
            JJ, X, Y, j, i, convective_radius, X_rad, Y_rad, coordinates
        )
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
def values_within_radius(
    array, X, Y, j, i, radius, X_rad, Y_rad, coordinates="geographic"
):
    """
    Efficiently find the values of an array at locations within a radius of a point.
    We mask values where the array is nan.
    """

    if coordinates == "geographic":
        y_cond = (
            equirectangular(Y_rad[j, i], X_rad[j, i], Y_rad[:, i], X_rad[j, i])
            <= radius
        )
        x_cond = (
            equirectangular(Y_rad[j, i], X_rad[j, i], Y_rad[j, i], X_rad[j, :])
            <= radius
        )
    elif coordinates == "cartesian":
        y_cond = (Y[:, i] - Y[j, i]) ** 2 <= radius**2
        x_cond = (X[j, :] - X[j, i]) ** 2 <= radius**2

    boxes = List()
    if coordinates == "geographic":
        array_list = [array, X_rad, Y_rad]
    elif coordinates == "cartesian":
        array_list = [array, X, Y]
    for arr in array_list:
        arr_box = arr[y_cond, :]
        arr_box = arr_box[:, x_cond]
        boxes.append(arr_box.flatten())
    [array_box, X_box, Y_box] = boxes

    if coordinates == "geographic":
        radius_cond = equirectangular(Y_box, X_box, Y_rad[j, i], X_rad[j, i]) <= radius
    elif coordinates == "cartesian":
        radius_cond = np.sqrt((Y_box - Y[j, i]) ** 2 + (X_box - X[j, i]) ** 2) <= radius
    radius_cond = radius_cond.flatten() & ~np.isnan(array_box.flatten())

    return array_box[radius_cond]
