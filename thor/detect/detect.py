"""Module for detecting objects in a grid."""

import copy
import numbers
from scipy import ndimage
import numpy as np
import xarray as xr
import thor.detect.preprocess as preprocess
from thor.log import setup_logger
from thor.detect.steiner import steiner_scheme
from thor.utils import get_time_interval

logger = setup_logger(__name__)


def threshold(grid, object_options):
    """Detect objects in the given grid using thresholding."""
    if object_options["detection"]["method"] != "threshold":
        raise ValueError("Detection method not set to threshold.")

    binary_grid = grid >= object_options["detection"]["threshold"]
    return binary_grid


def steiner(grid, object_options):
    """Detect objects in the given grid using the Steiner et al. method."""
    if object_options["detection"]["method"] != "steiner":
        raise ValueError("Detection method not set to steiner.")

    if "latitude" in grid.dims:
        coordinates = "geographic"
        x = grid.longitude.values
        y = grid.latitude.values
    elif "x" in grid.dims:
        coordinates = "cartesian"
        x = grid.x.values
        y = grid.y.values
    else:
        raise ValueError("Could not infer grid coordinates.")

    if "altitude" in grid.dims:
        raise ValueError(
            "Steiner et al. (1995) scheme only works with 2D grids. "
            "Apply a flattener first."
        )
    elif grid.altitude != 3e3:
        logger.warning(
            "Steiner et al. (1995) scheme designed to work on 3 km altitude grids. "
            f"grid altitude {grid.altitude.values[0]/1e3} km."
        )

    binary_grid = xr.full_like(grid, 0)
    binary_grid.name = "binary_grid"
    # try:
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y)
    elif x.ndim == 2 and y.ndim == 2:
        X, Y = x, y
    else:
        raise ValueError("x and y must both be one or two dimensional.")
    steiner_class = steiner_scheme(grid.values, X, Y, coordinates=coordinates)
    steiner_class = steiner_class.astype(int)
    steiner_class[steiner_class != 2] = 0
    binary_grid.data = steiner_class
    # except Exception as e:
    #     logger.debug(f"Steiner scheme failed: {e}")

    return binary_grid


detecter_dispatcher = {
    "threshold": threshold,
    "steiner": steiner,
}


flattener_dispatcher = {
    "vertical_max": preprocess.vertical_max,
    "cross_section": preprocess.cross_section,
}


def detect(track_input_records, tracks, level_index, obj, object_options, grid_options):
    """Detect objects in the given grid."""

    previous_grid = copy.deepcopy(tracks[level_index][obj]["current_grid"])
    tracks[level_index][obj]["previous_grids"].append(previous_grid)
    input_record = track_input_records[object_options["dataset"]]
    grid = input_record["current_grid"]
    tracks[level_index][obj]["time_interval"] = get_time_interval(grid, previous_grid)
    dataset = input_record["dataset"]
    if "gridcell_area" not in tracks[level_index][obj].keys():
        tracks[level_index][obj]["gridcell_area"] = dataset["gridcell_area"]

    if object_options["detection"]["flatten_method"] is not None:
        flattener = flattener_dispatcher.get(
            object_options["detection"]["flatten_method"]
        )
        processed_grid = flattener(grid, object_options)
    else:
        processed_grid = grid
    tracks[level_index][obj]["current_grid"] = processed_grid

    detecter = detecter_dispatcher.get(object_options["detection"]["method"])
    if detecter is None:
        raise ValueError("Invalid detection method.")
    binary_grid = detecter(processed_grid, object_options)
    mask = xr.full_like(binary_grid, 0, dtype=int)
    mask.data = ndimage.label(binary_grid)[0]
    mask.name = f"{object_options['name']}_mask"

    if (
        grid_options["name"] == "geographic"
        and "gridcell_area" in dataset.variables.keys()
    ):
        gridcell_area = dataset["gridcell_area"]
    elif grid_options["name"] == "cartesian":
        spacing = grid_options["cartesian_spacing"]
        gridcell_area = spacing[1] * spacing[2]
    else:
        ValueError("Invalid grid name or missing gridcell_area variable.")

    if object_options["detection"]["min_area"] is not None:
        mask = clear_small_area_objects(
            mask, object_options["detection"]["min_area"], gridcell_area
        )

    current_mask = copy.deepcopy(tracks[level_index][obj]["current_mask"])
    tracks[level_index][obj]["previous_masks"].append(current_mask)
    tracks[level_index][obj]["current_mask"] = mask


def clear_small_area_objects(mask, min_area, gridcell_area):
    """Takes in labelled image and clears objects less than min_size."""

    for obj in range(1, int(mask.max()) + 1):
        if isinstance(gridcell_area, xr.DataArray) and len(gridcell_area.shape) == 2:
            obj_area = gridcell_area.data[mask == obj].sum()
        elif isinstance(gridcell_area, numbers.Real) and gridcell_area > 0:
            obj_area = (mask == obj).sum() * gridcell_area
        else:
            raise ValueError("gridcell_area must be a positive number or a 2D array.")
        if obj_area < min_area:
            mask.data[mask == obj] = 0
    # Relabel the mask after clearing the small objects
    mask.data = ndimage.label(mask)[0]
    return mask
