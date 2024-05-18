"""Module for detecting objects in a grid."""

import numbers
import numpy as np
from scipy import ndimage
import thor.detect.preprocess as preprocess


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

    # Implement the Steiner method here.
    pass


detecter_dispatcher = {
    "threshold": threshold,
    "steiner": steiner,
}


flattener_dispatcher = {
    "vertical_max": preprocess.vertical_max,
    "cross_section": preprocess.cross_section,
}


def detect(track_input_records, object_options, grid_options):
    """Detect objects in the given grid."""

    grid = track_input_records["current_grid"]

    if object_options["detection"]["flatten"] is not None:
        flattener = flattener_dispatcher.get(object_options["detection"]["flatten"])
        processed_grid = flattener(grid)
    else:
        processed_grid = grid

    detecter = detecter_dispatcher.get(object_options["detection"]["method"])
    if detecter is None:
        raise ValueError("Invalid detection method.")
    binary_grid = detecter(processed_grid, object_options)
    mask = ndimage.label(binary_grid)

    dataset = track_input_records["dataset"]
    if grid_options["name"] == "geographic" and "cell_area" in dataset.variables.keys():
        cell_area = dataset["cell_area"]
    elif grid_options["name"] == "cartesian":
        spacing = grid_options["cartesian_spacing"]
        cell_area = spacing[1] * spacing[2]
    else:
        ValueError("Invalid grid name or missing cell_area variable.")

    if object_options["detection"]["min_area"] is not None:
        mask = clear_small_area_objects(mask, object_options["min_area"], cell_area)
    if mask.max() == 0:
        mask = None
    return mask


def clear_small_area_objects(mask, min_area, cell_area):
    """Takes in labelled image and clears objects less than min_size."""

    for obj in range(1, mask.max() + 1):
        if isinstance(cell_area, np.ndarray) and len(cell_area.shape) == 2:
            obj_area = cell_area[mask == obj].sum()
        elif isinstance(cell_area, numbers.Real) and cell_area > 0:
            obj_area = (mask == obj).sum() * cell_area
        else:
            raise ValueError("cell_area must be a positive number or a 2D array.")
        if obj_area < min_area:
            mask[mask == obj] = 0
    # Relabel the mask after clearing the small objects
    mask = ndimage.label(mask)
    return mask
