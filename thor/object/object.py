"""Methods for analyzing objects."""

import copy
import numpy as np
import xarray as xr
from thor.log import setup_logger
from thor.match.utils import get_masks
import thor.grid as thor_grid

logger = setup_logger(__name__)


def get_object_area(obj, mask, gridcell_area, grid_options):
    """Get object area. Note gridcell_area is in km^2 by default."""
    row_inds, col_inds = np.where(mask == obj)
    row_points = xr.Variable("mask_points", row_inds)
    col_points = xr.Variable("mask_points", col_inds)
    if grid_options["name"] == "cartesian":
        areas = gridcell_area.isel(y=row_points, x=col_points).values
    elif grid_options["name"] == "geographic":
        areas = gridcell_area.isel(latitude=row_points, longitude=col_points).values
    return areas.sum()


def get_object_center(obj, mask, grid_options, gridcell_area=None, grid=None):
    """Get object centre."""
    coord_names = thor_grid.get_coordinate_names(grid_options)
    row_inds, col_inds = np.where(mask == obj)
    if gridcell_area is not None or grid is not None:
        row_points = xr.Variable("mask_points", row_inds)
        col_points = xr.Variable("mask_points", col_inds)
        sel_dict = {coord_names[0]: row_points, coord_names[1]: col_points}
        areas = gridcell_area.isel(sel_dict).values
        if gridcell_area is not None and grid is None:
            row_inds = np.sum(row_inds * areas) / np.sum(areas)
            col_inds = np.sum(col_inds * areas) / np.sum(areas)
        elif gridcell_area is not None and grid is not None:
            grid_values = grid.isel(sel_dict).values
            row_inds = np.sum(row_points * grid_values * areas) / (
                np.sum(grid_values) * np.sum(areas)
            )
            col_inds = np.sum(col_points * grid_values * areas) / (
                np.sum(grid_values) * np.sum(areas)
            )
    else:
        row_inds = row_points / len(row_inds)
        col_inds = col_points / len(col_inds)
    center_row = np.round(np.sum(row_inds)).astype(int)
    center_col = np.round(np.sum(col_inds)).astype(int)

    return center_row, center_col, areas.sum()


def find_objects(box, mask):
    """Identifies objects found in the search region."""
    search_area = mask.values[
        box["row_min"] : box["row_max"], box["col_min"] : box["col_max"]
    ]
    objects = np.unique(search_area)
    return objects[objects != 0]


def empty_object_record():
    # Store records in "pixel" coordinates. Reconstruct flows in cartesian or geographic
    # coordinates as required.
    object_record = {
        "previous_ids": [],
        "matched_current_ids": [],
        "universal_ids": [],
        "flow_boxes": [],  # Extract box centers as required
        "search_boxes": [],  # Extract box centers as required
        "flows": [],
        "global_flows": [],
        "global_flow_boxes": [],
        "current_displacements": [],
        "previous_displacements": [],
        "previous_centers": [],  # These are gridcell area area weighted centers
        "matched_current_centers": [],  # These are gridcell area weighted centers
        "previous_weighted_centers": [],  # These are gridcell area and field weighted centers
        "matched_current_weighted_centers": [],  # These are gridcell area and field weighted centers
        "costs": [],  # Cost function in units of km
    }
    return object_record


def initialize_object_record(match_data, object_tracks, object_options):
    """Initialize record of object properties in previous and current masks."""

    previous_mask = get_masks(object_tracks, object_options)[1]
    total_previous_objects = np.max(previous_mask)
    previous_ids = np.arange(1, total_previous_objects + 1)

    universal_ids = np.arange(
        object_tracks["object_count"] + 1,
        object_tracks["object_count"] + total_previous_objects + 1,
    )
    object_tracks["object_count"] += total_previous_objects
    object_record = match_data.copy()
    object_record["previous_ids"] = previous_ids
    object_record["universal_ids"] = universal_ids
    object_tracks["object_record"] = object_record


def update_object_record(match_data, object_tracks, object_options):
    """Update record of object properties in previous and current masks."""

    previous_object_record = copy.deepcopy(object_tracks["object_record"])
    previous_mask = get_masks(object_tracks, object_options)[1]

    total_previous_objects = np.max(previous_mask.values)
    previous_ids = np.arange(1, total_previous_objects + 1)
    universal_ids = np.array([], dtype=int)

    for previous_id in np.arange(1, total_previous_objects + 1):
        # Check if object was matched in previous iteration
        if previous_id in previous_object_record["matched_current_ids"]:
            index = np.argwhere(
                previous_object_record["matched_current_ids"] == previous_id
            )
            index = index[0, 0]
            # Append the previously created universal id corresponding to previous_id
            universal_ids = np.append(
                universal_ids, previous_object_record["universal_ids"][index]
            )
        else:
            uid = object_tracks["object_count"] + 1
            object_tracks["object_count"] += 1
            universal_ids = np.append(universal_ids, uid)
            # Check if new object split from old?

    object_record = match_data.copy()
    object_record["universal_ids"] = universal_ids
    object_record["previous_ids"] = previous_ids
    object_tracks["object_record"] = object_record
