"""Functions for analyzing objects."""

import copy
import numpy as np
import xarray as xr
from thuner.log import setup_logger
import thuner.match.utils as utils
import thuner.grid as thuner_grid

logger = setup_logger(__name__)


def get_object_area(obj, mask, gridcell_area, grid_options):
    """Get object area. Note gridcell_area is in km^2 by default."""
    row_inds, col_inds = np.where(mask == obj)
    row_points = xr.Variable("mask_points", row_inds)
    col_points = xr.Variable("mask_points", col_inds)
    if grid_options.name == "cartesian":
        areas = gridcell_area.isel(y=row_points, x=col_points).values
    elif grid_options.name == "geographic":
        areas = gridcell_area.isel(latitude=row_points, longitude=col_points).values
    return areas.sum()


def get_object_center(obj, mask, grid_options, gridcell_area=None, grid=None):
    """Get object centre."""
    coord_names = thuner_grid.get_coordinate_names(grid_options)
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

    if center_row < 0:
        center_row = 0
        print(center_row)

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

    previous_mask = utils.get_masks(object_tracks, object_options)[1]
    total_previous_objects = np.max(previous_mask)
    previous_ids = np.arange(1, total_previous_objects + 1)

    universal_ids = np.arange(
        object_tracks["object_count"] + 1,
        object_tracks["object_count"] + total_previous_objects + 1,
    )
    object_tracks["object_count"] += total_previous_objects
    object_record = match_data.copy()

    # Get the parents obtained during the previous step
    current_parents = match_data["current_parents"]
    parents = relabel_parents(previous_ids, current_parents, universal_ids)
    # If object record new, initialize previous parents as empty list for each object
    object_record["previous_parents"] = [[] for i in range(len(previous_ids))]
    object_record["current_parents"] = parents

    object_record["previous_ids"] = previous_ids
    object_record["universal_ids"] = universal_ids
    object_tracks["object_record"] = object_record


def relabel_parents(previous_ids, parents, universal_ids):
    """Relabel parents with universal id."""
    new_parents = []
    for object_parents in parents:
        new_object_parents = []
        for obj_id in object_parents:
            universal_obj_id = universal_ids[previous_ids == obj_id][0]
            new_object_parents.append(universal_obj_id)
        new_parents.append(new_object_parents)
    return new_parents


def update_object_record(match_data, object_tracks, object_options):
    """Update record of object properties in previous and current masks."""

    previous_object_record = copy.deepcopy(object_tracks["object_record"])
    previous_mask = utils.get_masks(object_tracks, object_options)[1]

    total_previous_objects = np.max(previous_mask.values)
    previous_ids = np.arange(1, total_previous_objects + 1)
    universal_ids = np.array([], dtype=int)

    for previous_id in previous_ids:
        # Check if object was matched in previous iteration
        if previous_id in previous_object_record["matched_current_ids"]:
            cond = previous_object_record["matched_current_ids"] == previous_id
            index = np.argwhere(cond)
            index = index[0, 0]
            # Append the previously created universal id corresponding to previous_id
            universal_id = previous_object_record["universal_ids"][index]
            universal_ids = np.append(universal_ids, universal_id)
        else:
            universal_id = object_tracks["object_count"] + 1
            object_tracks["object_count"] += 1
            universal_ids = np.append(universal_ids, universal_id)
            # The object is new, so remove from parents of other objects
            # current_parents = match_data["current_parents"]
            # for object_parents in current_parents:
            #     if previous_id in object_parents:
            #         object_parents.remove(previous_id)

    object_record = match_data.copy()
    object_record["universal_ids"] = universal_ids
    # Update parents
    object_record["previous_parents"] = previous_object_record["current_parents"]
    current_parents = object_record["current_parents"]
    current_parents = relabel_parents(previous_ids, current_parents, universal_ids)
    if len(object_record["previous_parents"]) != len(previous_ids):
        print("asdf")
    object_record["current_parents"] = current_parents
    object_record["previous_ids"] = previous_ids
    object_tracks["object_record"] = object_record
