"""Functions for analyzing objects."""

import numpy as np
import xarray as xr
from thuner.log import setup_logger
import thuner.grid as thuner_grid

logger = setup_logger(__name__)


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
            row_inds = np.sum(row_points * grid_values * areas)
            row_inds /= np.sum(grid_values) * np.sum(areas)
            col_inds = np.sum(col_points * grid_values * areas)
            col_inds /= np.sum(grid_values) * np.sum(areas)
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
