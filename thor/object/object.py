"""Methods for analyzing objects."""

import numpy as np
import xarray as xr
from thor.utils import geodesic_distance


def get_bounding_box(obj, mask):
    """Get bounding box of object."""
    bounding_box = {}
    row_inds, col_inds = np.where(mask == obj)
    bounding_box["row_min"] = np.min(row_inds)
    bounding_box["row_max"] = np.max(row_inds)
    bounding_box["col_min"] = np.min(col_inds)
    bounding_box["col_max"] = np.max(col_inds)
    return bounding_box


def expand_bounding_box(bounding_box, row_margin, col_margin):
    """Expand bounding box by margins."""
    bounding_box["row_min"] = bounding_box["row_min"] - row_margin
    bounding_box["row_max"] = bounding_box["row_max"] + row_margin
    bounding_box["col_min"] = bounding_box["col_min"] - col_margin
    bounding_box["col_max"] = bounding_box["col_max"] + col_margin
    return bounding_box


def clip_bounding_box(bounding_box, dims):
    """Clip bounding box to image dimensions."""
    bounding_box["row_min"] = np.max([bounding_box["row_min"], 0])
    bounding_box["row_max"] = np.min([bounding_box["row_max"], dims[0]])
    bounding_box["col_min"] = np.max([bounding_box["col_min"], 0])
    bounding_box["col_max"] = np.min([bounding_box["col_max"], dims[1]])
    return bounding_box


def get_object_area(obj, mask, gridcell_area):
    """Get object area. Note gridcell_area is in km^2 by default."""
    row_inds, col_inds = np.where(mask == obj)
    row_points = xr.Variable("mask_points", row_inds)
    col_points = xr.Variable("mask_points", col_inds)
    areas = gridcell_area.isel(latitude=row_points, longitude=col_points).values
    return areas.sum()


def get_object_center(obj, mask, gridcell_area=None, grid=None):
    """Get object centre."""
    row_inds, col_inds = np.where(mask == obj)
    if gridcell_area is not None or grid is not None:
        row_points = xr.Variable("mask_points", row_inds)
        col_points = xr.Variable("mask_points", col_inds)
        if gridcell_area is not None:
            areas = gridcell_area.isel(latitude=row_points, longitude=col_points).values
            row_inds = np.sum(row_inds * areas) / np.sum(areas)
            col_inds = np.sum(col_inds * areas) / np.sum(areas)
        if grid is not None:
            grid_values = grid.isel(latitude=row_points, longitude=col_points).values
            row_inds = np.sum(row_points * grid_values) / np.sum(grid_values)
            col_inds = np.sum(col_points * grid_values) / np.sum(grid_values)
    else:
        row_points = row_points / len(row_inds)
        col_points = col_points / len(col_inds)
    center_row = np.round(np.sum(row_inds)).astype(int)
    center_col = np.round(np.sum(col_inds)).astype(int)
    latitudes = mask.latitude.values
    longitudes = mask.longitude.values
    center_lat = latitudes[center_row]
    center_lon = longitudes[center_col]
    return center_lat, center_lon


def find_objects(box, mask):
    """Identifies objects found in the search region."""

    search_area = mask.values[
        box["row_min"] : box["row_max"], box["col_min"] : box["col_max"]
    ]
    objects = np.unique(search_area)
    return objects[objects != 0]
