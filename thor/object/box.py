"""Methods for creating bounding boxes."""

import numpy as np
from thor.utils import geodesic_forward
from thor.log import setup_logger

logger = setup_logger(__name__)


def get_center(box):
    """Get the center indices of a box."""
    row = int(np.round((box["row_min"] + box["row_max"]) / 2))
    col = int(np.round((box["col_min"] + box["col_max"]) / 2))
    return [row, col]


def create_box(row_min, row_max, col_min, col_max):
    """Create a box dictionary."""
    box = {}
    box["row_min"] = row_min
    box["row_max"] = row_max
    box["col_min"] = col_min
    box["col_max"] = col_max
    return box


def get_bounding_box(obj, mask):
    """Get bounding box of object."""
    bounding_box = {}
    row_inds, col_inds = np.where(mask == obj)
    bounding_box["row_min"] = np.min(row_inds)
    bounding_box["row_max"] = np.max(row_inds)
    bounding_box["col_min"] = np.min(col_inds)
    bounding_box["col_max"] = np.max(col_inds)
    return bounding_box


def expand_box(box, row_margin, col_margin):
    """Expand bounding box by margins."""
    box["row_min"] = box["row_min"] - row_margin
    box["row_max"] = box["row_max"] + row_margin
    box["col_min"] = box["col_min"] - col_margin
    box["col_max"] = box["col_max"] + col_margin
    return box


def shift_box(box, row_shift, col_shift):
    """Expand bounding box by margins."""
    box["row_min"] = box["row_min"] + row_shift
    box["row_max"] = box["row_max"] + row_shift
    box["col_min"] = box["col_min"] + col_shift
    box["col_max"] = box["col_max"] + col_shift
    return box


def clip_box(box, dims):
    """Clip bounding box to image dimensions."""
    box["row_min"] = np.max([box["row_min"], 0])
    box["row_max"] = np.min([box["row_max"], dims[0] - 1])
    box["col_min"] = np.max([box["col_min"], 0])
    box["col_max"] = np.min([box["col_max"], dims[1] - 1])
    return box


def get_search_box(box, flow, latitudes, longitudes, search_margin, spacing, shape):
    row_margin, col_margin = get_gridcell_margins(
        box, latitudes, longitudes, search_margin, spacing
    )
    search_box = box.copy()
    search_box = expand_box(search_box, row_margin, col_margin)
    search_box = shift_box(search_box, flow[0], flow[1])
    search_box = clip_box(search_box, shape)
    return search_box


def get_unique_global_flow_box(
    latitudes, longitudes, global_flow_margin, grid_spacing, shape
):
    """Get the bounding box for global flow."""
    row = int(np.floor(len(latitudes) / 2))
    col = int(np.floor(len(longitudes) / 2))
    lat = latitudes[row]
    lon = longitudes[col]
    end_lon = geodesic_forward(lon, lat, 90, global_flow_margin * 1e3)[0]
    radius_lon = (end_lon - lon) % 360
    if lat < 0:
        end_lon, end_lat = geodesic_forward(lon, lat, 0, global_flow_margin * 1e3)[:2]
    else:
        end_lon, end_lat = geodesic_forward(lon, lat, 180, global_flow_margin * 1e3)[:2]
    radius_lat = np.abs(end_lat - lat)
    radius_row = int(np.ceil(radius_lat / grid_spacing[0]))
    radius_col = int(np.ceil(radius_lon / grid_spacing[1]))
    global_flow_box = create_box(
        row - radius_row, row + radius_row, col - radius_col, col + radius_col
    )
    global_flow_box = clip_box(global_flow_box, shape)
    box_spans_grid = (
        global_flow_box["row_min"] == 0
        and global_flow_box["row_max"] == len(latitudes) - 1
        and global_flow_box["col_min"] == 0
        and global_flow_box["row_min"] == len(longitudes) - 1
    )
    if not box_spans_grid:
        logger.warning("Unique global flow box does not span entire grid.")
    return global_flow_box


def get_box_coords(box, latitudes, longitudes):
    """Get the coordinates of a box."""
    box_latitudes = [
        latitudes[box["row_min"]],
        latitudes[box["row_min"]],
        latitudes[box["row_max"]],
        latitudes[box["row_max"]],
        latitudes[box["row_min"]],
    ]
    box_longitudes = [
        longitudes[box["col_min"]],
        longitudes[box["col_max"]],
        longitudes[box["col_max"]],
        longitudes[box["col_min"]],
        longitudes[box["col_min"]],
    ]
    return box_latitudes, box_longitudes


def get_box_center_coords(box, latitudes, longitudes):
    """Get the coordinates of the center of a box."""
    center_row = int(np.ceil((box["row_min"] + box["row_max"]) / 2))
    center_col = int(np.ceil((box["col_min"] + box["col_max"]) / 2))
    center_lat = latitudes[center_row]
    center_lon = longitudes[center_col]
    return center_lat, center_lon, center_row, center_col


def get_gridcell_margins(
    bounding_box, latitudes, longitudes, flow_margin, grid_spacing
):
    [row, col] = get_center(bounding_box)
    box_lat = latitudes[row]
    box_lon = longitudes[col] % 360
    # Avoid calculating forward geodesic over 360 degrees lon
    if box_lon > 180:
        end_lon = geodesic_forward(box_lon, box_lat, 270, flow_margin * 1e3)[0] % 360
        margin_lon = box_lon - end_lon
    else:
        end_lon = geodesic_forward(box_lon, box_lat, 90, flow_margin * 1e3)[0] % 360
        margin_lon = end_lon - box_lon
    end_lon, end_lat = geodesic_forward(box_lon, box_lat, 0, flow_margin * 1e3)[:2]
    end_lon = end_lon % 360
    # While its unlikely thor will be used in the arctic, adjust end_lat if pole crossed
    if end_lon != box_lon:
        end_lat = 90
    margin_lat = end_lat - box_lat
    flow_margin_row = int(np.ceil(margin_lat / grid_spacing[0]))
    flow_margin_col = int(np.ceil(margin_lon / grid_spacing[1]))
    return flow_margin_row, flow_margin_col
