"""Functions for creating bounding boxes."""

import numpy as np
from thuner.log import setup_logger
import thuner.grid as grid

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


def get_search_box(box, flow, search_margin, grid_options):
    """Get the search box associated with a given a bounding box."""
    row_margin, col_margin = get_margins_pixels(box, search_margin, grid_options)
    search_box = box.copy()
    search_box = expand_box(search_box, row_margin, col_margin)
    search_box = shift_box(search_box, flow[0], flow[1])
    search_box = clip_box(search_box, grid_options.shape)
    return search_box


def get_geographic_box_coords(box, grid_options):
    """Get the geographic coordinates of a box."""
    lats = np.array(grid_options.latitude)
    lons = np.array(grid_options.longitude)
    row_list = ["min", "min", "max", "max", "min"]
    col_list = ["min", "max", "max", "min", "min"]
    if grid_options.name == "geographic":
        box_lats = [lats[box[f"row_{l}"]] for l in row_list]
        box_lons = [lons[box[f"col_{l}"]] for l in col_list]
    elif grid_options.name == "cartesian":
        box_lats = [lats[box[f"row_{l}"], box[f"col_{l}"]] for l in row_list]
        box_lons = [lons[box[f"row_{l}"], box[f"col_{l}"]] for l in col_list]
    else:
        raise ValueError("Grid name must be 'cartesian' or 'geographic'.")

    return box_lats, box_lons


def get_box_center_coords(box, grid_options):
    """Get the coordinates of the center of a box."""
    center_row = int(np.ceil((box["row_min"] + box["row_max"]) / 2))
    center_col = int(np.ceil((box["col_min"] + box["col_max"]) / 2))

    row_coords, col_coords = grid.get_horizontal_coordinates(grid_options)
    center_row_coord = row_coords[center_row]
    center_col_coord = col_coords[center_col]
    return center_row_coord, center_col_coord, center_row, center_col


def get_margins_pixels(bounding_box, flow_margin, grid_options):
    """Get box margins in gridcell i.e "pixel" coordinates."""

    if grid_options.name == "cartesian":
        grid_spacing = grid_options.cartesian_spacing
        flow_margin_row = int(np.ceil(flow_margin * 1e3 / grid_spacing[0]))
        flow_margin_col = int(np.ceil(flow_margin * 1e3 / grid_spacing[1]))
    elif grid_options.name == "geographic":
        latitudes = grid_options.latitude
        longitudes = grid_options.longitude
        [row, col] = get_center(bounding_box)
        box_lat = latitudes[row]
        box_lon = longitudes[col] % 360
        flow_margin_row, flow_margin_col = get_geographic_margins(
            box_lat, box_lon, flow_margin, grid_options
        )
    else:
        raise ValueError("Grid name must be 'cartesian' or 'geographic'.")
    return flow_margin_row, flow_margin_col


def get_geographic_margins(lat, lon, flow_margin, grid_options):
    """Get box margins in geographic coordinates."""
    spacing = grid_options.geographic_spacing
    # Avoid calculating forward geodesic over +/- 90 degrees lat
    if lat < 0:
        end_lat = grid.geodesic_forward(lon, lat, 0, flow_margin * 1e3)[1]
    else:
        end_lat = grid.geodesic_forward(lon, lat, 180, flow_margin * 1e3)[1]
    margin_row = int(np.ceil(np.abs(end_lat - lat) / spacing[0]))
    # Avoid calculating forward geodesic over 0 or 360 degrees lon
    lon = lon % 360
    if lon > 180:
        end_lon = grid.geodesic_forward(lon, lat, 270, flow_margin * 1e3)[0] % 360
    else:
        end_lon = grid.geodesic_forward(lon, lat, 90, flow_margin * 1e3)[0] % 360
    margin_col = int(np.ceil(np.abs(end_lon - lon) / spacing[1]))
    return margin_row, margin_col
