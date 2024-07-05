"""Methods for creating bounding boxes."""

import numpy as np


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
