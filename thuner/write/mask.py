"""Write object masks to the unified zarr store."""

import numpy as np

from thuner.log import setup_logger
from thuner.utils import store_path
import thuner.write.attribute as attribute
import thuner.write.utils as utils


logger = setup_logger(__name__)


def write(object_tracks, object_options, output_directory):
    """Write a mask to the unified zarr store under ``masks/<object_name>``."""

    if object_options.tracking is None:
        mask_type = "next_mask"
    else:
        mask_type = "next_matched_mask"
    mask = getattr(object_tracks, mask_type)

    if mask is None:
        return
    mask = mask.copy()
    mask = mask.expand_dims("time")

    object_name = object_options.name
    store = store_path(output_directory)
    group = f"masks/{object_name}"
    store.parent.mkdir(parents=True, exist_ok=True)

    mask = mask.astype(np.uint32)
    coords = [c for c in mask.coords if c in ["x", "y", "latitude", "longitude"]]
    for coord in coords:
        mask.coords[coord] = mask.coords[coord].astype(np.float32)
    utils.pin_datetime_encoding(mask)

    message = f"Writing {object_name} masks to {store}::{group}."
    logger.info(message)
    if utils.zarr_group_exists(store, group):
        mask.to_zarr(store, group=group, mode="a", append_dim="time")
    else:
        mask.to_zarr(store, group=group, mode="a")