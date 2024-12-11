"""Functions for writing object masks."""

from memory_profiler import profile
import glob
import shutil
import numpy as np
import xarray as xr
import multiprocessing
from thuner.utils import format_time
from thuner.log import setup_logger


logger = setup_logger(__name__)


def write(object_tracks, object_options, output_directory):
    """Write masks to file."""

    if object_options.tracking is None:
        mask_type = "current_mask"
    else:
        mask_type = "current_matched_mask"
    mask = object_tracks[mask_type]
    if mask is None:
        return
    else:
        mask = mask.copy()
    mask = mask.expand_dims("time")

    object_name = object_options.name

    filepath = output_directory / f"masks/{object_name}.zarr"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    mask = mask.astype(np.uint32)
    coords = [c for c in mask.coords if c in ["x", "y", "latitude", "longitude"]]
    for coord in coords:
        mask.coords[coord] = mask.coords[coord].astype(np.float32)

    message = f"Writing {object_name} masks to {filepath}."
    logger.info(message)
    if not filepath.exists():
        mask.to_zarr(filepath, mode="w")
    else:
        mask.to_zarr(filepath, mode="a", append_dim="time")


def write_final(tracks, track_options, output_directory):
    """Write final masks to file."""

    for index, level_options in enumerate(track_options.levels):
        for object_options in level_options.objects:
            if not object_options.mask_options.save:
                continue
            obj_name = object_options.name
            write(tracks[index][obj_name], object_options, output_directory)
