"""Methods for writing object masks."""

import glob
import shutil
import numpy as np
import xarray as xr
from thor.utils import format_time
from thor.log import setup_logger


logger = setup_logger(__name__)


def write(mask_list, object_options, last_write_time, write_interval, output_directory):
    """Write attributes to file."""
    object_name = object_options["name"]
    last_write_str = format_time(last_write_time, filename_safe=False, day_only=False)
    current_str = format_time(
        last_write_time + write_interval, filename_safe=False, day_only=False
    )
    message = f"Writing {object_name} masks from {last_write_str} to {current_str}, "
    message += f"inclusive and non-inclusive, respectively."
    logger.info(message)
    combine_attrs_options = ["identical", "drop_conflicts"]
    for option in combine_attrs_options:
        try:
            masks = xr.concat(mask_list, dim="time", combine_attrs=option)
        except ValueError:
            message = f"Mask concatenation failed for {object_name} between "
            message += f"{last_write_str} and {current_str} with "
            message += f"combine_attrs='{option}'."
            logger.warning(message)
            if option == combine_attrs_options[-1]:
                message = "All mask concatenation attempts failed. "
                message += "Were masks created consistently?"
                raise ValueError(message)

    filepath = output_directory / f"masks/{object_name}/"
    filepath = filepath / f"{format_time(last_write_str)}.nc"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    masks.to_netcdf(filepath)


def write_final(tracks, track_options, output_directory):
    """Write final masks to file."""

    for index, level_options in enumerate(track_options):
        for obj in level_options.keys():
            if not level_options[obj]["mask_options"]["save"]:
                continue
            write(
                tracks[index][obj]["mask_list"],
                level_options[obj],
                tracks[index][obj]["last_write_time"],
                np.timedelta64(level_options[obj]["write_interval"]),
                output_directory,
            )


def aggregate(track_options, output_directory, clean_up=True):
    """Aggregate masks into single file."""

    for level_options in track_options:
        for obj in level_options.keys():
            if not level_options[obj]["mask_options"]["save"]:
                continue
            filepaths = glob.glob(f"{output_directory}/masks/{obj}/*.nc")
            masks = xr.open_mfdataset(filepaths)
            masks.to_netcdf(output_directory / f"masks/{obj}_masks.nc")
            if clean_up:
                shutil.rmtree(output_directory / f"masks/{obj}")
