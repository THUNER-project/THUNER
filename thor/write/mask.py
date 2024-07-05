"""Methods for writing object masks."""

import glob
import shutil
import xarray as xr
from thor.utils import drop_time, format_time
from thor.log import setup_logger


logger = setup_logger(__name__)


def update(object_tracks, object_options, output_directory):
    """Write masks to file."""
    if not object_options["mask_options"]["save"]:
        return

    current_day = drop_time(object_tracks["current_mask"].time.values)
    try:
        previous_day = drop_time(object_tracks["previous_masks"][-1].time.values)
    except (AttributeError, IndexError):
        previous_day = None

    if object_options["tracking"]["method"] is None:
        mask_type = "current_mask"
    else:
        mask_type = "current_matched_mask"

    if previous_day is None or current_day == previous_day:
        object_tracks["mask_list"].append(object_tracks[mask_type])
    elif current_day > previous_day:
        write(
            object_tracks["mask_list"],
            object_options["name"],
            previous_day,
            output_directory,
        )
        # Empty mask_list after saving
        object_tracks["mask_list"] = []
    return


def write(mask_list, object_name, day, output_directory):
    """Write masks to file."""
    day_str = format_time(day, filename_safe=False, day_only=True)
    logger.debug(f"Writing {object_name} masks for {day_str}.")
    combine_attrs_options = ["identical", "drop_conflicts"]
    for option in combine_attrs_options:
        try:
            masks = xr.concat(mask_list, dim="time", combine_attrs=option)
        except ValueError:
            logger.warning(
                f"Mask concatenation failed for {object_name} on {day} "
                f"with combine_attrs='{option}'."
            )
            if option == combine_attrs_options[-1]:
                raise ValueError(
                    "All mask concatenation attempts failed. "
                    "Were masks created consistently?"
                )

    filepath = output_directory / f"masks/{object_name}/{format_time(day)[:8]}.nc"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    masks.to_netcdf(filepath)


def write_final(tracks, track_options, output_directory, time):
    """Write final masks to file."""

    for index, level_options in enumerate(track_options):
        for obj in level_options.keys():
            if not level_options[obj]["mask_options"]["save"]:
                continue
            write(
                tracks[index][obj]["mask_list"],
                obj,
                drop_time(time),
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
