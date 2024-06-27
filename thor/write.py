"""Methods for writing THOR outputs."""

import xarray as xr
from thor.utils import drop_time


def write_masks(object_tracks, object_options):
    """Write masks to file."""

    current_day = drop_time(object_tracks["current_mask"].time)
    try:
        previous_day = drop_time(object_tracks["previous_masks"][-1].time)
    except KeyError or IndexError:
        previous_day = None

    if previous_day is None or current_day == previous_day:
        object_tracks["mask_list"].append(object_tracks["current_mask"])
    elif current_day > previous_day:
        masks = xr.concat(object_tracks["mask_list"], dim="time")
        masks.to_netcdf(
            object_options["mask_options"]["output_directory"]
            + f"/mask_{current_day}.nc"
        )
    return
