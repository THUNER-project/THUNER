"""Process ACCESS data."""

import os

# Check if system is unix-like, as xESMF is not supported on Windows
if os.name == "posix":
    import xesmf as xe
else:
    message = "Warning: Windows systems cannot run xESMF for regridding."
    message += "If you need regridding, consider using a Linux or MacOS system."
    print(message)

import copy
from urllib.parse import urlparse
import xarray as xr
import numpy as np
import pandas as pd
from typing import Literal
from pydantic import Field, model_validator
from thuner.log import setup_logger
import thuner.data._utils as _utils
import thuner.grid as grid
import thuner.utils as utils
from thuner.utils import DatetimeField

__all__ = ["AccessCOptions", "get_access_filepaths"]

logger = setup_logger(__name__)


class AccessCOptions(utils.BaseDatasetOptions):
    """Options for ACCESS-C OPS3 datasets."""

    def model_post_init(self, __context):
        """Use model_post_init to change default inherited values."""
        super().model_post_init(__context)
        url = "https://thredds.nci.org.au/thredds/catalog/wr45/"
        self._change_defaults(
            name="access",
            parent_remote=url,
            reuse_regridder=True,
            fields=["reflectivity"],
        )

    # Define additional fields for ACCESS
    run_start: DatetimeField = Field(..., description="Event start datetime.")
    domain: Literal["bn", "ph", "ad", "vt", "sy", "dn"] = Field(
        "dn", description="Model domain; brisbane, perth, adelaide, etc."
    )
    mode: Literal["fc", "an", "fcmm"] = Field("fcmm", description="Mode type.")
    levels: Literal["sfc", "ml"] = Field("sfc", description="Model level type.")
    filename: str = Field("radar_refl_1km.nc", description="Name of the file.")

    # Override get_filepaths and grid_from_dataset with ACCESS specific versions.
    def get_filepaths(self):
        return get_access_c_filepaths(self)

    def convert_dataset(self, time, filepath, track_options, grid_options):
        return convert_access(
            time=time,
            filepath=filepath,
            track_options=track_options,
            dataset_options=self,
            grid_options=grid_options,
        )

    @model_validator(mode="after")
    def _check_run_start(self):
        # Check that the time of run start is 0000, 0600, 1200, or 1800 UTC.
        valid_times = ["0000", "0600", "1200", "1800"]
        run_start_str = utils.format_time(self.run_start, filename_safe=True)
        if run_start_str[-4:] not in valid_times:
            raise ValueError(f"run_start must be one of {valid_times}.")
        return self

    @model_validator(mode="after")
    def _check_filepaths(self):
        if self.filepaths is None:
            logger.info("Generating ACCESS-C filepaths.")
            self.filepaths = get_access_c_filepaths(self)
        if self.filepaths is None:
            raise ValueError("filepaths not provided or badly formed.")
        return self


def get_access_c_filepaths(options: AccessCOptions):
    """
    Get ACCESS filepaths assuming same filepath structure as remote location.
    """

    filepaths = []
    base_url = utils.get_parent(options) + "/ops_aps3"
    # ACCESS-C only has 1 run per file. We will never match objects across distinct
    # runs, so we will only ever track one file at a time for access-c.
    time = pd.Timestamp(options.run_start)
    base_url += f"/access-{options.domain}/1"
    filepath = (
        f"{base_url}/{time.year:04}{time.month:02}{time.day:02}/"
        f"{time.hour:02}00/{options.mode}/{options.levels}/"
        f"{options.filename}"
    )
    filepaths = [filepath]

    return sorted(filepaths)


_names_dict = {
    "lat": "latitude",
    "lon": "longitude",
    "radar_refl_1km": "reflectivity",
    "maxcol_refl": "reflectivity",
}


def convert_access(
    time,
    filepath,
    track_options,
    dataset_options,
    grid_options,
):
    """
    Convert an ACCESS dataset to standard THUNER format.
    """
    time_str = utils.format_time(time, filename_safe=False)
    logger.info(f"Converting {dataset_options.name} dataset for time {time_str}.")

    try:
        access = xr.open_dataset(filepath, decode_timedelta=True)
    except:
        logger.error(f"Failed to open dataset at {filepath}.")
        raise
    filtered_names_dict = {
        k: v for k, v in _names_dict.items() if k in access.variables
    }
    access = access.rename(filtered_names_dict)
    access = access[dataset_options.fields]

    latitude, longitude = grid.infer_geographic_grid(grid_options, access)
    regridder = _utils.get_geographic_regridder(
        access, grid_options, dataset_options, latitude=latitude, longitude=longitude
    )
    ds = regridder(access)
    ds = _utils.copy_attributes(ds, access)

    if grid_options.domain_mask is None:
        domain_mask = np.ones_like(ds[dataset_options.fields[0]].isel(time=0, drop=True))
        domain_mask = xr.DataArray(
            domain_mask,
            coords={"latitude": latitude, "longitude": longitude},
            dims=("latitude", "longitude"),
            name="domain_mask",
        )
    else:
        domain_mask = grid_options.domain_mask
    ds["domain_mask"] = domain_mask
    utils.infer_grid_options(ds, grid_options)
    ds["longitude"] = ds["longitude"] % 360
    cell_areas = grid.get_cell_areas(grid_options)
    ds["gridcell_area"] = (["latitude", "longitude"], cell_areas)
    new_entries = {"units": "km^2", "standard_name": "area", "valid_min": 0}
    ds["gridcell_area"].attrs.update(new_entries)

    all_coords = utils.get_mask_boundary(ds["domain_mask"], grid_options)
    boundary_coords, simple_boundary_coords, boundary_mask = all_coords
    ds["boundary_mask"] = boundary_mask

    ds = _utils.apply_mask(ds, grid_options)
    # Mask reflectivity values below 0 dBZ, which are not meteorologically meaningful.
    if "reflectivity" in ds:
        ds["reflectivity"] = ds["reflectivity"].where(ds["reflectivity"] > 0)

    return ds, boundary_coords, simple_boundary_coords
