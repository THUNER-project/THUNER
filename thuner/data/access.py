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
from thuner.data.odim import convert_odim
import thuner.data._utils as _utils
import thuner.grid as grid
import thuner.utils as utils

__all__ = ["ACCESSOptions", "get_access_filepaths"]

logger = setup_logger(__name__)


class ACCESSOptions(utils.BaseDatasetOptions):
    """Options for ACCESS OPS3 datasets."""

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
    run_start: str | np.datetime64 = Field(..., description="Event start datetime.")
    model: Literal["g", "ge", "c"] = Field(
        "c", description="Model type; global, global ensemble, city etc."
    )
    domain: Literal["bn", "ph", "ad", "vt", "sy", "dn"] = Field(
        "dn", description="Model domain; brisbane, perth, adelaide, etc."
    )
    mode: Literal["fc", "an", "fcmm"] = Field("fcmm", description="Mode type.")
    levels: Literal["sfc", "ml"] = Field("sfc", description="Model level type.")
    # By default we only keep forecast data from 12 to 36 hours.
    # See Short & Lane (2023) https://doi.org/10.1175/MWR-D-23-0033.1.
    lead_start: int = Field(12, description="Lead time to start from in hours.")
    lead_end: int = Field(36, description="Lead time to end at in hours.")
    filename: str = Field("radar_refl_1km.nc", description="Name of the file.")

    # Override get_filepaths and grid_from_dataset with ACCESS specific versions.
    def get_filepaths(self):
        return get_access_filepaths(self)

    # def convert_dataset(self, time, filepath, track_options, grid_options):
    #     args = [time, filepath, track_options, self, grid_options]
    #     return convert_access(*args)

    # def update_boundary_data(self, dataset, input_record, boundary_coords):
    #     update_access_boundary_data(dataset, input_record, boundary_coords)

    @model_validator(mode="after")
    def _check_run_start(cls, values):
        # Check that the time of run start is 0000, 0600, 1200, or 1800 UTC.
        valid_times = ["0000", "0600", "1200", "1800"]
        run_start_str = utils.format_time(values.run_start, filename_safe=True)
        if run_start_str[-4:] not in valid_times:
            raise ValueError(f"run_start must be one of {valid_times}.")
        return values

    @model_validator(mode="after")
    def _check_filepaths(cls, values):
        if values.filepaths is None:
            logger.info("Generating access filepaths.")
            values.filepaths = get_access_filepaths(values)
        if values.filepaths is None:
            raise ValueError("filepaths not provided or badly formed.")
        return values


def get_access_filepaths(options: ACCESSOptions):
    """
    Get ACCESS filepaths assuming same filepath structure as remote location.
    """

    filepaths = []

    base_url = utils.get_parent(options) + "/ops_aps3"

    if options.model == "c":
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
    else:
        raise NotImplementedError(
            "Only ACCESS-C converters implemented. Convert manually first."
        )

    return sorted(filepaths)


_names_dict = {
    "lat": "latitude",
    "lon": "longitude",
    "radar_refl_1km": "reflectivity",
    "maxcol_refl": "max_col_refl",
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

    access = xr.open_dataset(filepath)
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
        # If no mask provided, use the whole grid.
        mask = np.zeros_like(ds[dataset_options.fields[0]].isel(time=0, drop=True))
        mask[1:-1, 1:-1] = 1
        mask = xr.DataArray(
            mask,
            coords={"latitude": latitude, "longitude": longitude},
            dims=("latitude", "longitude"),
            name="domain_mask",
        )
    ds["domain_mask"] = mask
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

    return ds, boundary_coords, simple_boundary_coords
