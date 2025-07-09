"""Process Himawari data."""

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
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
from typing import Literal
from pydantic import Field, model_validator
from thuner.log import setup_logger
import thuner.data._utils as _utils
import thuner.grid as grid
import thuner.utils as utils

logger = setup_logger(__name__)

__all__ = ["HimawariOptions", "get_himawari_filepaths"]

Bands = Literal[
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B10",
    "B11",
    "B12",
    "B13",
    "B14",
    "B15",
    "B16",
]

names_dict = {
    "channel_0014_brightness_temperature": "brightness_temperature",
}


class HimawariOptions(utils.BaseDatasetOptions):
    """Class for Himawari dataset options."""

    def model_post_init(self, __context):
        """
        If unset by user, change default values inherited from the base class.
        """
        super().model_post_init(__context)
        url = "https://dapds00.nci.org.au/thredds/fileServer/ra22"
        kwargs = {"fields": ["brightness_temperature"], "parent_remote": url}
        kwargs.update({"name": "himawari", "reuse_regridder": True})
        self._change_defaults(**kwargs)

    # Define additional fields for Himawari
    _desc = "Time frame for the observation."
    time_frame: Literal["arc", "nrt"] = Field("arc", description=_desc)
    _desc = "Observation type."
    region: Literal["fldk", "re03"] = Field("fldk", description=_desc)
    _desc = "Version of the Himawari data."
    version: Literal["v1-0", "latest"] = Field("latest", description=_desc)
    _desc = "Instrument name."
    instrument: Literal["AHI"] = Field("AHI", description=_desc)
    _desc = "Observation EM band."
    band: Bands = Field("B14", description=_desc)
    _desc = "Resolution of the data, i.e. size of the gridstep."
    resolution: Literal[500, 1000, 2000] = Field(2000, description=_desc)
    _desc = "Path to the coordinates ancillary file."
    coordinates_filepath: str | None = Field(None, description=_desc)

    # Override get_filepaths and grid_from_dataset with CPOL specific versions.
    def get_filepaths(self):
        """Get Himawari fielpaths."""
        filepaths = get_himawari_filepaths(self)
        # Subset to just those files that actually exist locally
        filepaths = sorted([fp for fp in filepaths if Path(fp).exists()])
        return filepaths

    def convert_dataset(self, time, filepath, track_options, grid_options):
        """Convert Himawari dataset."""
        args = [time, filepath, track_options, self, grid_options]
        return convert_himawari(*args)

    def update_boundary_data(self, dataset, input_record, boundary_coords):
        """Update Himawari boundary data."""
        update_himawari_boundary_data(dataset, input_record, boundary_coords)

    @model_validator(mode="after")
    def _check_filepaths(cls, values):
        if values.filepaths is None:
            logger.info("Generating Himawari filepaths.")
            filepaths = get_himawari_filepaths(values)
            # Subset to just those files that actually exist locally
            filepaths = sorted([fp for fp in filepaths if Path(fp).exists()])
            values.filepaths = filepaths
        if values.filepaths is None:
            raise ValueError("filepaths not provided or badly formed.")
        if values.coordinates_filepath is None:
            logger.info("Generating coordinates filepath.")
            values.coordinates_filepath = get_himawari_coordinates_filepath(values)
        if values.coordinates_filepath is None:
            raise ValueError("coordinates_filepath not provided or badly formed.")

        return values


def get_himawari_coordinates_filepath(options: HimawariOptions):
    """
    Get the coordinates filepath for Himawari data.
    """

    filepath = utils.get_parent(options)
    filepath += f"/satellite-products/{options.time_frame}/obs/"
    filepath += f"himawari-{options.instrument.lower()}/{options.region}/"
    filepath += f"{options.version}/ancillary/00000000000000-P1S-ABOM_GEOM_"
    filepath += f"SENSOR-PRJ_GEOS141_{options.resolution}-HIMAWARI8-AHI.nc"

    return filepath


def get_himawari_filepaths(options: HimawariOptions):
    """
    Get Himawari filepaths based on the provided options.
    """

    start = np.datetime64(options.start).astype("datetime64[m]")
    end = np.datetime64(options.end).astype("datetime64[m]")

    filepaths = []

    base_filepath = utils.get_parent(options)
    base_filepath += f"/satellite-products/{options.time_frame}/obs/"
    base_filepath += f"himawari-{options.instrument.lower()}/{options.region}/"
    base_filepath += f"{options.version}/"

    times = np.arange(start, end + np.timedelta64(10, "m"), np.timedelta64(10, "m"))
    times = pd.DatetimeIndex(times)
    for time in times:
        satellite = "HIMAWARI8" if time < pd.Timestamp("2022-12-13") else "HIMAWARI9"
        filepath = (
            f"{base_filepath}{time.year:04d}/{time.month:02d}/{time.day:02d}/"
            f"{time.hour:02d}{time.minute:02d}/"
            f"{time.year:04d}{time.month:02d}{time.day:02d}{time.hour:02d}"
            f"{time.minute:02d}00-P1S-ABOM_OBS_{options.band}-PRJ_GEOS141_"
            f"{options.resolution}-{satellite}-{options.instrument}.nc"
        )
        filepaths.append(filepath)
    return sorted(filepaths)


def get_himawari_ancillary_filepaths(options: HimawariOptions):
    """
    Get ancillary filepaths for Himawari data.
    """
    filepaths = []
    base_filepath = utils.get_parent(options)
    base_filepath += f"/satellite-products/{options.time_frame}/obs/"
    base_filepath += f"himawari-{options.instrument.lower()}/{options.region}/"
    base_filepath += f"{options.version}/ancillary/00000000000000-P1S-ABOM_GEOM_"
    for file_type in ["AUSDEM", "LAND", "SENSOR"]:
        for resolution in [500, 1000, 2000]:
            filepath = base_filepath + f"{file_type}-PRJ_GEOS141_{resolution}"
            filepath += "-HIMAWARI8-AHI.nc"
            filepaths.append(filepath)
    return sorted(filepaths)


def convert_himawari(
    time,
    filepath,
    track_options,
    dataset_options,
    grid_options,
):
    """
    Convert a Himawari dataset to a standard format.
    """
    time_str = utils.format_time(time, filename_safe=False)
    logger.info(f"Converting {dataset_options.name} dataset for time {time_str}.")

    himawari = xr.open_dataset(filepath)
    himawari = himawari.rename(names_dict)
    himawari = himawari[dataset_options.fields]
    coordinates = xr.open_dataset(dataset_options.coordinates_filepath)
    coord_names = ["lat", "lon", "invalid_navigation_mask"]
    coordinates = coordinates[coord_names].isel(time=0, drop=True)
    himawari.coords["latitude"] = coordinates["lat"]
    himawari.coords["longitude"] = coordinates["lon"]
    mask = np.logical_not(coordinates["invalid_navigation_mask"]).astype(np.float32)
    himawari["domain_mask"] = mask
    min_lat = grid_options.latitude[0]
    max_lat = grid_options.latitude[-1]
    min_lon = grid_options.longitude[0]
    max_lon = grid_options.longitude[-1]
    himawari = grid.subset_curvilinear(himawari, min_lat, max_lat, min_lon, max_lon)
    mask = np.isnan(himawari["latitude"]) | np.isnan(himawari["longitude"])
    himawari = himawari.where(~mask)

    logger.info("Regridding Himawari data.")
    regridder = _utils.get_geographic_regridder(himawari, grid_options, dataset_options)
    ds = regridder(himawari)
    ds = _utils.copy_attributes(ds, himawari)

    ds["longitude"] = ds["longitude"] % 360
    # Update grid_options if necessary
    utils.infer_grid_options(ds, grid_options)
    cell_areas = grid.get_cell_areas(grid_options)
    ds["gridcell_area"] = (["latitude", "longitude"], cell_areas)
    new_entries = {"units": "km^2", "standard_name": "area", "valid_min": 0}
    ds["gridcell_area"].attrs.update(new_entries)

    ds["domain_mask"] = _utils.smooth_mask(ds["domain_mask"].astype(int))
    all_coords = utils.get_mask_boundary(ds["domain_mask"], grid_options)
    boundary_coords, simple_boundary_coords, boundary_mask = all_coords
    ds["boundary_mask"] = boundary_mask

    ds = _utils.apply_mask(ds, grid_options)

    return ds, boundary_coords, simple_boundary_coords


def update_himawari_boundary_data(dataset, input_record, boundary_coords):
    """
    Update Himawari boundary data.
    """
    """Update boundary data using domain mask."""
    # Set data outside instrument range to NaN
    keys = ["next_domain_mask", "next_boundary_coordinates"]
    keys += ["next_boundary_mask"]
    if any(getattr(input_record, k) is None for k in keys):
        # Get the domain mask and domain boundary. Note this is the region where data
        # exists, not the detected object masks from the detect module.
        input_record.next_domain_mask = dataset["domain_mask"]
        input_record.next_boundary_coordinates = boundary_coords
        input_record.next_boundary_mask = dataset["boundary_mask"]
    else:
        domain_mask = copy.deepcopy(input_record.next_domain_mask)
        boundary_mask = copy.deepcopy(input_record.next_boundary_mask)
        boundary_coords = copy.deepcopy(input_record.next_boundary_coordinates)
        input_record.domain_masks.append(domain_mask)
        input_record.boundary_coodinates.append(boundary_coords)
        input_record.boundary_masks.append(boundary_mask)
        # Note for Himawari data the domain mask is calculated using a static file
        # which is constant for all times. Therefore, the mask is not updated for each
        # new file. Contrast this with, for instance, GridRad, where a
        # new mask is calculated for each time step based on the altitudes of the
        # objects being detected, and the required threshold on number of observations.
